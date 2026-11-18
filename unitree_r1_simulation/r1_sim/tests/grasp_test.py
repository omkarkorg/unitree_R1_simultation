#!/usr/bin/env python3

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/opt/xr_teleoperate")
os.chdir("/opt/xr_teleoperate/teleop")

import pinocchio as pin
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

OBJECT = "green_cylinder"
HAND_OPEN = np.zeros(6)
HAND_THUMB_OUT = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
FINGERS_CLOSED = 0.6
THUMB_FLEX = 0.3
IN_PALM = np.array([0.053, -0.010, 0.080])
SIDE_GAP = 0.08
PREGRASP_OFFSET = np.array([-0.005, -0.118, -0.052])
PALM_IN_WRIST_POS = np.array([0.0956, 0.0, 0.0])
PALM_IN_WRIST_ROT = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
LIFT_HEIGHT = 0.10
IK_RATE = 50.0


def log(msg):
    print(f"[grasp_test] {msg}", flush=True)


def smoothstep(s):
    return s * s * (3.0 - 2.0 * s)


def quat_matrix(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tilt_deg(q_xyzw):
    x, y, _, _ = q_xyzw
    return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


class SimState:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self._sub = ChannelSubscriber("rt/sim_state", String_)
        self._sub.Init(self._on_msg, 10)

    def _on_msg(self, msg):
        data = json.loads(msg.data)
        with self._lock:
            self._latest = data

    def get(self, timeout=0.0):
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._latest is not None:
                    return self._latest
            if time.time() >= deadline:
                return None
            time.sleep(0.05)

    def object_in_pelvis_frame(self, name):
        state = self.get()
        robot, obj = np.array(state["robot"]), np.array(state["objects"][name])
        return quat_matrix(robot[3:]).T @ (obj[:3] - robot[:3]), obj

    def object_in_right_palm(self, name):
        state = self.get()
        robot, wrist = np.array(state["robot"]), np.array(state["wrists"]["right"])
        obj = np.array(state["objects"][name])
        rot_wrist = quat_matrix(wrist[3:])
        palm_pos = wrist[:3] + rot_wrist @ PALM_IN_WRIST_POS
        rot_palm = rot_wrist @ PALM_IN_WRIST_ROT
        return rot_palm.T @ (obj[:3] - palm_pos), quat_matrix(robot[3:]).T @ rot_palm


class RightHand:
    def __init__(self, q):
        self._q = np.array(q, dtype=float)
        self._lock = threading.Lock()
        self._state = None
        self._sub = ChannelSubscriber("rt/brainco/right/state", MotorStates_)
        self._sub.Init(self._on_state, 10)
        self._pub = ChannelPublisher("rt/brainco/right/cmd", MotorCmds_)
        self._pub.Init()
        self._msg = MotorCmds_()
        self._msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
        for cmd in self._msg.cmds:
            cmd.dq = 1.0
        threading.Thread(target=self._loop, daemon=True).start()

    def set(self, q):
        with self._lock:
            self._q = np.array(q, dtype=float)

    def move(self, q_goal, duration):
        with self._lock:
            q_start = self._q.copy()
        n = max(1, int(duration * 100))
        for i in range(1, n + 1):
            self.set(q_start + (np.asarray(q_goal) - q_start) * smoothstep(i / n))
            time.sleep(0.01)

    def _on_state(self, msg):
        with self._lock:
            self._state = np.array([s.q for s in msg.states])

    def wait_for(self, motor, value, timeout):
        t_end = time.time() + timeout
        while time.time() < t_end:
            with self._lock:
                state = self._state
            if state is not None and state[motor] >= value:
                break
            time.sleep(0.01)
        return None if state is None else float(state[motor])

    def _loop(self):
        while True:
            with self._lock:
                q = self._q.copy()
            for i in range(6):
                self._msg.cmds[i].q = float(q[i])
            self._pub.Write(self._msg)
            time.sleep(0.01)


class Frames:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.client = None
        if out_dir:
            from teleimager.image_client import ImageClient
            os.makedirs(out_dir, exist_ok=True)
            self.client = ImageClient(host="127.0.0.1", request_bgr=True)

    def save(self, name):
        if self.client is None:
            return
        import cv2
        for _ in range(40):
            frame = self.client.get_head_frame().bgr
            if frame is not None:
                path = os.path.join(self.out_dir, f"{name}.jpg")
                cv2.imwrite(path, frame)
                log(f"saved {path}")
                return
            time.sleep(0.05)

    def close(self):
        if self.client is not None:
            self.client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description='Scripted pick test for the Isaac Sim simulator (run_sim.py); no headset needed.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interface", default=os.environ.get("DDS_INTERFACE", "lo"))
    parser.add_argument("--domain", type=int, default=int(os.environ.get("DDS_DOMAIN_ID", "1")))
    parser.add_argument("--min-lift", type=float, default=0.05, help="required cylinder lift [m]")
    parser.add_argument("--save-frames", default="", help="directory for head camera snapshots")
    parser.add_argument("--in-palm", type=float, nargs=3, default=IN_PALM.tolist(),
                        help="cylinder centre in the palm frame at the grasp [m] (normal, thumb side, fingers)")
    parser.add_argument("--side-gap", type=float, default=SIDE_GAP, help="pre-grasp distance along the palm normal [m]")
    parser.add_argument("--fingers", type=float, default=FINGERS_CLOSED, help="finger closure 0..1")
    parser.add_argument("--thumb-flex", type=float, default=THUMB_FLEX, help="thumb flexion 0..1")
    args = parser.parse_args()
    if args.domain != 1:
        log("refusing to run outside DDS domain 1 (the simulation domain)")
        return 2

    ChannelFactoryInitialize(args.domain, args.interface)
    sim = SimState()
    if sim.get(timeout=10.0) is None:
        log("FAIL: no rt/sim_state within 10 s - start the Isaac Sim simulator first "
            "(./scripts/run_sim.sh); the mock simulator has no physics")
        return 1
    if "wrists" not in sim.get():
        log("FAIL: rt/sim_state has no wrist poses - the simulator is older than this test")
        return 1

    reset_pub = ChannelPublisher("rt/reset_pose/cmd", String_)
    reset_pub.Init()

    from teleop.robot_control.robot_arm import R1_A5_ArmController
    from teleop.robot_control.robot_arm_ik import R1_A5_ArmIK

    frames = Frames(args.save_frames)
    hand = RightHand(HAND_OPEN)
    arm_ik = R1_A5_ArmIK()
    arm = R1_A5_ArmController(motion_mode=False, simulation_mode=True)

    model = arm_ik.reduced_robot.model
    data = model.createData()
    pin.framesForwardKinematics(model, data, np.zeros(model.nq))
    left_T0 = data.oMf[arm_ik.L_hand_id].homogeneous.copy()
    right_T0 = data.oMf[arm_ik.R_hand_id].homogeneous.copy()
    right_target = right_T0[:3, 3].copy()

    for _ in range(10):
        q_hold, tau_hold = arm_ik.solve_ik(left_T0, right_T0, np.zeros(model.nq), np.zeros(model.nq))
    arm.ctrl_dual_arm(q_hold, tau_hold)
    time.sleep(2.0)
    reset_pub.Write(String_(data="1"))
    time.sleep(1.5)

    def move_right(goal_pos, duration):
        nonlocal right_target
        start = right_target.copy()
        n = max(1, int(duration * IK_RATE))
        t_next = time.time()
        for i in range(1, n + 1):
            target = right_T0.copy()
            target[:3, 3] = start + (goal_pos - start) * smoothstep(i / n)
            q, tau = arm_ik.solve_ik(left_T0, target, arm.get_current_dual_arm_q(), arm.get_current_dual_arm_dq())
            arm.ctrl_dual_arm(q, tau)
            t_next += 1.0 / IK_RATE
            time.sleep(max(0.0, t_next - time.time()))
        right_target = np.array(goal_pos, dtype=float)

    cyl_p, cyl0 = sim.object_in_pelvis_frame(OBJECT)
    log(f"cylinder at {np.round(cyl0[:3], 3)} (world), tilt {tilt_deg(cyl0[3:]):.1f} deg")
    frames.save("0_start")

    in_palm = np.array(args.in_palm)
    pre_in_palm = in_palm + np.array([args.side_gap, 0.0, 0.0])
    log(f"pre-grasp: open hand next to the cylinder ({args.side_gap * 100:.0f} cm out)")
    move_right(cyl_p + PREGRASP_OFFSET, 3.0)
    for _ in range(4):
        time.sleep(0.8)
        seen, rot_palm = sim.object_in_right_palm(OBJECT)
        error = seen - pre_in_palm
        log(f"  cylinder in the palm frame {np.round(seen * 100, 1)} cm, error {np.linalg.norm(error) * 100:.1f} cm")
        if np.linalg.norm(error) < 0.004:
            break
        move_right(right_target + rot_palm @ error, 0.6)
    frames.save("1_pregrasp")

    log("swing the thumb out, slide the palm onto the cylinder")
    hand.move(HAND_THUMB_OUT, 0.6)
    thumb = hand.wait_for(0, 0.95, timeout=3.0)
    log(f"  thumb rotation {thumb:.2f} (0 in palm .. 1 swung out)" if thumb is not None else "  no hand state")
    time.sleep(0.2)
    _, rot_palm = sim.object_in_right_palm(OBJECT)
    move_right(right_target + rot_palm[:, 0] * args.side_gap, 1.5)
    time.sleep(0.5)
    seen, _ = sim.object_in_right_palm(OBJECT)
    log(f"  cylinder in the palm frame {np.round(seen * 100, 1)} cm (target {np.round(in_palm * 100, 1)})")
    frames.save("2_in_hand")

    log("grasp: close the fingers, then wrap the thumb")
    f = args.fingers
    hand.move(np.array([1.0, 0.0, f, f, f, f]), 0.8)
    time.sleep(0.3)
    hand.move(np.array([1.0, args.thumb_flex, f, f, f, f]), 0.5)
    time.sleep(0.8)
    frames.save("3_grasped")

    log(f"lift: {LIFT_HEIGHT * 100:.0f} cm")
    grasp_target = right_target.copy()
    move_right(grasp_target + np.array([0.0, 0.0, LIFT_HEIGHT]), 2.0)
    time.sleep(1.5)
    frames.save("4_lifted")

    _, cyl1 = sim.object_in_pelvis_frame(OBJECT)
    lift = float(cyl1[2] - cyl0[2])
    log(f"cylinder now at {np.round(cyl1[:3], 3)} (world), lifted {lift * 100:.1f} cm, "
        f"tilt {tilt_deg(cyl1[3:]):.1f} deg")
    ok = lift >= args.min_lift

    log("put it down, release and go home")
    move_right(grasp_target + np.array([0.0, 0.0, 0.01]), 1.5)
    hand.move(HAND_THUMB_OUT, 0.5)
    _, rot_palm = sim.object_in_right_palm(OBJECT)
    move_right(right_target - rot_palm[:, 0] * args.side_gap, 1.0)
    hand.move(HAND_OPEN, 0.5)
    arm.ctrl_dual_arm_go_home()
    reset_pub.Write(String_(data="1"))
    frames.close()

    print(f"\n==== grasp test: {'PASS' if ok else 'FAIL'} - cylinder lifted {lift * 100:.1f} cm "
          f"(required {args.min_lift * 100:.1f} cm) ====", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
