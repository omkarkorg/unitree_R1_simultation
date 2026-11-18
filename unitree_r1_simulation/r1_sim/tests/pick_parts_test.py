#!/usr/bin/env python3

import argparse
import ast
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/xr_teleoperate")
os.chdir("/opt/xr_teleoperate/teleop")

import pinocchio as pin
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/opt/r1_sim")
from grasp_test import RightHand, SimState, log, quat_matrix, smoothstep

URDF = "/opt/r1_model/r1_brainco.urdf"
PARTS = ("peg_round", "peg_square", "gear_small", "gear_medium", "ring_small", "ring_medium", "nut")
HAND_OPEN = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
HAND_CLOSED = np.array([0.8, 0.45, 0.65, 0.65, 0.65, 0.65])
FINGERS_DOWN = np.radians(55.0)
STAND = (0.12, 0.26)
HEIGHT, LEAN = 0.66, 0.30
TABLE_Z = 0.58
CLEAR_X = 0.15


def load_geometry():
    ns = {"np": np}
    tree = ast.parse(open("/opt/xr_teleoperate/teleop/teleop_hand_and_arm.py").read())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ("palm_frame", "RobotGeometry"):
            exec(compile(ast.Module([node], []), "teleop", "exec"), ns)
    return ns["RobotGeometry"](URDF, [0.095, 0.0, 0.0])


def hand_points(geometry) -> tuple:
    from r1_sim.joint_map import HAND_MOTOR_JOINTS, HAND_COUPLED_JOINTS, hand_motor_to_joint_positions
    model = pin.buildModelFromUrdf(URDF)
    data = model.createData()
    names = [f"right_{j}_joint" for j in (*HAND_MOTOR_JOINTS, *HAND_COUPLED_JOINTS)]

    def tips(motors):
        q = pin.neutral(model)
        for name, value in zip(names, hand_motor_to_joint_positions(motors)):
            q[model.joints[model.getJointId(name)].idx_q] = value
        pin.framesForwardKinematics(model, data, q)
        to_palm = np.linalg.inv(data.oMf[model.getFrameId("right_wrist_roll_link")].homogeneous @ geometry.palm_in_wrist["right"])
        return {f: (to_palm @ data.oMf[model.getFrameId(f"right_{f}_tip_link")].homogeneous)[:3, 3]
                for f in ("thumb", "index", "middle", "ring", "pinky")}
    closing = tips(HAND_OPEN + 0.62 * (HAND_CLOSED - HAND_OPEN))
    return 0.5 * closing["thumb"] + 0.25 * (closing["index"] + closing["middle"]), np.array(list(tips(HAND_OPEN).values()))


def pelvis_matrix(x, y, h, yaw, pitch):
    c, s, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    T[:3, 3] = [x, y, h]
    return T


def main() -> int:
    parser = argparse.ArgumentParser(description="Picking the kit's parts off the side table, the way an operator does (Isaac Sim, the placed body; no headset).")
    parser.add_argument("--parts", nargs="*", default=list(PARTS))
    parser.add_argument("--error", type=float, default=0.01, help="m: the hand's random placement error (each axis, up to)")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--min", type=int, default=0, help="fail unless at least this many parts stay in the hand")
    a = parser.parse_args()
    ChannelFactoryInitialize(1, "lo")
    sim = SimState()
    if sim.get(timeout=10.0) is None:
        log("FAIL: no rt/sim_state - start the simulator first")
        return 1
    from teleop.robot_control.robot_arm import R1_A5_ArmController
    from teleop.robot_control.robot_arm_ik import R1_A5_ArmIK
    geometry = load_geometry()
    pinch, open_tips = hand_points(geometry)
    log(f"pinch point in the palm frame {np.round(pinch, 3)}")
    arm_ik = R1_A5_ArmIK(ee_placement=geometry.palm_in_wrist, cost_weights=(5000.0, 1.0, 0.02, 0.1), filter_weights=(1.0,))
    model, data = arm_ik.reduced_robot.model, arm_ik.reduced_robot.model.createData()
    hand = RightHand(HAND_OPEN)
    arm = R1_A5_ArmController(motion_mode=True, simulation_mode=True)
    body_pub = ChannelPublisher("rt/sim/body_cmd", String_)
    body_pub.Init()
    reset_pub = ChannelPublisher("rt/reset_pose/cmd", String_)
    reset_pub.Init()
    rng = np.random.default_rng(3)
    pelvis = {"pose": [0.0, 0.0, 0.0, 0.725, 0.0]}

    def send_body(pose):
        pelvis["pose"] = list(pose)
        body_pub.Write(String_(data=json.dumps(dict(zip(("x", "y", "yaw", "height", "pitch"), pose)))))

    def walk(goal, seconds):
        start = np.array(pelvis["pose"])
        goal = np.array(goal)
        d = goal - start
        d[2] = (d[2] + np.pi) % (2 * np.pi) - np.pi
        n = max(1, int(seconds * 30))
        for k in range(1, n + 1):
            send_body(start + d * smoothstep(k / n))
            yield

    def left_home():
        pin.framesForwardKinematics(model, data, np.zeros(model.nq))
        return data.oMf[arm_ik.L_hand_id].homogeneous.copy(), data.oMf[arm_ik.R_hand_id].homogeneous.copy()
    left0, right0 = left_home()
    palm_now = {"T": None}

    def solve(palm_world):
        x, y, yaw, h, p = pelvis["pose"]
        T_ik = pelvis_matrix(x, y, h, yaw, p)
        T_ik[:3, 3] += T_ik[:3, :3] @ geometry.ik_origin
        arm_ik.reduced_robot.model.gravity.linear = T_ik[:3, :3].T @ np.array([0.0, 0.0, -9.81])
        q, tau = arm_ik.solve_ik(left0, np.linalg.inv(T_ik) @ palm_world, arm.get_current_dual_arm_q(), arm.get_current_dual_arm_dq())
        arm.ctrl_dual_arm(q, tau)
        palm_now["T"] = palm_world

    def move(goal, seconds):
        start = palm_now["T"].copy()
        n = max(1, int(seconds * 30))
        for k in range(1, n + 1):
            T = goal.copy()
            T[:3, 3] = start[:3, 3] + (goal[:3, 3] - start[:3, 3]) * smoothstep(k / n)
            solve(T)
            time.sleep(1 / 30)

    def hold(seconds):
        for _ in range(int(seconds * 30)):
            solve(palm_now["T"])
            time.sleep(1 / 30)

    def part_pose(name):
        return np.array(sim.get()["objects"][name], dtype=float)

    def attempt(name):
        reset_pub.Write(String_(data="2"))
        pelvis["pose"] = [0.0, 0.0, 0.0, 0.725, 0.0]
        hand.set(HAND_OPEN)
        arm.ctrl_dual_arm(np.zeros(model.nq), np.zeros(model.nq))
        time.sleep(2.5)
        p0 = part_pose(name)
        yaw = -np.pi / 2.0
        forward, left = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        stand = p0[:3] - forward * STAND[1] + left * STAND[0]
        if stand[0] > CLEAR_X:
            stand[0] = CLEAR_X
            to_part = p0[:2] - stand[:2]
            yaw = float(np.arctan2(to_part[1], to_part[0]) + np.arctan2(STAND[0], STAND[1]))
            forward, left = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        for _ in walk([stand[0], stand[1], yaw, HEIGHT, LEAN], 3.0):
            time.sleep(1 / 30)
        time.sleep(0.5)
        R = np.column_stack([np.cos(FINGERS_DOWN) * forward - np.sin(FINGERS_DOWN) * np.array([0.0, 0.0, 1.0]), left,
                             np.sin(FINGERS_DOWN) * forward + np.cos(FINGERS_DOWN) * np.array([0.0, 0.0, 1.0])])
        error = rng.uniform(-a.error, a.error, 3)
        at = lambda point: np.vstack([np.column_stack([R, point - R @ pinch]), [0, 0, 0, 1]])
        low = p0[:3] + error + [0.0, 0.0, 0.02 if name.startswith("peg") else 0.0]
        def pinch_now():
            w = np.array(sim.get()["wrists"]["right"], dtype=float)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = quat_matrix(w[3:]), w[:3]
            return (T @ geometry.palm_in_wrist["right"] @ np.append(pinch, 1.0))[:3]

        def palm_actual():
            w = np.array(sim.get()["wrists"]["right"], dtype=float)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = quat_matrix(w[3:]), w[:3]
            return T @ geometry.palm_in_wrist["right"]

        palm_now["T"] = palm_actual()
        move(at(low + [0.0, 0.0, 0.10]), 2.0)
        hold(0.3)
        for _ in range(4):
            hold(0.35)
            T = palm_now["T"].copy()
            T[:2, 3] += (low - pinch_now())[:2]
            move(T, 0.25)
        P = palm_actual()
        clear = TABLE_Z + 0.006 - float((P[:3, :3] @ open_tips.T)[2].min()) - P[2, 3]
        T = palm_now["T"].copy()
        T[2, 3] += max(clear, low[2] - pinch_now()[2])
        move(T, 1.0)
        hold(0.2)
        off = np.linalg.norm(pinch_now() - low)
        z0 = float(part_pose(name)[2])
        hand.move(HAND_CLOSED, 0.7)
        hold(0.5)
        held = sim.get().get("held", {}).get("right")
        gap = sim.get().get("grip_gap", {}).get("right")
        T = palm_now["T"].copy()
        T[2, 3] += 0.12
        move(T, 1.2)
        hold(0.6)
        lifted = float(part_pose(name)[2]) - z0
        in_hand0 = part_pose(name)[:3] - palm_actual()[:3, 3]
        held = held or sim.get().get("held", {}).get("right")
        x, y, yaw_, h, p = pelvis["pose"]
        start_T = palm_now["T"].copy()
        for k, _ in enumerate(walk([x - 0.40, y, yaw_, h, p], 1.5)):
            T = start_T.copy()
            T[0, 3] += pelvis["pose"][0] - x
            solve(T)
            time.sleep(1 / 30)
        hold(0.8)
        moved = np.linalg.norm(part_pose(name)[:3] - palm_actual()[:3, 3] - in_hand0)
        carried = lifted > 0.05 and moved < 0.03
        hand.move(HAND_OPEN, 0.5)
        hold(0.8)
        released = sim.get().get("held", {}).get("right") is None
        log(f"{name}: pinch {off * 100:.1f} cm from the aim, grip {held}"
            + (f" (thumb {gap[0] * 100:.1f} cm, fingers {gap[1] * 100:.1f} cm from it)" if gap else "") + f", lifted {lifted * 100:.1f} cm, moved {moved * 100:.1f} cm in the hand while the robot walked 0.4 m, "
            f"{'carried' if carried else 'LOST'}, {'let go' if released else 'still held'}")
        return lifted, moved, carried, released

    results = {}
    for name in a.parts:
        for _ in range(a.attempts):
            results[name] = attempt(name)
            if results[name][2]:
                break
    reset_pub.Write(String_(data="2"))
    kept = sum(r[2] for r in results.values())
    log(f"{kept}/{len(results)} parts carried (hand placed within +-{a.error * 1000:.0f} mm)")
    ok = kept >= a.min and all(r[3] for r in results.values())
    print("[PASS] picking parts" if ok else "[FAIL] picking parts", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
