#!/usr/bin/env python3

import argparse
import importlib.util
import os
import random
import sys
import threading
import time
import traceback

import numpy as np

sys.path.insert(0, "/opt/xr_teleoperate")
os.chdir("/opt/xr_teleoperate/teleop")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.utils.crc import CRC

BRAINCO_RANGES = np.array([1.5184, 1.0472, 1.4661, 1.4661, 1.4661, 1.4661])

RESULTS = []


def check(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                fn(*a, **k)
                RESULTS.append((name, True, f"{time.time() - t0:.1f}s"))
                print(f"[PASS] {name}", flush=True)
            except Exception as e:
                traceback.print_exc()
                RESULTS.append((name, False, str(e)))
                print(f"[FAIL] {name}: {e}", flush=True)
        return run
    return deco


@check("LowCmd CRC (r1_ros2 pure Python == unitree_sdk2py)")
def test_crc():
    spec = importlib.util.spec_from_file_location("r1_crc", "/opt/teleop/r1_crc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    crc = CRC()
    rng = random.Random(0)
    for _ in range(20):
        msg = unitree_hg_msg_dds__LowCmd_()
        msg.mode_pr = rng.randint(0, 1)
        msg.mode_machine = rng.randint(0, 255)
        for c in msg.motor_cmd:
            c.mode = rng.randint(0, 1)
            c.q, c.dq, c.tau = rng.uniform(-3, 3), rng.uniform(-5, 5), rng.uniform(-10, 10)
            c.kp, c.kd = rng.uniform(0, 200), rng.uniform(0, 5)
        expected = crc.Crc(msg)
        got = mod.hg_lowcmd_crc(msg)
        assert got == expected, f"crc mismatch {got:#x} != {expected:#x}"


@check("camera stream (teleimager ImageClient)")
def test_camera():
    from teleimager.image_client import ImageClient
    client = ImageClient(host="127.0.0.1", request_bgr=True)
    try:
        cfg = client.get_cam_config()["head_camera"]
        assert cfg["enable_zmq"] and not cfg["enable_webrtc"], cfg
        deadline = time.time() + 10
        frame = None
        while time.time() < deadline:
            frame = client.get_head_frame().bgr
            if frame is not None:
                break
            time.sleep(0.05)
        assert frame is not None, "no head camera frame within 10 s"
        assert tuple(frame.shape[:2]) == tuple(cfg["image_shape"]), (frame.shape, cfg["image_shape"])
        print(f"       head camera {frame.shape} binocular={cfg['binocular']}")
    finally:
        client.close()


@check("R1_A5 arms follow xr_teleoperate R1_A5_ArmController (sim mode)")
def test_arm(tolerance):
    import pinocchio as pin
    from teleop.robot_control.robot_arm import R1_A5_ArmController
    from teleop.robot_control.robot_arm_ik import R1_A5_ArmIK
    model = R1_A5_ArmIK().reduced_robot.model
    arm = R1_A5_ArmController(motion_mode=False, simulation_mode=True)
    target = np.array([-0.3, 0.25, 0.0, -0.4, 0.3, -0.3, -0.25, 0.0, -0.4, -0.3])
    tau = pin.rnea(model, model.createData(), target, np.zeros(model.nv), np.zeros(model.nv))
    arm.ctrl_dual_arm(target, tau)
    t0 = time.time()
    err = np.inf
    while time.time() < t0 + 6.0:
        err = np.max(np.abs(arm.get_current_dual_arm_q() - target))
        if err < tolerance:
            break
        time.sleep(0.05)
    print(f"       max arm joint error {err:.3f} rad after {time.time() - t0:.1f} s")
    arm.ctrl_dual_arm_go_home()
    assert err < tolerance, f"arm did not reach target (max error {err:.3f} rad)"


def _brainco_cmd(q_norm):
    msg = MotorCmds_()
    msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
    for i, q in enumerate(q_norm):
        msg.cmds[i].q = float(q)
        msg.cmds[i].dq = 1.0
    return msg


@check("BrainCo hands follow rt/brainco/<side>/cmd")
def test_hands(tolerance):
    states = {}
    lock = threading.Lock()
    subs, pubs = [], {}
    for side in ("left", "right"):
        sub = ChannelSubscriber(f"rt/brainco/{side}/state", MotorStates_)
        sub.Init(lambda m, s=side: _store(states, lock, s, m), 10)
        subs.append(sub)
        pub = ChannelPublisher(f"rt/brainco/{side}/cmd", MotorCmds_)
        pub.Init()
        pubs[side] = pub
    target = np.array([0.3, 0.5, 0.7, 0.7, 0.7, 0.7])
    deadline = time.time() + 6.0
    err = np.inf
    while time.time() < deadline:
        for pub in pubs.values():
            pub.Write(_brainco_cmd(target))
        time.sleep(0.01)
        with lock:
            if len(states) == 2:
                err = max(np.max(np.abs(np.array(states[s]) - target) * BRAINCO_RANGES) for s in states)
        if err < tolerance:
            break
    print(f"       max hand joint error {err:.3f} rad")
    for _ in range(100):
        for pub in pubs.values():
            pub.Write(_brainco_cmd(np.zeros(6)))
        time.sleep(0.01)
    assert err < tolerance, f"hands did not reach target (max error {err:.3f} rad)"


def _store(states, lock, side, msg):
    with lock:
        states[side] = [msg.states[i].q for i in range(6)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=0.15, help="joint tracking tolerance [rad]")
    parser.add_argument("--interface", default=os.environ.get("DDS_INTERFACE", "lo"))
    args = parser.parse_args()
    ChannelFactoryInitialize(1, args.interface)
    test_crc()
    test_camera()
    test_arm(args.tolerance)
    test_hands(args.tolerance)
    print("\n==== protocol test summary ====")
    for name, ok, info in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({info})")
    return 0 if all(ok for _, ok, _ in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
