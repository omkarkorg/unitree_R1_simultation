#!/usr/bin/env python3

import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/opt/r1_sim/tests")
sys.path.insert(0, "/opt/xr_teleoperate")
os.chdir("/opt/xr_teleoperate/teleop")

import pinocchio as pin
from fake_headset import HEAD, START_POSE, FakeHeadset
from teleop_hand_test import ARM_SLOTS, ipc, quat_matrix
from teleop_hand_test import UPSTREAM_IK_POINT as IK_POINT
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

MOVES = {"forward": (0, 0, -0.08), "back": (0, 0, 0.08), "up": (0, 0.08, 0), "down": (0, -0.08, 0),
         "outward": (0.08, 0, 0), "inward": (-0.08, 0, 0)}


def main() -> int:
    from teleop.robot_control.robot_arm_ik import R1_A5_ArmIK
    ik = R1_A5_ArmIK()
    model = ik.reduced_robot.model
    data = model.createData()
    lock, last = threading.Lock(), {}

    def on_cmd(msg):
        with lock:
            last["q"] = np.array([msg.motor_cmd[i].q for i in ARM_SLOTS])

    def on_state(msg):
        state = json.loads(msg.data)
        wrist, pelvis = state.get("wrists", {}).get("right"), state.get("robot")
        if wrist:
            with lock:
                last["ee"] = np.asarray(wrist[:3]) + quat_matrix(wrist[3:]) @ IK_POINT - np.asarray(pelvis[:3])

    ChannelFactoryInitialize(1, "lo")
    subs = [ChannelSubscriber("rt/lowcmd", LowCmd_), ChannelSubscriber("rt/sim_state", String_)]
    subs[0].Init(on_cmd, 10)
    subs[1].Init(on_state, 10)

    def settle_and_measure():
        time.sleep(2.0)
        with lock:
            q, ee = last["q"].copy(), last["ee"].copy()
        pin.framesForwardKinematics(model, data, q)
        return data.oMf[ik.R_hand_id].translation.copy(), ee

    def to_robot(v):
        return np.array([-v[2], -v[0], v[1]])

    hs = FakeHeadset().start()
    hs.set_hands()
    time.sleep(1.0)
    if ipc("CMD_START") is None:
        print("no IPC reply - start teleop with --ipc first")
        return 1
    time.sleep(5.0)
    right0 = np.array(START_POSE["right"])
    cmd0, act0 = settle_and_measure()
    head_rel = to_robot(right0 - np.array(HEAD))
    print(f"start pose: absolute-mode IK target {np.round(head_rel + [0.15, 0, 0.45], 3)} m (pelvis frame), "
          f"commanded end effector {np.round(cmd0, 3)}, actual {np.round(act0, 3)}")
    for name, move in MOVES.items():
        hs.set_hands(right=tuple(right0 + move))
        cmd, act = settle_and_measure()
        expect = to_robot(np.array(move))
        g_cmd = float((cmd - cmd0) @ expect / (expect @ expect))
        g_act = float((act - act0) @ expect / (expect @ expect))
        print(f"{name:8s} operator {np.round(expect * 100, 1)} cm | commanded {np.round((cmd - cmd0) * 100, 1)} cm "
              f"(gain {g_cmd:.2f}) | actual {np.round((act - act0) * 100, 1)} cm (gain {g_act:.2f})")
        hs.set_hands()
        cmd0, act0 = settle_and_measure()
    ipc("CMD_STOP")
    time.sleep(1.5)
    hs.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
