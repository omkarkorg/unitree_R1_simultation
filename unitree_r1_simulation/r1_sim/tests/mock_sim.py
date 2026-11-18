#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time

import math

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r1_sim import loco_spec
from r1_sim.dds_bridge import HAND_DRIVE_KD, HAND_DRIVE_KP, HAND_NUM_JOINTS, JointServo, R1DdsBridge
from r1_sim.image_server import TeleimagerServer
from r1_sim.lab_stream import LabStatePublisher
from r1_sim.layout import LAB_STATE_HZ, SCENES, TABLE_TOP_Z, lab_description, scene_objects
from r1_sim.joint_map import HAND_SIDES, body_joint_names

COMMAND_TIMEOUT, COMMAND_TIMEOUT_STAND = 0.5, 3.0
V_MAX, WZ_MAX = 1.0, 2.0


def quat_xyzw(yaw: float, pitch: float) -> list[float]:
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return [-sy * sp, cy * sp, sy * cp, cy * cp]


def main() -> int:
    parser = argparse.ArgumentParser(description='GPU-free stand-in for run_sim.py (same DDS topics and camera stream, no physics).')
    parser.add_argument("--dds-domain", type=int, default=int(os.environ.get("DDS_DOMAIN_ID", "1")))
    parser.add_argument("--dds-interface", default=os.environ.get("DDS_INTERFACE", "lo"))
    parser.add_argument("--rate", type=float, default=200.0)
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to run (0 = forever)")
    parser.add_argument("--stereo", type=int, default=int(os.environ.get("SIM_STEREO", "0")),
                        help="1 = side-by-side stereo frames, like run_sim.py --stereo")
    parser.add_argument("--scene", default=os.environ.get("SIM_SCENE", "assembly"), choices=SCENES)
    parser.add_argument("--locomotion", type=int, default=int(os.environ.get("SIM_LOCOMOTION", "1")), choices=(0, 1))
    parser.add_argument("--body", default=os.environ.get("SIM_BODY", "exact"), choices=("exact", "balance"),
                        help="what run_sim.py's --body would be (exact also takes pose commands)")
    args = parser.parse_args()
    floating = bool(args.locomotion)

    names = body_joint_names()
    bridge = R1DdsBridge(names, domain_id=args.dds_domain, interface=args.dds_interface, enable_hands=True)
    width = 640 * (2 if args.stereo else 1)
    geometry = {"vfov_deg": 70.0, "pitch_deg": 50.0, "parent": "head_yaw_link", "mount_xyz": [0.095, 0.0, 0.0],
                "baseline_m": 0.064 if args.stereo else 0.0}
    pelvis_height = loco_spec.H_NOM if floating else 0.75
    server = TeleimagerServer((480, width), binocular=bool(args.stereo), fps=30, geometry=geometry,
                              extra={"lab": lab_description(TABLE_TOP_Z, pelvis_height, True, args.scene, floating,
                                                            body=args.body if floating else "fixed")})
    lab_pub = LabStatePublisher()
    objects = {o["name"]: list(o["start"]) for o in scene_objects(args.scene, TABLE_TOP_Z)}
    head = [names.index("head_pitch_joint"), names.index("head_yaw_joint")]
    legs = [names.index(n) for n in loco_spec.LEG_JOINTS]
    base = {"x": 0.0, "y": 0.0, "yaw": 0.0, "h": pelvis_height, "pitch": 0.0, "cmd": [0.0, 0.0, 0.0, pelvis_height, 0.0],
            "hold_until": 0.0}

    nb = len(names)
    q_hold = np.array([loco_spec.DEFAULT_POSE[n] for n in names]) if floating else np.zeros(nb)
    body = JointServo(q_hold, np.full(nb, 60.0), np.full(nb, 2.0), np.full(nb, 60.0))
    nh = HAND_NUM_JOINTS
    hands = {s: JointServo(np.zeros(nh), np.full(nh, HAND_DRIVE_KP), np.full(nh, HAND_DRIVE_KD), np.full(nh, 2.0))
             for s in HAND_SIDES}
    q = {"body": q_hold.copy(), **{s: np.zeros(nh) for s in HAND_SIDES}}
    servo = {"body": body, **hands}

    dt = 1.0 / args.rate
    alpha = 1.0 - np.exp(-dt / 0.05)
    t0 = time.perf_counter()
    next_t = t0
    step = 0
    print(f"[mock_sim] DDS domain {args.dds_domain} iface {args.dds_interface}; camera 480x{width}; scene "
          f"{args.scene}; {'free-standing (rt/sim/body_cmd, rt/arm_sdk)' if floating else 'pelvis fixed'}", flush=True)
    try:
        while args.duration <= 0 or time.perf_counter() - t0 < args.duration:
            low, arm = bridge.get_body_command(), bridge.get_arm_sdk_command() if floating else None
            body.apply(arm if arm is not None and (low is None or arm.stamp >= low.stamp) else low)
            if floating:
                got = bridge.get_body_target()
                if time.monotonic() < base["hold_until"]:
                    got = None
                age = time.monotonic() - got[1] if got else np.inf
                if got and "x" in got[0] and args.body == "exact":
                    t = got[0]
                    dx, dy = t["x"] - base["x"], t["y"] - base["y"]
                    dist, dyaw = math.hypot(dx, dy), (t["yaw"] - base["yaw"] + math.pi) % (2 * math.pi) - math.pi
                    k = min(1.0, V_MAX * dt / dist) if dist > 1e-9 else 0.0
                    base["x"] += k * dx
                    base["y"] += k * dy
                    base["yaw"] += max(-WZ_MAX * dt, min(WZ_MAX * dt, dyaw))
                    base["cmd"] = [0.0, 0.0, 0.0, t["height"], t["pitch"]]
                    vx = vy = wz = 0.0
                else:
                    vel = got and "vx" in got[0]
                    if vel and age < COMMAND_TIMEOUT:
                        base["cmd"] = loco_spec.clamp_command([got[0][k] for k in ("vx", "vy", "wz", "height", "pitch")],
                                                              envelope=args.body == "balance")
                    elif age < COMMAND_TIMEOUT_STAND:
                        base["cmd"] = [0.0, 0.0, 0.0, *base["cmd"][3:]]
                    else:
                        base["cmd"] = [0.0, 0.0, 0.0, loco_spec.H_NOM, 0.0]
                    vx, vy, wz = base["cmd"][:3]
                _, _, _, h_cmd, p_cmd = base["cmd"]
                c, s_ = math.cos(base["yaw"]), math.sin(base["yaw"])
                base["x"] += (c * vx - s_ * vy) * dt
                base["y"] += (s_ * vx + c * vy) * dt
                base["yaw"] = (base["yaw"] + wz * dt + math.pi) % (2 * math.pi) - math.pi
                a = 1.0 - math.exp(-dt / 0.2)
                base["h"] += a * (h_cmd - base["h"])
                base["pitch"] += a * (p_cmd - base["pitch"])
                body.q_des[legs] = q_hold[legs]
                body.kp[legs] = 60.0
            for s in HAND_SIDES:
                hands[s].apply(bridge.get_hand_command(s))
            for key, sv in servo.items():
                active = sv.kp > 0.0
                q[key] = np.where(active, q[key] + alpha * (sv.q_des - q[key]), q[key])
            for request in bridge.pop_reset_requests():
                print(f"[mock_sim] reset request {request!r}", flush=True)
                if request == "2":
                    base.update(x=0.0, y=0.0, yaw=0.0, h=pelvis_height, pitch=0.0, cmd=[0.0, 0.0, 0.0, pelvis_height, 0.0],
                                hold_until=time.monotonic() + 0.5)
                    q["body"][:] = q_hold
                    for s in HAND_SIDES:
                        q[s][:] = 0.0
            sim_time = step * dt
            zero3 = np.zeros(3)
            qx, qy, qz, qw = quat_xyzw(base["yaw"], base["pitch"])
            root = [base["x"], base["y"], base["h"], qx, qy, qz, qw]
            bridge.publish_body_state(q["body"], np.zeros(nb), body.torque(q["body"], np.zeros(nb)),
                                      np.array([qw, qx, qy, qz]), zero3, np.array([0.0, 0.0, 9.81]),
                                      np.array([0.0, base["pitch"], base["yaw"]]), sim_time)
            for s in HAND_SIDES:
                bridge.publish_hand_state(s, q[s], np.zeros(nh), hands[s].torque(q[s], np.zeros(nh)))
            if step % max(1, int(round(args.rate / LAB_STATE_HZ))) == 0:
                lab_pub.publish(sim_time, np.concatenate([q["body"]] + [q[s] for s in HAND_SIDES]), objects, root=root,
                                stance=root[:2] if floating else None)
            if step % max(1, int(round(args.rate / 10))) == 0:
                bridge.publish_sim_state({"sim_time": round(sim_time, 3), "pose_format": "x y z qx qy qz qw (world frame, m)",
                                          "robot": [round(float(v), 4) for v in root], "objects": objects})
            if step % max(1, int(args.rate / 30)) == 0:
                img = Image.new("RGB", (width, 480), (40, 44, 52))
                draw = ImageDraw.Draw(img)
                draw.text((20, 20), f"R1 mock sim  t={sim_time:7.2f}s", fill=(255, 255, 255))
                draw.text((20, 40), f"lowcmd msgs={bridge.stats['lowcmd']} crc_err={bridge.stats['lowcmd_crc_errors']}",
                          fill=(200, 200, 200))
                draw.text((20, 60), f"L shoulder pitch q={q['body'][names.index('left_shoulder_pitch_joint')]:+.3f}",
                          fill=(200, 200, 200))
                server.submit_frame(np.asarray(img), {"head_q": [round(float(q["body"][i]), 5) for i in head]})
            if step % int(args.rate * 2) == 0:
                print(f"[mock_sim] t={sim_time:6.1f}s stats={bridge.stats} jpeg={server.frames_sent} "
                      f"config_requests={server.config_requests}", flush=True)
            step += 1
            next_t += dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[mock_sim] final stats={bridge.stats} jpeg={server.frames_sent}", flush=True)
        server.close()
        lab_pub.close()
        bridge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
