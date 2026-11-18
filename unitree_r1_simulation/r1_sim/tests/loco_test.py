#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np

PHASES = [
    ("stand", 3.0, [0.0, 0.0, 0.0, 0.725, 0.0], ("still",)),
    ("turn left 0.6 rad/s", 2.62, [0.0, 0.0, 0.6, 0.725, 0.0], ("wz",)),
    ("stand", 2.0, [0.0, 0.0, 0.0, 0.725, 0.0], ("still",)),
    ("walk forward 0.4 m/s", 4.0, [0.4, 0.0, 0.0, 0.725, 0.0], ("vx",)),
    ("stop", 3.0, [0.0, 0.0, 0.0, 0.725, 0.0], ("still",)),
    ("walk back 0.3 m/s", 4.0, [-0.3, 0.0, 0.0, 0.725, 0.0], ("vx",)),
    ("sideways left 0.2 m/s", 3.0, [0.0, 0.2, 0.0, 0.725, 0.0], ("vy",)),
    ("sideways right 0.2 m/s", 3.0, [0.0, -0.2, 0.0, 0.725, 0.0], ("vy",)),
    ("walk 0.6 m/s", 2.5, [0.6, 0.0, 0.0, 0.725, 0.0], ("vx",)),
    ("stop", 3.0, [0.0, 0.0, 0.0, 0.725, 0.0], ("still",)),
    ("squat to 0.65 m", 4.0, [0.0, 0.0, 0.0, 0.65, 0.0], ("height", "still")),
    ("squat 0.65 m + lean 0.45", 4.0, [0.0, 0.0, 0.0, 0.65, 0.45], ("height", "pitch", "still")),
    ("lean 0.55 rad", 4.0, [0.0, 0.0, 0.0, 0.70, 0.55], ("pitch", "still")),
    ("stand up", 3.0, [0.0, 0.0, 0.0, 0.725, 0.0], ("height", "still")),
    ("walk low 0.3 m/s", 3.0, [0.3, 0.0, 0.0, 0.70, 0.05], ("vx", "height")),
    ("turn right 0.6 rad/s", 2.62, [0.0, 0.0, -0.6, 0.725, 0.0], ("wz",)),
    ("stand", 3.0, [0.0, 0.0, 0.0, 0.725, 0.0], ("still",)),
]
TOL = {"vx": 0.12, "vy": 0.10, "wz": 0.20, "height": 0.035, "pitch": 0.08, "still": 0.05}
SOFT = ("walk back 0.3 m/s", "sideways left 0.2 m/s", "sideways right 0.2 m/s")
SOFT_MIN = 0.25


def quat_xyzw_to_yaw_pitch(q):
    x, y, z, w = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    g = R.T @ np.array([0.0, 0.0, -1.0])
    return yaw, math.atan2(g[0], -g[2])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--quick", action="store_true", help="only stand, walk, turn, squat")
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

    ChannelFactoryInitialize(args.domain, args.interface)
    pub = ChannelPublisher("rt/sim/body_cmd", String_)
    pub.Init()
    samples: list[tuple[float, list, float]] = []
    lock = threading.Lock()

    def on_state(msg):
        d = json.loads(msg.data)
        with lock:
            samples.append((time.monotonic(), d["robot"], d["sim_time"]))

    sub = ChannelSubscriber("rt/sim_state", String_)
    sub.Init(on_state, 10)
    t0 = time.monotonic()
    while not samples:
        if time.monotonic() - t0 > 60:
            print("no rt/sim_state from the simulator")
            return 1
        time.sleep(0.1)

    phases = [p for p in PHASES if not args.quick or p[0] in ("stand", "turn left 0.6 rad/s", "walk forward 0.4 m/s",
                                                              "stop", "squat to 0.65 m", "stand up")]
    ok, fell, warnings = True, False, []
    print(f"{'phase':32s} {'command':>36s}   measured (second half of the phase)")
    for name, secs, cmd, checks in phases:
        start = time.monotonic()
        while time.monotonic() - start < secs:
            pub.Write(String_(data=json.dumps(dict(zip(("vx", "vy", "wz", "height", "pitch"), cmd)))))
            time.sleep(0.02)
        with lock:
            seg = [s for s in samples if s[0] >= start + secs / 2.0]
        if len(seg) < 4:
            print(f"{name:32s} too few samples"); ok = False
            continue
        sim_t = np.array([s[2] for s in seg])
        pos = np.array([s[1][:3] for s in seg])
        yp = np.array([quat_xyzw_to_yaw_pitch(s[1][3:]) for s in seg])
        dt = sim_t[-1] - sim_t[0]
        yaw_mid = math.atan2(np.mean(np.sin(yp[:, 0])), np.mean(np.cos(yp[:, 0])))
        d = pos[-1, :2] - pos[0, :2]
        v_head = np.array([d[0] * math.cos(yaw_mid) + d[1] * math.sin(yaw_mid),
                           -d[0] * math.sin(yaw_mid) + d[1] * math.cos(yaw_mid)]) / dt
        wz = float(np.unwrap(yp[:, 0])[-1] - np.unwrap(yp[:, 0])[0]) / dt
        meas = {"vx": v_head[0], "vy": v_head[1], "wz": wz, "height": float(np.mean(pos[:, 2])),
                "pitch": float(np.mean(yp[:, 1])), "still": float(np.linalg.norm(v_head))}
        want = {"vx": cmd[0], "vy": cmd[1], "wz": cmd[2], "height": cmd[3], "pitch": cmd[4], "still": 0.0}
        if np.min(pos[:, 2]) < 0.35:
            fell = True
        bad = [k for k in checks if abs(meas[k] - want[k]) > TOL[k]]
        slow = []
        if name in SOFT:
            slow = [k for k in bad if k in ("vx", "vy") and meas[k] * want[k] >= SOFT_MIN * want[k] ** 2]
            bad = [k for k in bad if k not in slow]
            warnings += [f"{name}: {meas[k]:+.2f} of {want[k]:+.2f} m/s" for k in slow]
        ok &= not bad and not fell
        print(f"{name:32s} v ({cmd[0]:+.2f},{cmd[1]:+.2f}) wz {cmd[2]:+.2f} h {cmd[3]:.2f} p {cmd[4]:+.2f}   "
              f"v ({meas['vx']:+.2f},{meas['vy']:+.2f}) wz {wz:+.2f} h {meas['height']:.3f} p {meas['pitch']:+.2f}"
              f"{'  FELL' if fell else ''}{'  off: ' + ', '.join(bad) if bad else ''}{'  slow: ' + ', '.join(slow) if slow else ''}")
        if fell:
            break
    for _ in range(25):
        pub.Write(String_(data=json.dumps({"vx": 0.0, "vy": 0.0, "wz": 0.0, "height": 0.725, "pitch": 0.0})))
        time.sleep(0.02)
    reset = ChannelPublisher("rt/reset_pose/cmd", String_)
    reset.Init()
    time.sleep(0.5)
    reset.Write(String_(data="2"))
    time.sleep(1.0)
    for w in warnings:
        print(f"[WARN] slower than commanded: {w}")
    print("LOCO TEST " + ("PASSED" + (f" ({len(warnings)} warning(s))" if warnings else "") if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
