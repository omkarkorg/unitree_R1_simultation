from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("R1_SIM_DIR", "/workspace/r1_sim"))

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--no-arms", action="store_true", help="keep the upper body still")
parser.add_argument("--envelope", action="store_true",
                    help="instead of loco_test's phases: a grid of squat depth x lean (standing, from and back to a "
                         "stand each time) and of walking 0.3 m/s at several heights, to map where it is reliable")
add_launcher_args(parser)
args = parser.parse_args()

import r1_loco
from r1_loco.env_cfg import FEET, R1WholeBodyEnvCfg_PLAY

env_cfg = R1WholeBodyEnvCfg_PLAY()
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device
env_cfg.commands.body.resampling_time_range = (1e9, 1e9)
env_cfg.events.reset_base.params["pose_range"] = {"yaw": (0.0, 0.0)}
env_cfg.events.reset_base.params["velocity_range"] = {}
if args.no_arms:
    env_cfg.actions.upper_body.still_prob = 1.0

with launch_simulation(env_cfg, args):
    import math

    import numpy as np
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    from r1_sim import loco_spec as spec
    from r1_sim.locomotion import load_policy

    sys.path.insert(0, os.path.join(os.environ.get("R1_SIM_DIR", "/workspace/r1_sim"), "tests"))
    from loco_test import PHASES, TOL

    if args.envelope:
        stand = ("stand", 2.0, [0.0, 0.0, 0.0, spec.H_NOM, 0.0], ())
        PHASES = []
        for h in (0.50, 0.55, 0.60, 0.65, 0.70):
            for p in (0.0, 0.15, 0.3, 0.45, 0.55):
                PHASES += [stand, (f"squat {h:.2f} m, lean {p:.2f}", 3.0, [0.0, 0.0, 0.0, h, p], ("height", "pitch", "still"))]
        for h, p in ((0.725, 0.0), (0.70, 0.0), (0.70, 0.05), (0.68, 0.05), (0.65, 0.10)):
            for vx in (0.3, -0.3):
                PHASES += [stand, (f"walk {vx:+.1f} m/s at {h:.3f} m, lean {p:.2f}", 3.0, [vx, 0.0, 0.0, h, p], ("vx", "height"))]

    net, meta = load_policy(args.policy)
    net = net.to(env_cfg.sim.device)
    env = ManagerBasedRLEnv(env_cfg)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    feet = [sensor.body_names.index(n) for n in FEET]
    feet_bodies = [robot.body_names.index(n) for n in FEET]
    term = env.command_manager.get_term("body")
    dt = env.step_dt
    print(f"policy {args.policy} (iteration {meta.get('iteration')}), {args.num_envs} robots, upper body "
          f"{'still' if args.no_arms else 'moving at random'}")
    print(f"{'phase':32s}   measured: v (x,y)  wz   height pitch | fell | 1 foot down | sliding")
    ok = True
    for name, secs, cmd, checks in PHASES:
        quiet = args.envelope and name == "stand"
        term.cmd[:] = torch.tensor(cmd, device=env.device)
        n = int(round(secs / dt))
        rec = {"v": [], "wz": [], "h": [], "p": [], "single": [], "slide": []}
        fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for i in range(n):
            with torch.inference_mode():
                act = net(obs["policy"])
            obs, _, term_, trunc, _ = env.step(act)
            fell |= term_ & ~trunc
            term.cmd[:] = torch.tensor(cmd, device=env.device)
            if i >= n // 2:
                d = robot.data
                v = quat_apply_inverse(yaw_quat(d.root_quat_w.torch), d.root_lin_vel_w.torch)[:, :2]
                pitch, _ = spec.pelvis_pitch_roll(d.projected_gravity_b.torch)
                down = sensor.data.net_forces_w.torch[:, feet].norm(dim=-1) > 1.0
                fv = d.body_lin_vel_w.torch[:, feet_bodies, :2].norm(dim=-1)
                rec["v"].append(v.mean(0))
                rec["wz"].append(d.root_ang_vel_w.torch[:, 2].mean())
                rec["h"].append(d.root_pos_w.torch[:, 2].mean())
                rec["p"].append(pitch.mean())
                rec["single"].append((down.sum(-1) == 1).float().mean())
                rec["slide"].append((fv * down).sum() / down.sum().clamp_min(1))
        v = torch.stack(rec["v"]).mean(0).tolist()
        m = {"vx": v[0], "vy": v[1], "wz": float(torch.stack(rec["wz"]).mean()), "height": float(torch.stack(rec["h"]).mean()),
             "pitch": float(torch.stack(rec["p"]).mean()), "still": math.hypot(*v)}
        want = {"vx": cmd[0], "vy": cmd[1], "wz": cmd[2], "height": cmd[3], "pitch": cmd[4], "still": 0.0}
        bad = [k for k in checks if abs(m[k] - want[k]) > TOL[k]]
        n_fell = int(fell.sum())
        ok &= not bad and n_fell == 0
        if quiet and n_fell == 0:
            continue
        print(f"{name:32s}   ({m['vx']:+.2f},{m['vy']:+.2f}) {m['wz']:+.2f}  {m['height']:.3f}  {m['pitch']:+.2f} | "
              f"{n_fell:4d} | {float(torch.stack(rec['single']).mean()):5.0%}      | {float(torch.stack(rec['slide']).mean()):.3f} m/s"
              f"{'  off: ' + ', '.join(bad) if bad else ''}")
    print("EVALUATION " + ("PASSED" if ok else "FAILED"))
    env.close()
