from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--only", default="", help="comma-separated: pegs, gears, rings, nut")
parser.add_argument("--snapshots", default="", help="save pictures of the kit (start, assembled) in this folder")
add_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
if args.snapshots:
    args.enable_cameras = True

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg

TABLE_Z = 0.58
TRACE = bool(os.environ.get("KIT_TRACE"))
SNAP_VIEWS = {"snap_table": ((-0.05, 0.0, 1.05), (0.42, -0.02, 0.60)),
              "snap_side": ((0.10, -0.15, 1.05), (0.10, -0.75, 0.58))}
MAX_FORCE, MAX_PUSH, MAX_TORQUE = 6.0, 2.0, 0.15

sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device,
                                  physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2))
with launch_simulation(sim_cfg, args):
    import numpy as np
    import torch

    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import instantiate

    from r1_sim import assembly as kit
    from r1_sim.layout import floor_box, scene_boxes, scene_fixtures, scene_objects
    from r1_sim.scene import _convex_cfg, _prim_name, _static_box, filter_thread_contacts

    sim = sim_utils.SimulationContext(sim_cfg)
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    for b in scene_boxes("assembly", TABLE_Z) + [floor_box()]:
        setattr(cfg, b["name"], _static_box(f"{{ENV_REGEX_NS}}/{_prim_name(b['name'])}", tuple(b["size"]),
                                            tuple(b["center"]), tuple(b["colour"])))
    for fx in scene_fixtures("assembly", TABLE_Z):
        setattr(cfg, fx.name, AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/{_prim_name(fx.name)}",
                                           spawn=_convex_cfg(fx.pieces, fx.colour, 0.0, fx.thread, fx.piece_names)))
    parts = {o["name"]: o for o in scene_objects("assembly", TABLE_Z)}
    for o in parts.values():
        setattr(cfg, o["name"], RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/{_prim_name(o['name'])}",
                                               spawn=_convex_cfg(o["pieces"], o["colour"], o["mass"], o["thread"],
                                                                 o["piece_names"]),
                                               init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(o["start"][:3]),
                                                                                         rot=tuple(o["start"][3:]))))
    if args.snapshots:
        from isaaclab.sensors import CameraCfg
        cfg.dome = AssetBaseCfg(prim_path="/World/Dome", spawn=sim_utils.DomeLightCfg(intensity=900.0))
        cfg.sun = AssetBaseCfg(prim_path="/World/Sun", spawn=sim_utils.DistantLightCfg(intensity=900.0, angle=1.0),
                               init_state=AssetBaseCfg.InitialStateCfg(rot=(0.0, 0.3007058, 0.0, 0.9537170)))
        def look_at(eye, target):
            z = np.asarray(target, float) - eye
            z /= np.linalg.norm(z)
            x = np.cross(z, [0.0, 0.0, 1.0])
            x /= np.linalg.norm(x)
            R = np.column_stack([x, np.cross(z, x), z])
            from r1_sim.kinematic_body import quat_xyzw
            return tuple(float(v) for v in quat_xyzw(R))

        import numpy as np
        for name, (eye, target) in SNAP_VIEWS.items():
            setattr(cfg, name, CameraCfg(prim_path=f"{{ENV_REGEX_NS}}/{_prim_name(name)}", update_period=0.0, width=1280, height=800,
                                         data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=14.0,
                                         horizontal_aperture=20.955, clipping_range=(0.05, 20.0)),
                                         offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at(np.array(eye), target),
                                                                    convention="ros")))
    scene = instantiate(cfg)
    print(f"nut/bolt: {filter_thread_contacts()} piece pairs filtered (only the thread ridges touch)")
    sim.reset()
    at = kit.targets(TABLE_Z)
    dt = sim_cfg.dt

    def snapshot(tag):
        if not args.snapshots:
            return
        from PIL import Image
        os.makedirs(args.snapshots, exist_ok=True)
        t_end = time.time() + 10.0
        while True:
            scene.write_data_to_sim()
            sim.step(render=False)
            sim.render()
            scene.update(dt)
            frames = [scene[n].data.output["rgb"].torch[0, ..., :3].float() for n in SNAP_VIEWS]
            if all(float(f.std()) > 5.0 for f in frames) or time.time() > t_end:
                break
        for _ in range(10):
            sim.render()
            scene.update(dt)
        for name in SNAP_VIEWS:
            rgb = scene[name].data.output["rgb"].torch[0, ..., :3].cpu().numpy()
            path = os.path.join(args.snapshots, f"kit_{tag}_{name.split('_')[1]}.jpg")
            Image.fromarray(rgb.astype("uint8")).save(path, quality=90)
            print(f"snapshot {path}", flush=True)

    snapshot("start")

    def quat_z(a):
        return [0.0, 0.0, math.sin(a / 2), math.cos(a / 2)]

    def R_of(q):
        x, y, z, w = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def state(name):
        d = scene[name].data
        return (d.root_pos_w.torch[0].cpu().numpy().astype(float), d.root_quat_w.torch[0].cpu().numpy().astype(float),
                d.root_lin_vel_w.torch[0].cpu().numpy().astype(float), d.root_ang_vel_w.torch[0].cpu().numpy().astype(float))

    def place(name, pos, yaw=0.0):
        obj = scene[name]
        pose = torch.tensor([[*pos, *quat_z(yaw)]], dtype=torch.float32)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6))
        obj.reset()

    def park_all():
        for k, o in enumerate(parts):
            place(o, [-1.0 - 0.15 * (k % 4), -1.5 + 0.15 * (k // 4), 0.05])

    class Hand:
        W_POS, W_ROT, W_SPIN = 60.0, 80.0, 60.0

        def __init__(self, name, max_spin=MAX_TORQUE):
            o = parts[name]
            self.name, self.m, self.max_spin = name, o["mass"], max_spin
            pts = np.vstack([np.asarray(p) for p in o["pieces"]])
            self.Iz = 0.8 * self.m * float(np.mean(pts[:, 0] ** 2 + pts[:, 1] ** 2))
            self.Ixy = 0.5 * self.Iz + self.m * float(np.ptp(pts[:, 2])) ** 2 / 12.0
            self.target, self.yaw_rate, self.yaw_hold, self.free_z, self.press = None, None, None, False, 0.0
            self.push, self.spin_i = True, 0.0

        def wrench(self):
            p, q, v, w = state(self.name)
            f = np.zeros(3)
            if self.push:
                kp, kd = self.m * self.W_POS ** 2, 2.0 * self.m * self.W_POS
                f = kp * (self.target - p) - kd * v
                if self.free_z:
                    f[2] = -self.press - kd * v[2]
                n = np.linalg.norm(f[:2])
                if n > MAX_FORCE:
                    f[:2] *= MAX_FORCE / n
                f[2] = float(np.clip(f[2], -MAX_PUSH, MAX_FORCE)) + self.m * 9.81
            z_axis = R_of(q)[:, 2]
            tau = self.Ixy * (self.W_ROT ** 2 * np.cross(z_axis, [0.0, 0.0, 1.0]) - 2.0 * self.W_ROT * np.array([w[0], w[1], 0.0]))
            n = np.linalg.norm(tau)
            tau = tau if n <= MAX_TORQUE else tau * MAX_TORQUE / n
            if self.yaw_rate is not None:
                err = self.yaw_rate - w[2]
                self.spin_i = float(np.clip(self.spin_i + self.max_spin / 1.5 * err * dt, -self.max_spin, self.max_spin))
                spin = self.Iz * self.W_SPIN * err + self.spin_i
            elif self.yaw_hold is not None:
                spin = self.Iz * (self.W_ROT ** 2 * math.remainder(self.yaw_hold - yaw_of(q), 2 * math.pi) - 2.0 * self.W_ROT * w[2])
            else:
                spin = 0.0
            tau[2] = float(np.clip(spin, -self.max_spin, self.max_spin))
            return f, tau

    def run(hands, seconds, record=None):
        log = []
        for _ in range(int(round(seconds / dt))):
            for h in hands:
                f, tau = h.wrench()
                scene[h.name].permanent_wrench_composer.set_forces_and_torques_index(
                    forces=torch.tensor(f, dtype=torch.float32).view(1, 1, 3),
                    torques=torch.tensor(tau, dtype=torch.float32).view(1, 1, 3), body_ids=[0], is_global=True)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            if record:
                log.append([state(n) for n in record])
        return log

    def release(hands):
        for h in hands:
            scene[h.name].permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.zeros(1, 1, 3), torques=torch.zeros(1, 1, 3), body_ids=[0], is_global=True)

    def lower(hand, xy, z_from, z_to, speed=0.03):
        hand.target = np.array([*xy, z_from])
        run([hand], 0.5)
        z = z_from
        while z > z_to:
            z = max(z_to, z - speed * dt * 10)
            hand.target = np.array([*xy, z])
            run([hand], 10 * dt)
        run([hand], 0.6)

    def yaw_of(q):
        R = R_of(q)
        return math.atan2(R[1, 0], R[0, 0])

    results = []

    def check(name, ok, detail):
        results.append((name, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} {detail}", flush=True)

    only = set(filter(None, args.only.split(",")))
    t_start = time.time()
    print(f"assembly kit physics test (CPU PhysX {1 / dt:.0f} Hz); compliant hand: max {MAX_FORCE} N, {MAX_TORQUE} Nm")

    if not only or "pegs" in only:
        print("pegs: into their holes, started 1.4 mm off centre (more than the round hole's 0.93 mm play: the chamfer guides)")
        for kind in ("square", "round", "hex"):
            park_all()
            name = f"peg_{kind}"
            x, y = at[name]
            place(name, [x + 0.001, y - 0.001, at["peg_board_top"] + kit.PEG_LENGTH / 2 + 0.01])
            hand = Hand(name)
            lower(hand, (x + 0.001, y - 0.001), at["peg_board_top"] + kit.PEG_LENGTH / 2 + 0.01, TABLE_Z + kit.PEG_LENGTH / 2 - 0.004)
            release([hand])
            run([], 0.5)
            p, q, _, _ = state(name)
            depth = at["peg_board_top"] - (p[2] - kit.PEG_LENGTH / 2)
            check(f"peg {kind} inserted", depth > kit.PEG_BOARD_THICKNESS - 0.004 and np.hypot(p[0] - x, p[1] - y) < 0.002,
                  f"bottom {depth * 1000:.1f} mm deep (board {kit.PEG_BOARD_THICKNESS * 1000:.0f} mm), "
                  f"off centre {np.hypot(p[0] - x, p[1] - y) * 1000:.1f} mm")

    if not only or "gears" in only:
        print("gears: onto their shafts, then the large one is turned")
        park_all()
        phases = {"gear_large": 0.0, "gear_small": math.pi / kit.GEAR_TEETH[1], "gear_medium": 0.0}
        hands = []
        for name in ("gear_large", "gear_small", "gear_medium"):
            x, y = at[name]
            place(name, [x, y, at["shaft_top"] + 0.02], phases[name])
            hand = Hand(name)
            hand.yaw_hold = phases[name]
            hand.target = np.array([x, y, at["shaft_top"] + 0.02])
            run([hand] + hands, 0.5)
            z = at["shaft_top"] + 0.02
            k = 0
            while z > at["plate_top"] + kit.GEAR_THICKNESS / 2 - 0.002:
                z -= 0.03 * dt * 10
                hand.target = np.array([x, y, z])
                run([hand] + hands, 10 * dt)
                k += 1
                if TRACE and k % 4 == 0:
                    msg = []
                    for n in ("gear_large", "gear_small", "gear_medium")[:len(hands) + 1]:
                        p, q, v, w = state(n)
                        msg.append(f"{n.split('_')[1]} xy({(p[0] - at[n][0]) * 1000:+.1f},{(p[1] - at[n][1]) * 1000:+.1f}) "
                                   f"z {(p[2] - at['plate_top']) * 1000:5.1f} yaw {math.degrees(yaw_of(q)):6.1f} w {w[2]:+.2f}")
                    print(f"      lowering {name.split('_')[1]} target z {(z - at['plate_top']) * 1000:5.1f}: " + " | ".join(msg), flush=True)
            run([hand] + hands, 0.6)
            hand.push = False
            hands.append(hand)
        release(hands)
        run([], 0.3)
        for n in ("gear_large", "gear_small", "gear_medium"):
            p, q, _, _ = state(n)
            print(f"      {n}: ({p[0]:.4f}, {p[1]:.4f}), {(p[2] - at['plate_top']) * 1000:.1f} mm above the plate, "
                  f"yaw {math.degrees(yaw_of(q)):.1f} deg, tilt {math.degrees(math.acos(np.clip(R_of(q)[2, 2], -1, 1))):.1f} deg",
                  flush=True)
        z = [state(n)[0][2] for n in ("gear_large", "gear_small", "gear_medium")]
        on = [abs(zz - (at["plate_top"] + kit.GEAR_THICKNESS / 2)) < 0.003 for zz in z]
        check("gears on their shafts", all(on), "heights above the plate (mm): " +
              ", ".join(f"{(zz - at['plate_top']) * 1000:.1f}" for zz in z))
        drive = Hand("gear_large")
        drive.push, drive.yaw_rate = False, 1.5
        log = run([drive], 3.0, record=("gear_large", "gear_small", "gear_medium"))
        release([drive])
        yaws = np.unwrap(np.array([[yaw_of(s[1]) for s in row] for row in log]), axis=0)
        turned = yaws[-1] - yaws[len(yaws) // 3]
        ratio_s, ratio_m = turned[1] / turned[0], turned[2] / turned[0]
        teeth = kit.GEAR_TEETH
        want_s, want_m = -teeth[0] / teeth[1], teeth[0] / teeth[2]
        check("gear train turns", abs(turned[0]) > 1.0 and abs(ratio_s - want_s) < 0.15 * abs(want_s)
              and abs(ratio_m - want_m) < 0.15 * abs(want_m),
              f"large {math.degrees(turned[0]):.0f} deg: small x{ratio_s:+.2f} (teeth {want_s:+.2f}), "
              f"medium x{ratio_m:+.2f} (teeth {want_m:+.2f})")

    if not only or "rings" in only:
        print("rings: stacked on one post")
        park_all()
        x, y = at["post_0"]
        for k, name in enumerate(("ring_large", "ring_medium", "ring_small")):
            place(name, [x, y, at["post_top"] + 0.02])
            hand = Hand(name)
            lower(hand, (x, y), at["post_top"] + 0.02, at["plate_top"] + kit.RING_THICKNESS / 2 + k * kit.RING_THICKNESS - 0.003)
            release([hand])
            run([], 0.3)
        z = [state(n)[0][2] - at["plate_top"] for n in ("ring_large", "ring_medium", "ring_small")]
        want = [kit.RING_THICKNESS * (k + 0.5) for k in range(3)]
        check("rings stacked", all(abs(a - b) < 0.003 for a, b in zip(z, want)),
              "heights (mm): " + ", ".join(f"{a * 1000:.1f} (want {b * 1000:.1f})" for a, b in zip(z, want)))

    if not only or "nut" in only:
        print("nut: onto the bolt, screwed down, then up again")
        park_all()
        x, y = at["bolt"]
        place("nut", [x, y, at["bolt_top"] + kit.NUT_HEIGHT / 2 + 0.002])
        hand = Hand("nut", max_spin=0.05)
        hand.target = np.array([x, y, at["bolt_top"]])
        hand.free_z, hand.press, hand.yaw_rate = True, 0.5, 0.0
        run([hand], 1.0)
        hand.yaw_rate = -math.pi
        log = run([hand], 5.0, record=("nut",))
        yaw = np.unwrap([yaw_of(r[0][1]) for r in log])
        zz = np.array([r[0][0][2] for r in log])
        tilt = max(math.degrees(math.acos(np.clip(R_of(r[0][1])[2, 2], -1, 1))) for r in log[len(log) // 2:])
        turns = (yaw[-1] - yaw[len(yaw) // 4]) / (2 * math.pi)
        per_turn = (zz[-1] - zz[len(zz) // 4]) / turns if abs(turns) > 0.1 else float("nan")
        check("nut screws down", turns < -1.5 and abs(per_turn - kit.THREAD_PITCH) < 0.0015 and tilt < 6.0,
              f"{-turns:.2f} turns, {per_turn * 1000:.2f} mm per turn (pitch {kit.THREAD_PITCH * 1000:.0f}), "
              f"tilt up to {tilt:.1f} deg, {(at['bolt_top'] - (zz[-1] - kit.NUT_HEIGHT / 2)) * 1000:.0f} mm on the bolt")
        hand.yaw_rate, hand.press, hand.spin_i = math.pi, 0.2, 0.0
        log = run([hand], 3.0, record=("nut",))
        for k in range(0, len(log) if TRACE else 0, 60):
            p, q, v, w = log[k][0]
            print(f"      loosen t={k * dt:.1f}s z {(p[2] - at['bolt_top']) * 1000:6.1f} mm yaw rate {w[2]:+.2f} rad/s "
                  f"tilt {math.degrees(math.acos(np.clip(R_of(q)[2, 2], -1, 1))):.1f} xy off {np.hypot(p[0] - x, p[1] - y) * 1000:.1f} mm "
                  f"wrench {np.round(hand.wrench()[1], 4)}", flush=True)
        yaw = np.unwrap([yaw_of(r[0][1]) for r in log])
        zz = np.array([r[0][0][2] for r in log])
        turns = (yaw[-1] - yaw[len(yaw) // 4]) / (2 * math.pi)
        per_turn = (zz[-1] - zz[len(zz) // 4]) / turns if abs(turns) > 0.1 else float("nan")
        check("nut unscrews", turns > 0.8 and abs(per_turn - kit.THREAD_PITCH) < 0.0015,
              f"{turns:.2f} turns back, {per_turn * 1000:.2f} mm per turn")
        release([hand])

    if args.snapshots:
        for kind in ("square", "round", "hex"):
            x, y = at[f"peg_{kind}"]
            place(f"peg_{kind}", [x, y, TABLE_Z + kit.PEG_LENGTH / 2 + 0.001])
        for name, ph in (("gear_large", 0.0), ("gear_small", math.pi / kit.GEAR_TEETH[1]), ("gear_medium", 0.0)):
            x, y = at[name]
            place(name, [x, y, at["plate_top"] + kit.GEAR_THICKNESS / 2 + 0.001], ph)
        x, y = at["post_0"]
        for k, name in enumerate(("ring_large", "ring_medium", "ring_small")):
            place(name, [x, y, at["plate_top"] + kit.RING_THICKNESS * (k + 0.5) + 0.001])
        if "nut" not in only and only:
            x, y = at["bolt"]
            place("nut", [x, y, at["bolt_top"] - 0.016])
        run([], 0.5)
        snapshot("assembled")
    n_ok = sum(ok for _, ok in results)
    print(f"ASSEMBLY PHYSICS {'PASSED' if n_ok == len(results) else 'FAILED'}: {n_ok}/{len(results)} checks "
          f"({time.time() - t_start:.0f} s)", flush=True)
    os._exit(0 if n_ok == len(results) else 1)
