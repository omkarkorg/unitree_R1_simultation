from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from . import loco_spec as spec

SIDES = ("left", "right")
LEG_CHAIN = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")

V_MAX, WZ_MAX = 1.0, 2.0
H_RATE, PITCH_RATE = 0.6, 1.5
COMMAND_TIMEOUT = 0.5
STEP_TRIGGER = 0.06
STEP_TIME, STEP_TIME_PER_SPEED, MAX_STEP_TIME = 0.35, 0.35, 0.60
STEP_LIFT = 0.05
SPEED_FILTER = 0.2
TOE, HEEL, MAX_ROLL = 0.12, 0.05, 0.6
LEG_STRETCH = 0.985
MAX_REACH = 0.28


def _rot(axis, angle):
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    c, s, C = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _rpy(r, p, y):
    return _rot((0, 0, 1), y) @ _rot((0, 1, 0), p) @ _rot((1, 0, 0), r)


def rot_z(a):
    return _rot((0, 0, 1), a)


def _log(R):
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / max(np.linalg.norm(v), 1e-12) * angle


class Leg:
    def __init__(self, joints):
        self.origins = [(np.asarray(xyz, dtype=float), _rpy(*rpy)) for xyz, rpy, _, _ in joints]
        self.axes = [np.asarray(axis, dtype=float) for _, _, axis, _ in joints]
        self.lower = np.array([lim[0] for _, _, _, lim in joints])
        self.upper = np.array([lim[1] for _, _, _, lim in joints])

    def frames(self, q):
        p, R, out = np.zeros(3), np.eye(3), []
        for (xyz, R0), axis, qi in zip(self.origins, self.axes, q):
            p = p + R @ xyz
            R = R @ R0
            out.append((p.copy(), R @ axis))
            R = R @ _rot(axis, qi)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return out, T

    def fk(self, q):
        return self.frames(q)[1]

    def ik(self, target, q0, iterations=12):
        q = np.clip(np.asarray(q0, dtype=float).copy(), self.lower, self.upper)
        for _ in range(iterations + 1):
            joints, T = self.frames(q)
            e = np.concatenate([target[:3, 3] - T[:3, 3], T[:3, :3] @ _log(T[:3, :3].T @ target[:3, :3])])
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3 or _ == iterations:
                break
            J = np.array([np.concatenate([np.cross(a, T[:3, 3] - p), a]) for p, a in joints]).T
            dq = J.T @ np.linalg.solve(J @ J.T + 2.5e-3 * np.eye(6), e)
            dq *= min(1.0, 0.3 / max(np.abs(dq).max(), 1e-12))
            q = np.clip(q + dq, self.lower, self.upper)
        return q, float(np.linalg.norm(e[:3]))


def load_legs(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    legs = {}
    for side in SIDES:
        chain = []
        for name in LEG_CHAIN:
            j = joints[f"{side}_{name}_joint"]
            o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            chain.append((xyz, rpy, [float(v) for v in a.get("xyz").split()],
                          (float(lim.get("lower")), float(lim.get("upper")))))
        legs[side] = Leg(chain)
    return legs


def pelvis_matrix(x, y, z, yaw, pitch):
    T = np.eye(4)
    T[:3, :3] = rot_z(yaw) @ _rot((0, 1, 0), pitch)
    T[:3, 3] = (x, y, z)
    return T


def quat_xyzw(R):
    t = np.trace(R)
    if t > max(R[0, 0], R[1, 1], R[2, 2]):
        r = math.sqrt(1.0 + t)
        w, x, y, z = r / 2.0, (R[2, 1] - R[1, 2]) / (2 * r), (R[0, 2] - R[2, 0]) / (2 * r), (R[1, 0] - R[0, 1]) / (2 * r)
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        r = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        x, w, y, z = r / 2.0, (R[2, 1] - R[1, 2]) / (2 * r), (R[0, 1] + R[1, 0]) / (2 * r), (R[0, 2] + R[2, 0]) / (2 * r)
    elif R[1, 1] >= R[2, 2]:
        r = math.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2])
        y, w, x, z = r / 2.0, (R[0, 2] - R[2, 0]) / (2 * r), (R[0, 1] + R[1, 0]) / (2 * r), (R[1, 2] + R[2, 1]) / (2 * r)
    else:
        r = math.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2])
        z, w, x, y = r / 2.0, (R[1, 0] - R[0, 1]) / (2 * r), (R[0, 2] + R[2, 0]) / (2 * r), (R[1, 2] + R[2, 1]) / (2 * r)
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class KinematicBody:
    def __init__(self, urdf_path, start=(0.0, 0.0, 0.0)):
        self.legs = load_legs(urdf_path)
        self.q_default = {s: np.array([spec.DEFAULT_POSE[f"{s}_{n}_joint"] for n in LEG_CHAIN]) for s in SIDES}
        stance = {s: self.legs[s].fk(self.q_default[s]) for s in SIDES}
        self.foot_z = spec.H_NOM + float(np.mean([stance[s][2, 3] for s in SIDES]))
        self.home = {s: stance[s][:2, 3].copy() for s in SIDES}
        self.foot_R0 = {s: stance[s][:3, :3].copy() for s in SIDES}
        straight = {s: self.legs[s].frames(np.zeros(len(LEG_CHAIN))) for s in SIDES}
        self.hip = {s: straight[s][0][0][0].copy() for s in SIDES}
        self.leg_reach = LEG_STRETCH * min(np.linalg.norm(straight[s][1][:3, 3] - self.hip[s]) for s in SIDES)
        self.start = tuple(float(v) for v in start)
        self.reset()

    def reset(self, hold=0.0):
        self.hold = float(hold)
        x, y, yaw = self.start
        self.pose = np.array([x, y, yaw, spec.H_NOM, 0.0])
        self.target = self.pose.copy()
        self.velocity, self.velocity_age = None, math.inf
        self.q = {s: self.q_default[s].copy() for s in SIDES}
        self.feet = {s: self._home_foot(s) for s in SIDES}
        self.swing = None
        self.speed = np.zeros(2)
        self.command = [0.0, 0.0, 0.0, spec.H_NOM, 0.0]
        self.targets = {}

    def _home_foot(self, side, lead=(0.0, 0.0)):
        x, y, yaw = self.pose[:3]
        h = rot_z(yaw)[:2, :2] @ self.home[side]
        return np.array([x + h[0] + lead[0], y + h[1] + lead[1], yaw])

    def set_command(self, cmd: dict | None, age: float):
        if cmd is None or self.hold > 0.0:
            return
        h = min(max(float(cmd["height"]), spec.H_MIN), spec.H_NOM)
        p = min(max(float(cmd["pitch"]), spec.PITCH_MIN), spec.PITCH_MAX)
        if "x" in cmd:
            self.velocity = None
            self.target = np.array([float(cmd["x"]), float(cmd["y"]), float(cmd["yaw"]), h, p])
        else:
            v = spec.clamp_command([cmd["vx"], cmd["vy"], cmd["wz"], h, p], envelope=False)
            self.velocity, self.velocity_age = v[:3], age
            self.target[3:] = h, p
        self.command = [float(cmd.get("vx", cmd.get("x", 0.0))), float(cmd.get("vy", cmd.get("y", 0.0))),
                        float(cmd.get("wz", cmd.get("yaw", 0.0))), h, p]

    def step(self, dt, solve_legs=True):
        self.hold = max(0.0, self.hold - dt)
        if self.velocity is not None:
            vx, vy, wz = self.velocity if self.velocity_age < COMMAND_TIMEOUT else (0.0, 0.0, 0.0)
            self.velocity_age += dt
            c, s = math.cos(self.target[2]), math.sin(self.target[2])
            self.target[0] = self.pose[0] + (c * vx - s * vy) * dt
            self.target[1] = self.pose[1] + (s * vx + c * vy) * dt
            self.target[2] = _wrap(self.pose[2] + wz * dt)
        d = self.target - self.pose
        d[2] = _wrap(d[2])
        dist = math.hypot(d[0], d[1])
        if dist > V_MAX * dt:
            d[:2] *= V_MAX * dt / dist
        for i, rate in ((2, WZ_MAX), (3, H_RATE), (4, PITCH_RATE)):
            d[i] = max(-rate * dt, min(rate * dt, d[i]))
        vel = d[:2] / dt if dt > 0 else np.zeros(2)
        self.pose = self.pose + d
        self.pose[2] = _wrap(self.pose[2])
        self._step_feet(dt, vel)
        x, y, yaw, h, p = self.pose
        T_pelvis = pelvis_matrix(x, y, h, yaw, p)
        if not solve_legs and self.targets:
            return T_pelvis, self.targets
        T_inv = np.linalg.inv(T_pelvis)
        targets = {}
        for side in SIDES:
            fx, fy, fyaw = self.feet[side]
            lift = 0.0
            if self.swing is not None and self.swing[0] == side:
                _, a, b, t, duration = self.swing
                u = min(1.0, t / duration)
                s_ = 0.5 - 0.5 * math.cos(math.pi * u)
                fx, fy = a[0] + s_ * (b[0] - a[0]), a[1] + s_ * (b[1] - a[1])
                fyaw = a[2] + s_ * _wrap(b[2] - a[2])
                lift = STEP_LIFT * math.sin(math.pi * u)
            foot = np.eye(4)
            foot[:3, :3] = rot_z(fyaw) @ self.foot_R0[side]
            foot[:3, 3] = (fx, fy, self.foot_z + lift)
            foot = self._roll(foot, fyaw, (T_pelvis @ np.append(self.hip[side], 1.0))[:3])
            q, err = self.legs[side].ik(T_inv @ foot, self.q[side])
            if err > 0.005:
                q2, err2 = self.legs[side].ik(T_inv @ foot, self.q_default[side], iterations=30)
                q = q2 if err2 < err else q
            self.q[side] = q
            targets.update({f"{side}_{n}_joint": float(v) for n, v in zip(LEG_CHAIN, self.q[side])})
        self.targets = targets
        return T_pelvis, targets

    def _roll(self, foot, yaw, hip):
        p = foot[:3, 3]
        if np.linalg.norm(p - hip) <= self.leg_reach:
            return foot
        forward, left = rot_z(yaw)[:3, 0], rot_z(yaw)[:3, 1]
        behind = (p - hip) @ forward < 0.0
        pivot = p + (TOE if behind else -HEEL) * forward - np.array([0.0, 0.0, self.foot_z])
        angle = 0.0
        while angle < MAX_ROLL:
            angle += 0.02
            R = _rot(left, angle if behind else -angle)
            q = pivot + R @ (p - pivot)
            if np.linalg.norm(q - hip) <= self.leg_reach:
                break
        out = foot.copy()
        out[:3, :3], out[:3, 3] = R @ foot[:3, :3], q
        return out

    def _step_feet(self, dt, vel):
        self.speed = self.speed + (vel - self.speed) * dt / (SPEED_FILTER + dt)
        step_time = min(STEP_TIME + STEP_TIME_PER_SPEED * math.hypot(*self.speed), MAX_STEP_TIME)
        lead = tuple(self.speed * step_time / 2.0)
        for side in SIDES:
            if self.swing is None or self.swing[0] != side:
                home = self._home_foot(side)
                d = self.feet[side][:2] - home[:2]
                if np.linalg.norm(d) > MAX_REACH:
                    self.feet[side][:2] = home[:2] + d * (MAX_REACH / np.linalg.norm(d))
        if self.swing is not None:
            side, a, _, t, duration = self.swing
            t += dt
            goal = self._home_foot(side, lead=lead)
            if t >= duration:
                self.feet[side] = goal
                self.swing = None
            else:
                self.swing = (side, a, goal, t, duration)
            return
        err = {}
        for side in SIDES:
            home = self._home_foot(side)
            err[side] = math.hypot(*(self.feet[side][:2] - home[:2])) + 0.1 * abs(_wrap(self.feet[side][2] - home[2]))
        side = max(err, key=err.get)
        if err[side] > STEP_TRIGGER:
            self.swing = (side, self.feet[side].copy(), self._home_foot(side, lead=lead), 0.0, step_time)

    def stance(self):
        return (self.feet["left"][:2] + self.feet["right"][:2]) / 2.0
