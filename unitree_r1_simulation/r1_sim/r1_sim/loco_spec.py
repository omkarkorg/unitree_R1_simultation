from __future__ import annotations

from .joint_map import body_joint_names

SPEC_VERSION = "r1-wholebody-2"

PHYSICS_HZ = 120.0
DECIMATION = 3
POLICY_HZ = PHYSICS_HZ / DECIMATION

OBS_JOINTS: tuple[str, ...] = tuple(body_joint_names())
LEG_JOINTS: tuple[str, ...] = tuple(n for n in OBS_JOINTS if any(k in n for k in ("hip", "knee", "ankle")))
UPPER_JOINTS: tuple[str, ...] = tuple(n for n in OBS_JOINTS if n not in LEG_JOINTS)

UPPER_RANGES: dict[str, tuple[float, float]] = {
    "waist_roll_joint": (-0.1, 0.1),
    "waist_yaw_joint": (-0.3, 0.3),
    "head_pitch_joint": (-0.6, 0.6),
    "head_yaw_joint": (-1.2, 1.2),
}
for _s, _sign in (("left", 1.0), ("right", -1.0)):
    UPPER_RANGES[f"{_s}_shoulder_pitch_joint"] = (-2.0, 0.6)
    UPPER_RANGES[f"{_s}_shoulder_roll_joint"] = tuple(sorted((-0.2 * _sign, 1.5 * _sign)))
    UPPER_RANGES[f"{_s}_shoulder_yaw_joint"] = (-1.2, 1.2)
    UPPER_RANGES[f"{_s}_elbow_joint"] = (-0.9, 2.0)
    UPPER_RANGES[f"{_s}_wrist_roll_joint"] = (-1.5, 1.5)

DEFAULT_POSE: dict[str, float] = {n: 0.0 for n in OBS_JOINTS}
for _side in ("left", "right"):
    DEFAULT_POSE[f"{_side}_hip_pitch_joint"] = -0.25
    DEFAULT_POSE[f"{_side}_knee_joint"] = 0.50
    DEFAULT_POSE[f"{_side}_ankle_pitch_joint"] = -0.25

LEG_GAINS: dict[str, tuple[float, float]] = {}
for _n in LEG_JOINTS:
    if "knee" in _n:
        LEG_GAINS[_n] = (200.0, 5.0)
    elif "ankle" in _n:
        LEG_GAINS[_n] = (60.0, 3.0)
    else:
        LEG_GAINS[_n] = (150.0, 4.0)
ARMATURE = 0.01

UPPER_HOLD_GAINS: dict[str, tuple[float, float]] = {}
for _n in UPPER_JOINTS:
    UPPER_HOLD_GAINS[_n] = (200.0, 5.0) if "waist" in _n else (15.0, 1.0) if "head" in _n else (60.0, 2.0)

ACTION_SCALE = 0.25
GAIT_PERIOD = 0.8
STANCE_SHARE = 0.55
ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.05
HISTORY = 5

BODY_COMMAND = ("vx", "vy", "wz", "height", "pitch")
H_NOM = 0.725
H_MIN = 0.50
PITCH_MAX = 0.55
PITCH_MIN = -0.05
VX_RANGE = (-0.5, 0.8)
VY_RANGE = (-0.4, 0.4)
WZ_RANGE = (-1.0, 1.0)
WALK_HEIGHT_MIN = 0.65
WALK_PITCH_MAX = 0.15

USE_H_MIN = 0.65
USE_WALK_H_MIN = 0.70
USE_WALK_PITCH_MAX = 0.05


def clamp_command(cmd: list[float], envelope: bool = True) -> list[float]:
    vx = min(max(float(cmd[0]), VX_RANGE[0]), VX_RANGE[1])
    vy = min(max(float(cmd[1]), VY_RANGE[0]), VY_RANGE[1])
    wz = min(max(float(cmd[2]), WZ_RANGE[0]), WZ_RANGE[1])
    walking = envelope and (abs(vx) > 1e-3 or abs(vy) > 1e-3 or abs(wz) > 1e-3)
    h_min = (USE_WALK_H_MIN if walking else USE_H_MIN) if envelope else H_MIN
    height = min(max(float(cmd[3]), h_min), H_NOM)
    pitch = min(max(float(cmd[4]), PITCH_MIN), USE_WALK_PITCH_MAX if walking else PITCH_MAX)
    return [vx, vy, wz, height, pitch]


def frame_dim() -> int:
    return 3 + 3 + len(BODY_COMMAND) + 2 + 2 * len(OBS_JOINTS) + len(LEG_JOINTS)


def obs_dim() -> int:
    return HISTORY * frame_dim()


def gait_phase(t):
    return (t % GAIT_PERIOD) / GAIT_PERIOD


def observation_frame(ang_vel_b, gravity_b, command, phase, joint_pos_rel, joint_vel, last_action):
    import math

    import torch

    cmd = torch.cat([command[:, :3], command[:, 3:4] - H_NOM, command[:, 4:5]], dim=-1)
    clock = torch.cat([torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)], dim=-1)
    return torch.cat([ang_vel_b * ANG_VEL_SCALE, gravity_b, cmd, clock, joint_pos_rel, joint_vel * JOINT_VEL_SCALE,
                      last_action], dim=-1)


def pelvis_pitch_roll(gravity_b):
    import torch

    pitch = torch.atan2(gravity_b[:, 0], -gravity_b[:, 2])
    roll = torch.atan2(-gravity_b[:, 1], -gravity_b[:, 2])
    return pitch, roll
