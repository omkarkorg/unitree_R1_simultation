from __future__ import annotations

import json
import os
from collections import deque

import numpy as np
import torch

from . import loco_spec as spec

COMMAND_TIMEOUT = 0.5
COMMAND_TIMEOUT_STAND = 3.0
STAND = [0.0, 0.0, 0.0, spec.H_NOM, 0.0]


def load_policy(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"leg controller network not found: {path} (train it: ./scripts/train_loco.sh)")
    extra = {"meta.json": ""}
    net = torch.jit.load(path, map_location="cpu", _extra_files=extra)
    meta = json.loads(extra["meta.json"] or "{}")
    if meta.get("spec_version") != spec.SPEC_VERSION:
        raise RuntimeError(f"{path} was trained for {meta.get('spec_version')!r}, the simulator expects "
                           f"{spec.SPEC_VERSION!r} (r1_sim/r1_sim/loco_spec.py)")
    if meta.get("obs_dim") != spec.obs_dim() or meta.get("action_dim") != len(spec.LEG_JOINTS):
        raise RuntimeError(f"{path}: network size {meta.get('obs_dim')} -> {meta.get('action_dim')} does not match "
                           f"loco_spec ({spec.obs_dim()} -> {len(spec.LEG_JOINTS)})")
    net.eval()
    return net, meta


class LegController:
    def __init__(self, policy_path: str, obs_joint_index, leg_joint_index):
        self.net, self.meta = load_policy(policy_path)
        self.obs_idx = np.asarray(obs_joint_index, dtype=np.int64)
        self.leg_idx = np.asarray(leg_joint_index, dtype=np.int64)
        self.default_obs = np.array([spec.DEFAULT_POSE[n] for n in spec.OBS_JOINTS])
        self.default_leg = np.array([spec.DEFAULT_POSE[n] for n in spec.LEG_JOINTS])
        self.kp = np.array([spec.LEG_GAINS[n][0] for n in spec.LEG_JOINTS])
        self.kd = np.array([spec.LEG_GAINS[n][1] for n in spec.LEG_JOINTS])
        self.reset()

    def reset(self) -> None:
        self.history: deque = deque(maxlen=spec.HISTORY)
        self.last_action = np.zeros(len(spec.LEG_JOINTS))
        self.q_des = self.default_leg.copy()
        self.command = list(STAND)
        self.command_age = np.inf
        self.steps = 0

    def set_command(self, cmd, age: float) -> None:
        self.command_age = age
        if cmd is not None and age < COMMAND_TIMEOUT:
            self.command = spec.clamp_command(cmd)
        elif age < COMMAND_TIMEOUT_STAND:
            self.command = [0.0, 0.0, 0.0, *self.command[3:]]
        else:
            self.command = list(STAND)

    def step(self, ang_vel_b, gravity_b, q, dq) -> np.ndarray:
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32)).view(1, -1)
        phase = spec.gait_phase(self.steps / spec.POLICY_HZ)
        frame = spec.observation_frame(t(ang_vel_b), t(gravity_b), t(self.command), t([phase]),
                                       t(np.asarray(q)[self.obs_idx] - self.default_obs), t(np.asarray(dq)[self.obs_idx]),
                                       t(self.last_action))
        if not self.history:
            self.history.extend([frame] * spec.HISTORY)
        else:
            self.history.append(frame)
        obs = torch.cat(list(self.history), dim=-1)
        with torch.inference_mode():
            action = self.net(obs)[0].numpy().astype(np.float64)
        self.last_action = action
        self.q_des = self.default_leg + spec.ACTION_SCALE * action
        self.steps += 1
        return self.q_des

    def observation(self) -> np.ndarray:
        return torch.cat(list(self.history), dim=-1)[0].numpy()
