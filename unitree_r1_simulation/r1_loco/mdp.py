from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from r1_sim import loco_spec as spec

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _progress(env, ramp_steps: int) -> float:
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, float(env.common_step_counter) / ramp_steps)


class BodyCommand(CommandTerm):
    cfg: BodyCommandCfg

    def __init__(self, cfg: BodyCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.cmd = torch.zeros(self.num_envs, 5, device=self.device)
        self.cmd[:, 3] = spec.H_NOM
        self.is_work = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for key in ("error_vel_xy", "error_wz", "error_height", "error_pitch"):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)
        self._err_sum = torch.zeros(self.num_envs, 4, device=self.device)
        self._n = torch.zeros(self.num_envs, device=self.device)
        self.cfg.element_names = list(spec.BODY_COMMAND)

    @property
    def command(self) -> torch.Tensor:
        return self.cmd

    def _update_metrics(self):
        d = self.robot.data
        vel = quat_apply_inverse(yaw_quat(d.root_quat_w.torch), d.root_lin_vel_w.torch)[:, :2]
        pitch, _ = spec.pelvis_pitch_roll(d.projected_gravity_b.torch)
        err = torch.stack([torch.linalg.norm(vel - self.cmd[:, :2], dim=-1),
                           torch.abs(d.root_ang_vel_w.torch[:, 2] - self.cmd[:, 2]),
                           torch.abs(d.root_pos_w.torch[:, 2] - self.cmd[:, 3]),
                           torch.abs(pitch - self.cmd[:, 4])], dim=-1)
        self._err_sum += err
        self._n += 1.0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        ids = slice(None) if env_ids is None else env_ids
        mean = self._err_sum[ids] / self._n[ids].clamp_min(1.0).unsqueeze(-1)
        for i, key in enumerate(("error_vel_xy", "error_wz", "error_height", "error_pitch")):
            self.metrics[key][ids] = mean[:, i]
        extras = super().reset(env_ids)
        self._err_sum[ids] = 0.0
        self._n[ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        n = len(env_ids)
        if n == 0:
            return
        level = _progress(self._env, self.cfg.ramp_steps)
        dev = self.device
        u = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi)
        work = torch.rand(n, device=dev) < self.cfg.work_prob
        vx = u(*spec.VX_RANGE) * (torch.rand(n, device=dev) > 0.15)
        vy = u(*spec.VY_RANGE) * (torch.rand(n, device=dev) > 0.4)
        wz = u(*spec.WZ_RANGE) * (torch.rand(n, device=dev) > 0.3)
        h_walk = spec.H_NOM - level * u(0.0, spec.H_NOM - spec.WALK_HEIGHT_MIN)
        p_walk = level * u(0.0, spec.WALK_PITCH_MAX)
        depth = u(0.0, 1.0) ** 0.7
        h_work = spec.H_NOM - level * depth * (spec.H_NOM - spec.H_MIN)
        p_work = level * u(spec.PITCH_MIN, spec.PITCH_MAX)
        zero = torch.zeros(n, device=dev)
        self.cmd[env_ids, 0] = torch.where(work, zero, vx)
        self.cmd[env_ids, 1] = torch.where(work, zero, vy)
        self.cmd[env_ids, 2] = torch.where(work, zero, wz)
        self.cmd[env_ids, 3] = torch.where(work, h_work, h_walk)
        self.cmd[env_ids, 4] = torch.where(work, p_work, p_walk)
        self.is_work[env_ids] = work

    def _update_command(self):
        pass


@configclass
class BodyCommandCfg(CommandTermCfg):
    class_type: type = BodyCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (3.0, 7.0)
    work_prob: float = 0.4
    ramp_steps: int = 40_000


class UpperBodyMotion(ActionTerm):
    cfg: UpperBodyMotionCfg

    def __init__(self, cfg: UpperBodyMotionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        ids, names = self._asset.find_joints(list(spec.UPPER_JOINTS), preserve_order=True, as_proxy=True)
        if list(names) != list(spec.UPPER_JOINTS):
            raise RuntimeError(f"upper-body joints mismatch: {names}")
        self._ids = ids.torch
        n, j = self.num_envs, len(spec.UPPER_JOINTS)
        self._default = torch.tensor([spec.DEFAULT_POSE[k] for k in spec.UPPER_JOINTS], device=self.device)
        lo = torch.tensor([cfg.ranges[k][0] for k in spec.UPPER_JOINTS], device=self.device)
        hi = torch.tensor([cfg.ranges[k][1] for k in spec.UPPER_JOINTS], device=self.device)
        self._lo, self._hi = lo, hi
        self._from = self._default.repeat(n, 1)
        self._to = self._default.repeat(n, 1)
        self._target = self._default.repeat(n, 1)
        self._t = torch.zeros(n, device=self.device)
        self._move = torch.ones(n, device=self.device)
        self._hold = torch.zeros(n, device=self.device)
        self._empty = torch.zeros(n, 0, device=self.device)
        self._j = j

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._target

    def _new_moves(self, env_ids: torch.Tensor):
        n = len(env_ids)
        if n == 0:
            return
        level = _progress(self._env, self.cfg.ramp_steps)
        r = torch.rand(n, self._j, device=self.device)
        goal = self._default + level * (self._lo + r * (self._hi - self._lo) - self._default)
        still = (torch.rand(n, device=self.device) < self.cfg.still_prob).unsqueeze(-1)
        self._from[env_ids] = self._target[env_ids]
        self._to[env_ids] = torch.where(still, self._target[env_ids], goal)
        self._t[env_ids] = 0.0
        self._move[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.move_time_range)
        self._hold[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.hold_time_range)

    def process_actions(self, actions: torch.Tensor):
        self._t += self._env.step_dt
        done = torch.nonzero(self._t >= self._move + self._hold).flatten()
        self._new_moves(done)
        s = torch.clamp(self._t / self._move, 0.0, 1.0)
        s = (0.5 - 0.5 * torch.cos(math.pi * s)).unsqueeze(-1)
        self._target = self._from + s * (self._to - self._from)

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids if env_ids is not None else slice(None)]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._target[env_ids] = self._default
        self._new_moves(env_ids)


@configclass
class UpperBodyMotionCfg(ActionTermCfg):
    class_type: type = UpperBodyMotion
    asset_name: str = "robot"
    ranges: dict = MISSING
    move_time_range: tuple[float, float] = (0.4, 2.0)
    hold_time_range: tuple[float, float] = (0.0, 1.5)
    still_prob: float = 0.2
    ramp_steps: int = 30_000


def actor_frame(env: ManagerBasedRLEnv, command_name: str, action_name: str, noise: bool,
                asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    d = asset.data
    ang = d.root_ang_vel_b.torch
    grav = d.projected_gravity_b.torch
    q = d.joint_pos.torch[:, asset_cfg.joint_ids] - d.default_joint_pos.torch[:, asset_cfg.joint_ids]
    dq = d.joint_vel.torch[:, asset_cfg.joint_ids]
    if noise:
        ang = ang + torch.empty_like(ang).uniform_(-0.2, 0.2)
        grav = grav + torch.empty_like(grav).uniform_(-0.05, 0.05)
        q = q + torch.empty_like(q).uniform_(-0.01, 0.01)
        dq = dq + torch.empty_like(dq).uniform_(-1.5, 1.5)
    cmd = env.command_manager.get_command(command_name)
    last = env.action_manager.get_term(action_name).raw_actions
    return spec.observation_frame(ang, grav, cmd, episode_phase(env).unsqueeze(-1), q, dq, last)


def episode_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    return spec.gait_phase(env.episode_length_buf.float() * env.step_dt)


def privileged(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
               ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = (sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0).float()
    return torch.cat([asset.data.root_lin_vel_b.torch, asset.data.root_pos_w.torch[:, 2:3] - spec.H_NOM, contact],
                     dim=-1)


def track_height_exp(env: ManagerBasedRLEnv, command_name: str, std: float,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.root_pos_w.torch[:, 2] - env.command_manager.get_command(command_name)[:, 3]
    return torch.exp(-torch.square(err / std))


def track_pitch_exp(env: ManagerBasedRLEnv, command_name: str, std: float,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pitch, _ = spec.pelvis_pitch_roll(asset.data.projected_gravity_b.torch)
    err = pitch - env.command_manager.get_command(command_name)[:, 4]
    return torch.exp(-torch.square(err / std))


def roll_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    _, roll = spec.pelvis_pitch_roll(asset.data.projected_gravity_b.torch)
    return torch.square(roll)


def _standing(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(cmd[:, :2], dim=-1) < 0.1) & (torch.abs(cmd[:, 2]) < 0.1)


def stand_still_feet_down(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    down = sensor.data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1)[0] > 1.0
    return torch.sum(~down, dim=1).float() * _standing(env, command_name)


def stand_still_feet_motion(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    speed = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids].norm(dim=-1).sum(dim=1)
    return speed * _standing(env, command_name)


def feet_too_close(env: ManagerBasedRLEnv, min_distance: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet = asset.data.body_link_pose_w.torch[:, asset_cfg.body_ids, :3]
    delta = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), feet[:, 0] - feet[:, 1])
    return torch.clamp(min_distance - delta[:, 1], min=0.0)


def feet_clearance(env: ManagerBasedRLEnv, command_name: str, target_height: float, asset_cfg: SceneEntityCfg
                   ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.body_link_pose_w.torch[:, asset_cfg.body_ids, 2]
    speed = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    err = torch.clamp(target_height - z, min=0.0)
    return torch.sum(torch.square(err) * speed, dim=1) * ~_standing(env, command_name)


def feet_gait(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    down = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0
    phase = episode_phase(env)
    stance = torch.stack([phase < spec.STANCE_SHARE, (phase + 0.5) % 1.0 < spec.STANCE_SHARE], dim=-1)
    match = (down == stance).float().mean(dim=-1)
    return match * ~_standing(env, command_name)
