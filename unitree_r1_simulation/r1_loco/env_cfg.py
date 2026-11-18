from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.core.velocity import mdp as vel

from r1_sim import loco_spec as spec

from . import mdp
from .robot import r1_cfg

FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
FOOT_Z = 0.053
FALL_BODIES = ["pelvis_link", "waist_roll_link", "waist_yaw_link", "head_.*", ".*_shoulder_.*", ".*_elbow_link",
               ".*_wrist_roll_link", ".*_hip_.*"]

UPPER_RANGES = spec.UPPER_RANGES
assert set(UPPER_RANGES) == set(spec.UPPER_JOINTS)


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(friction_combine_mode="multiply",
                                                        restitution_combine_mode="multiply",
                                                        static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False,
    )
    robot = r1_cfg("train")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class CommandsCfg:
    body = mdp.BodyCommandCfg()


@configclass
class ActionsCfg:
    legs = base.JointPositionActionCfg(asset_name="robot", joint_names=list(spec.LEG_JOINTS), preserve_order=True,
                                       scale=spec.ACTION_SCALE, use_default_offset=True)
    upper_body = mdp.UpperBodyMotionCfg(ranges=UPPER_RANGES, move_time_range=(0.8, 3.0), ramp_steps=40_000)


def _obs_joints():
    return SceneEntityCfg("robot", joint_names=list(spec.OBS_JOINTS), preserve_order=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        frame = ObsTerm(func=mdp.actor_frame, history_length=spec.HISTORY, flatten_history_dim=True,
                        params={"command_name": "body", "action_name": "legs", "noise": True,
                                "asset_cfg": _obs_joints()})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        frame = ObsTerm(func=mdp.actor_frame, history_length=spec.HISTORY, flatten_history_dim=True,
                        params={"command_name": "body", "action_name": "legs", "noise": False,
                                "asset_cfg": _obs_joints()})
        privileged = ObsTerm(func=mdp.privileged,
                             params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET,
                                                                  preserve_order=True)})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    physics_material = EventTerm(func=base.randomize_rigid_body_material, mode="startup",
                                 params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                                         "static_friction_range": (0.4, 1.2), "dynamic_friction_range": (0.4, 1.0),
                                         "restitution_range": (0.0, 0.0), "num_buckets": 64,
                                         "make_consistent": True})
    torso_mass = EventTerm(func=base.randomize_rigid_body_mass, mode="startup",
                           params={"asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                                   "mass_distribution_params": (-1.0, 2.0), "operation": "add"})
    hand_payload = EventTerm(func=base.randomize_rigid_body_mass, mode="startup",
                             params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_wrist_roll_link"),
                                     "mass_distribution_params": (0.0, 0.6), "operation": "add"})
    torso_com = EventTerm(func=base.randomize_rigid_body_com, mode="startup",
                          params={"asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                                  "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}})
    leg_gains = EventTerm(func=base.randomize_actuator_gains, mode="startup",
                          params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(spec.LEG_JOINTS)),
                                  "stiffness_distribution_params": (0.85, 1.15),
                                  "damping_distribution_params": (0.85, 1.15), "operation": "scale"})
    waist_gains = EventTerm(func=base.randomize_actuator_gains, mode="startup",
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*"),
                                    "stiffness_distribution_params": (0.25, 1.25),
                                    "damping_distribution_params": (0.5, 1.25), "operation": "scale"})
    arm_gains = EventTerm(func=base.randomize_actuator_gains, mode="startup",
                          params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_joint",
                                                                                    ".*_wrist_roll_joint"]),
                                  "stiffness_distribution_params": (0.5, 1.5),
                                  "damping_distribution_params": (0.7, 1.5), "operation": "scale"})
    reset_base = EventTerm(func=base.reset_root_state_uniform, mode="reset",
                           params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
                                   "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.2, 0.2),
                                                      "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3),
                                                      "yaw": (-0.3, 0.3)}})
    reset_joints = EventTerm(func=base.reset_joints_by_offset, mode="reset",
                             params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)})
    push_robot = EventTerm(func=base.push_by_setting_velocity, mode="interval", interval_range_s=(6.0, 12.0),
                           params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}})
    hand_forces = EventTerm(func=base.apply_external_force_torque, mode="interval", interval_range_s=(2.0, 5.0),
                            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_wrist_roll_link"),
                                    "force_range": (-5.0, 5.0), "torque_range": (-0.5, 0.5)})


@configclass
class RewardsCfg:
    track_lin_vel_xy = RewTerm(func=vel.track_lin_vel_xy_yaw_frame_exp, weight=2.0,
                               params={"command_name": "body", "std": 0.5})
    track_ang_vel_z = RewTerm(func=vel.track_ang_vel_z_world_exp, weight=1.5, params={"command_name": "body", "std": 0.5})
    track_height = RewTerm(func=mdp.track_height_exp, weight=1.0, params={"command_name": "body", "std": 0.05})
    track_pitch = RewTerm(func=mdp.track_pitch_exp, weight=1.0, params={"command_name": "body", "std": 0.1})
    termination = RewTerm(func=base.is_terminated, weight=-200.0)
    roll = RewTerm(func=mdp.roll_l2, weight=-1.0)
    stand_still_feet_down = RewTerm(func=mdp.stand_still_feet_down, weight=-1.0,
                                    params={"command_name": "body",
                                            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})
    stand_still_feet_motion = RewTerm(func=mdp.stand_still_feet_motion, weight=-0.5,
                                      params={"command_name": "body",
                                              "asset_cfg": SceneEntityCfg("robot", body_names=FEET)})
    feet_gait = RewTerm(func=mdp.feet_gait, weight=1.0,
                        params={"command_name": "body",
                                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)})
    feet_air_time = RewTerm(func=vel.feet_air_time_positive_biped, weight=1.0,
                            params={"command_name": "body", "threshold": 0.35,
                                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})
    feet_slide = RewTerm(func=vel.feet_slide, weight=-0.5,
                         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
                                 "asset_cfg": SceneEntityCfg("robot", body_names=FEET)})
    feet_clearance = RewTerm(func=mdp.feet_clearance, weight=-30.0,
                             params={"command_name": "body", "target_height": FOOT_Z + 0.06,
                                     "asset_cfg": SceneEntityCfg("robot", body_names=FEET)})
    feet_too_close = RewTerm(func=mdp.feet_too_close, weight=-10.0,
                             params={"min_distance": 0.12,
                                     "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True)})
    undesired_contacts = RewTerm(func=base.undesired_contacts, weight=-1.0,
                                 params={"threshold": 1.0,
                                         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee_link")})
    lin_vel_z = RewTerm(func=base.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy = RewTerm(func=base.ang_vel_xy_l2, weight=-0.05)
    torques = RewTerm(func=base.joint_torques_l2, weight=-2.0e-6,
                      params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(spec.LEG_JOINTS))})
    joint_acc = RewTerm(func=base.joint_acc_l2, weight=-1.0e-7,
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])})
    action_rate = RewTerm(func=base.action_rate_l2, weight=-0.008)
    joint_limits = RewTerm(func=base.joint_pos_limits, weight=-1.0,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*", ".*_knee_joint"])})
    hip_deviation = RewTerm(func=base.joint_deviation_l1, weight=-0.1,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint",
                                                                                      ".*_hip_roll_joint"])})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base.time_out, time_out=True)
    fall_contact = DoneTerm(func=base.illegal_contact,
                            params={"threshold": 1.0,
                                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FALL_BODIES)})
    tilted = DoneTerm(func=base.bad_orientation, params={"limit_angle": 1.0})
    too_low = DoneTerm(func=base.root_height_below_minimum, params={"minimum_height": 0.35})


@configclass
class R1WholeBodyEnvCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2,
                                                        gpu_max_rigid_patch_count=10 * 2**15))
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = spec.DECIMATION
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / spec.PHYSICS_HZ
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class R1WholeBodyEnvCfg_PLAY(R1WholeBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.events.push_robot = None
        self.events.hand_forces = None
        self.commands.body.ramp_steps = 0
        self.actions.upper_body.ramp_steps = 0
