from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from r1_sim import loco_spec as spec

URDF = os.environ.get("R1_URDF", "/opt/r1_model/r1_brainco.urdf")
USD_CACHE = os.environ.get("R1_USD_CACHE", "/root/.cache/r1_sim/usd")


@configclass
class R1SpawnerCfg(sim_utils.UsdFileCfg):
    func: object = "r1_loco.spawn:spawn_r1"
    usd_path: str = ""
    urdf_path: str = URDF
    cache_dir: str = USD_CACHE
    variant: str = "train"


def _gains(names, gains):
    return {n: gains[n][0] for n in names}, {n: gains[n][1] for n in names}


def r1_cfg(variant: str = "train") -> ArticulationCfg:
    leg_kp, leg_kd = _gains(spec.LEG_JOINTS, spec.LEG_GAINS)
    waist = [n for n in spec.UPPER_JOINTS if "waist" in n]
    arms = [n for n in spec.UPPER_JOINTS if any(k in n for k in ("shoulder", "elbow", "wrist"))]
    head = [n for n in spec.UPPER_JOINTS if "head" in n]
    up = {g: _gains(names, spec.UPPER_HOLD_GAINS) for g, names in (("waist", waist), ("arms", arms), ("head", head))}
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=R1SpawnerCfg(
            variant=variant,
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyCfg(disable_gravity=False, retain_accelerations=False, linear_damping=0.0,
                                          angular_damping=0.0, max_linear_velocity=1000.0,
                                          max_angular_velocity=1000.0, max_depenetration_velocity=1.0),
            articulation_props=[PhysxArticulationCfg(enabled_self_collisions=False, solver_position_iteration_count=8,
                                                     solver_velocity_iteration_count=4)],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, spec.H_NOM + 0.02),
            joint_pos=dict(spec.DEFAULT_POSE),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "legs": ImplicitActuatorCfg(joint_names_expr=list(spec.LEG_JOINTS), stiffness=leg_kp, damping=leg_kd,
                                        armature=spec.ARMATURE),
            "waist": ImplicitActuatorCfg(joint_names_expr=waist, stiffness=up["waist"][0], damping=up["waist"][1],
                                         armature=spec.ARMATURE),
            "arms": ImplicitActuatorCfg(joint_names_expr=arms, stiffness=up["arms"][0], damping=up["arms"][1],
                                        armature=spec.ARMATURE),
            "head": ImplicitActuatorCfg(joint_names_expr=head, stiffness=up["head"][0], damping=up["head"][1],
                                        armature=spec.ARMATURE),
        },
    )
