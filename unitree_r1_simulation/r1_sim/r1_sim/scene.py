from __future__ import annotations

import math
from dataclasses import MISSING, dataclass

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg, PhysxRigidBodyCfg

from . import loco_spec
from .dds_bridge import HAND_DRIVE_KD, HAND_DRIVE_KP
from .joint_map import HAND_SIDES, body_joint_names, hand_joint_names
from .layout import TABLE_TOP_Z, floor_box, scene_boxes, scene_fixtures, scene_objects
from .math_utils import camera_quat_ros, horizontal_aperture_for_vfov, quat_from_axis_angle, quat_mul, wxyz_to_xyzw

OBJECT_MASS = 0.08
FLOOR_RESET_Z = 0.06
IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)


@dataclass
class CameraOptions:
    width: int = 640
    height: int = 480
    pitch_down_deg: float = 50.0
    vfov_deg: float = 70.0
    stereo: bool = False
    baseline: float = 0.064
    mount_xyz: tuple[float, float, float] = (0.095, 0.0, 0.0)


HOLD_GAINS = {
    "legs": (200.0, 5.0),
    "waist": (200.0, 5.0),
    "arms": (60.0, 2.0),
    "head": (20.0, 1.0),
    "hands": (HAND_DRIVE_KP, HAND_DRIVE_KD),
}


def _hand_joint_names() -> list[str]:
    return [name for side in HAND_SIDES for name in hand_joint_names(side)]


def robot_spawn_cfg(usd_path: str, fixed: bool = True) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=usd_path,
        fix_root_link=fixed,
        rigid_props=PhysxRigidBodyCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
        ],
    )


def robot_cfg(usd_path: str, pelvis_height: float, floating: bool = False, stance: bool = False) -> ArticulationCfg:
    joint_pos = {name: 0.0 for name in body_joint_names() + _hand_joint_names()}
    legs = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
    leg_kp, leg_kd = HOLD_GAINS["legs"]
    if floating or stance:
        joint_pos.update(loco_spec.DEFAULT_POSE)
        legs = list(loco_spec.LEG_JOINTS)
        leg_kp = {n: loco_spec.LEG_GAINS[n][0] for n in legs}
        leg_kd = {n: loco_spec.LEG_GAINS[n][1] for n in legs}
        pelvis_height = loco_spec.H_NOM + (0.005 if floating else 0.0)
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=robot_spawn_cfg(usd_path, fixed=not floating),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, pelvis_height),
            rot=IDENTITY_XYZW,
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=legs, stiffness=leg_kp, damping=leg_kd, armature=loco_spec.ARMATURE,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*_joint"],
                stiffness=HOLD_GAINS["waist"][0], damping=HOLD_GAINS["waist"][1], armature=0.01,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"],
                stiffness=HOLD_GAINS["arms"][0], damping=HOLD_GAINS["arms"][1], armature=0.01,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_.*_joint"],
                stiffness=HOLD_GAINS["head"][0], damping=HOLD_GAINS["head"][1], armature=0.01,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=_hand_joint_names(),
                stiffness=HOLD_GAINS["hands"][0], damping=HOLD_GAINS["hands"][1], armature=0.001,
            ),
        },
    )


def _object_props() -> dict:
    return dict(
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
            PhysxRigidBodyCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
        ],
        mass_props=sim_utils.MassCfg(mass=OBJECT_MASS),
        collision_props=[sim_utils.UsdPhysicsCollisionCfg(), PhysxCollisionCfg(contact_offset=0.005, rest_offset=0.0)],
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2, dynamic_friction=1.0, restitution=0.0,
            friction_combine_mode="max", restitution_combine_mode="min",
        ),
    )


def _static_box(prim_path: str, size, pos, color, friction=0.8) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=IDENTITY_XYZW),
    )


def head_camera_cfg(head_link_path: str, name: str, y_offset: float, opts: CameraOptions) -> CameraCfg:
    focal_length = 10.0
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{head_link_path}/{name}",
        update_period=0.0,
        width=opts.width,
        height=opts.height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture_for_vfov(
                math.radians(opts.vfov_deg), focal_length, opts.width, opts.height),
            clipping_range=(0.02, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(opts.mount_xyz[0], opts.mount_xyz[1] + y_offset, opts.mount_xyz[2]),
            rot=tuple(float(v) for v in wxyz_to_xyzw(camera_quat_ros(math.radians(opts.pitch_down_deg)))),
            convention="ros",
        ),
    )


def debug_camera_cfg() -> CameraCfg:
    width, height, focal_length = 640, 480, 10.0
    look_left = quat_mul(quat_from_axis_angle([0.0, 0.0, 1.0], math.pi / 2.0), camera_quat_ros(math.radians(14.0)))
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/DebugCam",
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture_for_vfov(math.radians(50.0), focal_length, width, height),
            clipping_range=(0.02, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.30, -0.70, 0.80), rot=tuple(float(v) for v in wxyz_to_xyzw(look_left)),
                                   convention="ros"),
    )


@configclass
class ConvexBodyCfg(sim_utils.SpawnerCfg):
    func: object = "r1_sim.assembly_usd:spawn_convex_body"
    pieces: list = MISSING
    colour: tuple = (0.5, 0.5, 0.5)
    mass: float = 0.0
    friction: float = 0.5
    material_name: str = "assembly_parts"
    contact_offset: float = 0.002
    position_iterations: int = 16
    piece_names: list | None = None


def _convex_cfg(pieces, colour, mass, thread: bool, piece_names=None) -> ConvexBodyCfg:
    from .assembly import PART_FRICTION, THREAD_FRICTION
    return ConvexBodyCfg(pieces=[[list(map(float, p)) for p in piece] for piece in pieces], colour=tuple(colour),
                         mass=float(mass), piece_names=piece_names,
                         friction=THREAD_FRICTION if thread else PART_FRICTION,
                         material_name="assembly_threads" if thread else "assembly_parts",
                         contact_offset=0.0003 if thread else 0.002)


def build_scene_cfg(robot_usd_path: str, pelvis_height: float, cam: CameraOptions, head_link_path: str,
                    debug_cam: bool = False, table_z: float = TABLE_TOP_Z, scene: str = "pick_place",
                    floating: bool = False, stance: bool = False):
    static = {b["name"]: b for b in scene_boxes(scene, table_z) + [floor_box()]}

    def box(prim, name, friction=0.8):
        b = static[name]
        return _static_box(prim, tuple(b["size"]), tuple(b["center"]), tuple(b["colour"]), friction=friction)

    @configclass
    class R1TabletopSceneCfg(InteractiveSceneCfg):
        floor = box("/World/Floor", "floor", friction=1.0)
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=600.0, color=(0.9, 0.9, 0.9)),
        )
        sun_light = AssetBaseCfg(
            prim_path="/World/SunLight",
            spawn=sim_utils.DistantLightCfg(intensity=850.0, color=(1.0, 0.98, 0.95), angle=1.0),
            init_state=AssetBaseCfg.InitialStateCfg(rot=(0.0, 0.3007058, 0.0, 0.9537170)),
        )

        robot = robot_cfg(robot_usd_path, pelvis_height, floating=floating, stance=stance)

        head_cam = head_camera_cfg(head_link_path, "head_cam", +cam.baseline / 2.0 if cam.stereo else 0.0, cam)
        head_cam_right: CameraCfg | None = (
            head_camera_cfg(head_link_path, "head_cam_right", -cam.baseline / 2.0, cam) if cam.stereo else None
        )
        debug_cam_side: CameraCfg | None = debug_camera_cfg() if debug_cam else None

    cfg = R1TabletopSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    for name in static:
        if name != "floor":
            setattr(cfg, name, box(f"{{ENV_REGEX_NS}}/{_prim_name(name)}", name))
    for fx in scene_fixtures(scene, table_z):
        setattr(cfg, fx.name, AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/{_prim_name(fx.name)}",
                                           spawn=_convex_cfg(fx.pieces, fx.colour, 0.0, fx.thread, fx.piece_names)))
    for o in scene_objects(scene, table_z):
        init = RigidObjectCfg.InitialStateCfg(pos=tuple(o["start"][:3]), rot=tuple(o["start"][3:]))
        if o["shape"] == "convex":
            spawn = _convex_cfg(o["pieces"], o["colour"], o["mass"], o["thread"], o.get("piece_names"))
        elif o["shape"] == "cylinder":
            spawn = sim_utils.CylinderCfg(
                radius=o["radius"], height=o["height"], axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(o["colour"]), roughness=0.4),
                **_object_props())
        else:
            spawn = sim_utils.CuboidCfg(
                size=tuple(o["size"]),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(o["colour"]), roughness=0.4),
                **_object_props())
        setattr(cfg, o["name"], RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/{_prim_name(o['name'])}", spawn=spawn,
                                               init_state=init))
    return cfg


def _prim_name(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in name.split("_"))


def filter_thread_contacts(env_path: str = "/World/envs/env_0") -> int:
    from .assembly import THREAD_CONTACTS
    from .assembly_usd import filter_contacts
    c = THREAD_CONTACTS
    return filter_contacts(f"{env_path}/{_prim_name(c['nut'])}", f"{env_path}/{_prim_name(c['bolt'])}", c["filtered"])
