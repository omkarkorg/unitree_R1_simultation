from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET

from .joint_map import HAND_SIDES, hand_joint_names

VARIANTS = ("fixed", "floating", "train")


def robot_urdf_converter_cfg(urdf_path: str, usd_dir: str, fix_base: bool = True):
    import isaaclab.sim as sim_utils

    return sim_utils.UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_dir,
        fix_base=fix_base,
        merge_fixed_joints=True,
        make_instanceable=False,
        self_collision=False,
        collision_type="Convex Hull",
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )


def write_welded_hand_urdf(src_urdf: str, dst_urdf: str) -> str:
    tree = ET.parse(src_urdf)
    finger_joints = {n for side in HAND_SIDES for n in hand_joint_names(side)}
    welded = 0
    for joint in tree.getroot().findall("joint"):
        if joint.get("name") in finger_joints:
            joint.set("type", "fixed")
            for tag in ("limit", "axis", "dynamics", "mimic"):
                for el in joint.findall(tag):
                    joint.remove(el)
            welded += 1
    if welded != len(finger_joints):
        raise RuntimeError(f"welded {welded} finger joints, expected {len(finger_joints)}")
    src_dir = os.path.dirname(os.path.abspath(src_urdf))
    for mesh in tree.getroot().iter("mesh"):
        name = mesh.get("filename", "")
        if not os.path.isabs(name) and "://" not in name:
            mesh.set("filename", os.path.normpath(os.path.join(src_dir, name)))
    os.makedirs(os.path.dirname(os.path.abspath(dst_urdf)), exist_ok=True)
    tree.write(dst_urdf, encoding="utf-8", xml_declaration=True)
    return dst_urdf


def convert_robot_urdf(urdf_path: str, cache_dir: str, variant: str = "fixed", log=print) -> str:
    from isaaclab.sim.converters import UrdfConverter
    from isaaclab.utils import to_dict

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    folder = cache_dir if variant == "fixed" else os.path.join(cache_dir, variant)
    if variant == "train":
        urdf_path = write_welded_hand_urdf(urdf_path, os.path.join(folder, "urdf", "r1_brainco_welded_hands.urdf"))
    cfg = robot_urdf_converter_cfg(urdf_path, folder, fix_base=variant == "fixed")
    settings = {k: v for k, v in to_dict(cfg).items() if k not in ("asset_path", "usd_dir", "usd_file_name")}
    digest = hashlib.md5(json.dumps(settings, sort_keys=True, default=str).encode())
    with open(urdf_path, "rb") as f:
        digest.update(f.read())
    key = digest.hexdigest()[:16]
    os.makedirs(folder, exist_ok=True)
    stale = [e for e in os.listdir(folder)
             if e != key and (re.fullmatch(r"[0-9a-f]{16}|r1_(dex3|brainco)(_\d+)?", e)
                              or e in (".asset_hash", "config.yaml"))]
    for entry in stale:
        path = os.path.join(folder, entry)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    if stale:
        log(f"removed {len(stale)} outdated entries from the USD cache ({variant})")
    cfg.usd_dir = os.path.join(folder, key)
    if os.path.isdir(cfg.usd_dir):
        log(f"robot USD cache ({variant}): {cfg.usd_dir}")
    else:
        log(f"converting {urdf_path} to USD in {cfg.usd_dir} ({variant}, first start only)")
    return UrdfConverter(cfg).usd_path


def find_link_relpath(usd_path: str, link_name: str) -> str:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    if not root:
        raise RuntimeError(f"{usd_path} has no default prim")
    matches = [p for p in stage.Traverse() if p.GetName() == link_name]
    bodies = [p for p in matches if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    candidates = sorted(bodies or matches, key=lambda p: len(str(p.GetPath())))
    if not candidates:
        raise RuntimeError(f"link '{link_name}' not found in {usd_path}")
    root_path, path = str(root.GetPath()), str(candidates[0].GetPath())
    if not path.startswith(root_path + "/"):
        raise RuntimeError(f"link prim {path} is not below the default prim {root_path}")
    return path[len(root_path) + 1:]
