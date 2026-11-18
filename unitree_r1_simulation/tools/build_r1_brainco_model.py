#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

R1_URDF_NAME = "R1.urdf"
MESH_PREFIX_HAND = "brainco_"
HAND_MATERIALS = {"palm_fingers": ("brainco_dark", "0.2 0.2 0.2 1"), "adapter": ("brainco_adapter", "0.7 0.7 0.7 1")}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
EXPECTED_BODY_DOF = 26


def _load_joint_map(path: str):
    spec = importlib.util.spec_from_file_location("r1_joint_map", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mesh_elements(link: ET.Element):
    for tag in ("visual", "collision"):
        for el in link.findall(tag):
            geom = el.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is not None:
                yield el, mesh


def _resolve(src_dir: str, filename: str) -> str:
    if filename.startswith("package://"):
        filename = filename.split("/", 3)[-1]
    path = os.path.normpath(os.path.join(src_dir, filename))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"mesh not found: {path}")
    return path


def _copy_meshes(root: ET.Element, src_dir: str, out_mesh_dir: str, mesh_prefix: str, skip_links=()):
    for link in root.findall("link"):
        if link.get("name") in skip_links:
            continue
        for _, mesh in _mesh_elements(link):
            src = _resolve(src_dir, mesh.get("filename"))
            dst_name = os.path.basename(src)
            dst = os.path.join(out_mesh_dir, dst_name)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
            mesh.set("filename", mesh_prefix + dst_name)


def _clip_triangles_x_le(vertices: np.ndarray, faces: np.ndarray, cut_x: float):
    dist = cut_x - vertices[:, 0]
    inside = dist >= 0.0
    face_inside = inside[faces]
    keep_faces = faces[face_inside.all(axis=1)]
    crossing = faces[face_inside.any(axis=1) & ~face_inside.all(axis=1)]
    new_vertices = [vertices]
    next_index = len(vertices)
    new_faces = [keep_faces]
    for tri in crossing:
        poly = []
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            if inside[a]:
                poly.append(a)
            if inside[a] != inside[b]:
                t = dist[a] / (dist[a] - dist[b])
                new_vertices.append((vertices[a] + t * (vertices[b] - vertices[a]))[None, :])
                poly.append(next_index)
                next_index += 1
        for i in range(1, len(poly) - 1):
            new_faces.append(np.array([[poly[0], poly[i], poly[i + 1]]], dtype=faces.dtype))
    out = trimesh.Trimesh(np.vstack(new_vertices), np.vstack(new_faces), process=False)
    out.remove_unreferenced_vertices()
    return out


def _crop_mesh_keep_below_x(src: str, dst: str, cut_x: float) -> trimesh.Trimesh:
    mesh = trimesh.load(src, force="mesh", process=False)
    if mesh.vertices[:, 0].min() >= cut_x:
        raise ValueError(f"cut plane x={cut_x} removes the whole mesh {src}")
    cropped = _clip_triangles_x_le(np.asarray(mesh.vertices, dtype=np.float64),
                                   np.asarray(mesh.faces, dtype=np.int64), cut_x)
    if len(cropped.faces) == 0:
        raise RuntimeError(f"cropping {src} at x={cut_x} produced an empty mesh")
    if cropped.vertices[:, 0].max() > cut_x + 1e-9:
        raise RuntimeError(f"cropping {src} left vertices beyond x={cut_x}")
    cropped.export(dst, file_type="stl")
    return cropped


def _crop_wrist(link: ET.Element, side: str, src_dir: str, out_mesh_dir: str, mesh_prefix: str, cut_x: float):
    dst_name = f"{side}_wrist_roll_link_cropped.STL"
    dst = os.path.join(out_mesh_dir, dst_name)
    count = 0
    for el, mesh in _mesh_elements(link):
        origin = el.find("origin")
        if origin is not None:
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            if any(abs(v) > 1e-9 for v in xyz + rpy):
                raise RuntimeError(f"{link.get('name')}: non-identity mesh origin is not supported for cropping")
        if not os.path.exists(dst):
            cropped = _crop_mesh_keep_below_x(_resolve(src_dir, mesh.get("filename")), dst, cut_x)
            print(f"[build] cropped {link.get('name')} at x={cut_x:.3f} m -> {dst_name} "
                  f"(x range {cropped.vertices[:, 0].min():.4f}..{cropped.vertices[:, 0].max():.4f})")
        mesh.set("filename", mesh_prefix + dst_name)
        count += 1
    if count == 0:
        raise RuntimeError(f"{link.get('name')} has no mesh to crop")


def _add_material(root: ET.Element, name: str, rgba: str):
    for mat in root.findall("material"):
        if mat.get("name") == name:
            return
    mat = ET.Element("material", {"name": name})
    ET.SubElement(mat, "color", {"rgba": rgba})
    root.insert(0, mat)


def _xyz(joint: ET.Element) -> list[float]:
    origin = joint.find("origin")
    return [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]


def _rpy(joint: ET.Element) -> str:
    origin = joint.find("origin")
    return origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"


def _hand_link_name(side: str, name: str) -> str:
    if name == f"{side}_base2_link":
        return f"{side}_hand_adapter_link"
    if name == f"{side}_base_link":
        return f"{side}_hand_palm_link"
    m = re.fullmatch(rf"{side}_({'|'.join(FINGERS)})_tip(_link)?", name, flags=re.IGNORECASE)
    if m:
        return f"{side}_{m.group(1).lower()}_tip_link"
    if name.endswith("_Link"):
        return name[:-len("_Link")] + "_link"
    return name


def _append_hand(root: ET.Element, side: str, brainco_urdf: str, out_mesh_dir: str, mesh_prefix: str,
                 cut_x: float, jm, strip_mimic: bool):
    up = ET.parse(brainco_urdf).getroot()
    src_dir = os.path.dirname(brainco_urdf)
    links = {l.get("name"): l for l in up.findall("link")}
    joints = {j.get("name"): j for j in up.findall("joint")}
    children: dict[str, list[ET.Element]] = {}
    for j in joints.values():
        children.setdefault(j.find("parent").get("link"), []).append(j)

    adapter_j, palm_j = joints.get(f"{side}_base2_joint"), joints.get(f"{side}_base_joint")
    for j, child in ((adapter_j, f"{side}_base2_link"), (palm_j, f"{side}_base_link")):
        if j is None or j.get("type") != "fixed" or j.find("parent").get("link") != f"{side}_wrist_yaw_link" \
                or j.find("child").get("link") != child:
            raise RuntimeError(f"unexpected upstream mount for {child} - check {brainco_urdf}")
    if _rpy(adapter_j).split() != ["0", "0", "0"] or any(abs(v) > 1e-9 for v in _xyz(adapter_j)[1:]) \
            or any(abs(v) > 1e-9 for v in _xyz(palm_j)[1:]):
        raise RuntimeError("upstream adapter/palm mount is not a pure offset along the wrist x axis")
    shift = cut_x - _xyz(adapter_j)[0]

    hand_links, hand_joints, stack = [f"{side}_base_link"], [], [f"{side}_base_link"]
    while stack:
        for j in children.get(stack.pop(), []):
            hand_joints.append(j)
            hand_links.append(j.find("child").get("link"))
            stack.append(hand_links[-1])
    hand_links.append(f"{side}_base2_link")

    tip_links = set()
    for name in hand_links:
        link = copy.deepcopy(links[name])
        new_name = _hand_link_name(side, name)
        link.set("name", new_name)
        if new_name.endswith("_tip_link"):
            tip_links.add(new_name)
            for tag in ("visual", "collision"):
                for el in link.findall(tag):
                    link.remove(el)
        material = HAND_MATERIALS["adapter" if name.endswith("_base2_link") else "palm_fingers"][0]
        for el, mesh in _mesh_elements(link):
            src = _resolve(src_dir, mesh.get("filename"))
            dst_name = MESH_PREFIX_HAND + os.path.basename(src)
            shutil.copyfile(src, os.path.join(out_mesh_dir, dst_name))
            mesh.set("filename", mesh_prefix + dst_name)
            if el.tag == "visual":
                for mat in el.findall("material"):
                    el.remove(mat)
                ET.SubElement(el, "material", {"name": material})
        root.append(link)

    for j in hand_joints:
        joint = copy.deepcopy(j)
        joint.find("parent").set("link", _hand_link_name(side, joint.find("parent").get("link")))
        joint.find("child").set("link", _hand_link_name(side, joint.find("child").get("link")))
        if joint.find("child").get("link") in tip_links:
            finger = joint.find("child").get("link")[len(side) + 1:-len("_tip_link")]
            joint.set("name", f"{side}_{finger}_tip_joint")
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics", "mimic", "safety_controller"):
                for el in joint.findall(tag):
                    joint.remove(el)
        root.append(joint)

    for j, new_child in ((adapter_j, f"{side}_hand_adapter_link"), (palm_j, f"{side}_hand_palm_link")):
        name = f"{side}_hand_{'adapter' if new_child.endswith('adapter_link') else 'palm'}_joint"
        mount = ET.Element("joint", {"name": name, "type": "fixed"})
        ET.SubElement(mount, "origin", {"xyz": f"{_xyz(j)[0] + shift:.6f} 0 0", "rpy": _rpy(j)})
        ET.SubElement(mount, "parent", {"link": f"{side}_wrist_roll_link"})
        ET.SubElement(mount, "child", {"link": new_child})
        root.append(mount)

    by_name = {j.get("name"): j for j in root.findall("joint")}
    for motor, upper in zip(jm.HAND_MOTOR_JOINTS, jm.HAND_MOTOR_RANGES):
        limit = by_name[f"{side}_{motor}_joint"].find("limit")
        if abs(float(limit.get("lower")) - 0.0) > 1e-9 or abs(float(limit.get("upper")) - upper) > 1e-6:
            raise RuntimeError(f"{side}_{motor}_joint limits {limit.attrib} differ from joint_map.HAND_MOTOR_RANGES")
    for coupled, (motor_index, ratio) in jm.HAND_COUPLED_JOINTS.items():
        joint = by_name[f"{side}_{coupled}_joint"]
        mimic = joint.find("mimic")
        master = f"{side}_{jm.HAND_MOTOR_JOINTS[motor_index]}_joint"
        if mimic is None or mimic.get("joint") != master or abs(float(mimic.get("multiplier", "1")) - ratio) > 1e-6 \
                or abs(float(mimic.get("offset", "0"))) > 1e-9:
            raise RuntimeError(f"{joint.get('name')}: upstream mimic differs from joint_map.HAND_COUPLED_JOINTS")
        if strip_mimic:
            joint.remove(mimic)
    return shift


def _validate(root: ET.Element, out_dir: str, jm, strip_mimic: bool):
    links = [l.get("name") for l in root.findall("link")]
    joints = root.findall("joint")
    names = [j.get("name") for j in joints]
    if len(set(links)) != len(links):
        raise RuntimeError("duplicate link names")
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate joint names")
    mats = [m.get("name") for m in root.findall("material")]
    if len(set(mats)) != len(mats):
        raise RuntimeError("duplicate global material names")
    children = [j.find("child").get("link") for j in joints]
    if len(set(children)) != len(children):
        raise RuntimeError("a link has more than one parent joint")
    roots = [l for l in links if l not in set(children)]
    if roots != ["pelvis_link"]:
        raise RuntimeError(f"expected single root 'pelvis_link', got {roots}")
    for j in joints:
        for key in ("parent", "child"):
            if j.find(key).get("link") not in links:
                raise RuntimeError(f"joint {j.get('name')} references unknown link")
    movable = [j.get("name") for j in joints if j.get("type") in ("revolute", "continuous", "prismatic")]
    hand_expected = [n for side in jm.HAND_SIDES for n in jm.hand_joint_names(side)]
    hand = [n for n in movable if n in hand_expected]
    body = [n for n in movable if n not in hand_expected]
    if sorted(hand) != sorted(hand_expected):
        raise RuntimeError(f"hand joints differ from joint_map: {sorted(set(hand) ^ set(hand_expected))}")
    if sorted(body) != sorted(jm.R1_BODY_JOINT_SLOTS):
        raise RuntimeError(f"body joints differ from joint_map: {sorted(set(body) ^ set(jm.R1_BODY_JOINT_SLOTS))}")
    if len(body) != EXPECTED_BODY_DOF:
        raise RuntimeError(f"unexpected body DoF {len(body)}")
    if strip_mimic and any(j.find("mimic") is not None for j in joints):
        raise RuntimeError("mimic tags left although --strip-mimic was given")
    for link in root.findall("link"):
        for _, mesh in _mesh_elements(link):
            fn = mesh.get("filename")
            if fn.startswith("package://"):
                fn = "meshes/" + os.path.basename(fn)
            if not os.path.isfile(os.path.join(out_dir, fn)):
                raise RuntimeError(f"missing mesh file {fn}")
    return len(body), len(hand)


def build(r1_dir: str, brainco_urdf: str, joint_map_path: str, out_dir: str, cut_x: float, mesh_prefix: str,
          urdf_name: str, strip_mimic: bool) -> str:
    jm = _load_joint_map(joint_map_path)
    out_mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(out_mesh_dir, exist_ok=True)
    tree = ET.parse(os.path.join(r1_dir, R1_URDF_NAME))
    root = tree.getroot()
    root.set("name", "r1_brainco")
    wrists = {f"{s}_wrist_roll_link" for s in ("left", "right")}
    _copy_meshes(root, r1_dir, out_mesh_dir, mesh_prefix, skip_links=wrists)
    for link in root.findall("link"):
        if link.get("name") in wrists:
            _crop_wrist(link, link.get("name").split("_")[0], r1_dir, out_mesh_dir, mesh_prefix, cut_x)
    for name, rgba in HAND_MATERIALS.values():
        _add_material(root, name, rgba)
    for side in jm.HAND_SIDES:
        shift = _append_hand(root, side, brainco_urdf, out_mesh_dir, mesh_prefix, cut_x, jm, strip_mimic)
    print(f"[build] BrainCo adapter + hand mounted at x={cut_x:.4f} m in wrist_roll_link "
          f"(G1 mount shifted by {shift:+.4f} m along the wrist axis)")
    ET.indent(tree, space="  ")
    out_path = os.path.join(out_dir, urdf_name)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    body, hand = _validate(ET.parse(out_path).getroot(), out_dir, jm, strip_mimic)
    print(f"[build] wrote {out_path}: {body} body DoF + {hand} hand joints "
          f"({len(jm.HAND_SIDES) * jm.HAND_NUM_MOTORS} motors; coupled joints "
          f"{'driven by the simulator' if strip_mimic else 'as URDF <mimic>'})")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Compose a Unitree R1 (26 DoF) + 2x BrainCo Revo2 hand (6 motors, 11 joints each) URDF.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--r1-dir", required=True, help="unitree_ros/robots/r1_description")
    parser.add_argument("--brainco-urdf", required=True,
                        help="unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf")
    parser.add_argument("--joint-map", required=True, help="r1_sim/r1_sim/joint_map.py (hand joint definitions)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--wrist-cut-x", type=float, default=0.078,
                        help="forearm cut / hand adapter position along wrist_roll_link x [m]")
    parser.add_argument("--mesh-prefix", default="meshes/",
                        help='prefix written into <mesh filename>, e.g. "meshes/" or "package://r1_description/meshes/"')
    parser.add_argument("--urdf-name", default="r1_brainco.urdf")
    parser.add_argument("--strip-mimic", action="store_true",
                        help="remove the <mimic> tags (the simulator drives the coupled joints itself)")
    args = parser.parse_args()
    build(args.r1_dir, args.brainco_urdf, args.joint_map, args.out_dir, args.wrist_cut_x, args.mesh_prefix,
          args.urdf_name, args.strip_mimic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
