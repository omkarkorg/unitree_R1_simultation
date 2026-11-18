#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

FACE_RATIO = 0.10
MIN_FACES, MAX_FACES = 200, 2500
SHARP_EDGE_DEG = 35.0
DEFAULT_COLOUR = (0.7, 0.7, 0.7)


def rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def origin_matrix(el) -> np.ndarray:
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        T[:3, :3] = rpy_matrix([float(v) for v in o.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return T


def resolve(urdf_dir: str, filename: str) -> str:
    if filename.startswith("package://"):
        filename = filename.split("/", 3)[-1]
    path = os.path.normpath(os.path.join(urdf_dir, filename))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"mesh not found: {filename} -> {path}")
    return path


def colour_of(visual, materials: dict):
    mat = visual.find("material")
    if mat is None:
        return DEFAULT_COLOUR
    c = mat.find("color")
    if c is not None:
        return tuple(float(v) for v in c.get("rgba").split()[:3])
    return materials.get(mat.get("name"), DEFAULT_COLOUR)


def light_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.merge_vertices()
    target = int(np.clip(len(mesh.faces) * FACE_RATIO, MIN_FACES, MAX_FACES))
    if len(mesh.faces) > target:
        mesh = mesh.simplify_quadric_decimation(face_count=target)
    return trimesh.graph.smooth_shade(mesh, angle=np.radians(SHARP_EDGE_DEG))


def build(urdf: str) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    root = ET.parse(urdf).getroot()
    urdf_dir = os.path.dirname(os.path.abspath(urdf))
    materials = {}
    for m in root.findall("material"):
        c = m.find("color")
        if c is not None:
            materials[m.get("name")] = tuple(float(v) for v in c.get("rgba").split()[:3])
    meta, verts, faces = [], [], []
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            T = origin_matrix(visual)
            if geom.find("mesh") is not None:
                m = geom.find("mesh")
                mesh = trimesh.load(resolve(urdf_dir, m.get("filename")), force="mesh")
                mesh.apply_scale([float(v) for v in m.get("scale", "1 1 1").split()])
            elif geom.find("box") is not None:
                mesh = trimesh.creation.box([float(v) for v in geom.find("box").get("size").split()])
            elif geom.find("cylinder") is not None:
                c = geom.find("cylinder")
                mesh = trimesh.creation.cylinder(float(c.get("radius")), float(c.get("length")), sections=24)
            elif geom.find("sphere") is not None:
                mesh = trimesh.creation.icosphere(2, float(geom.find("sphere").get("radius")))
            else:
                continue
            if len(mesh.faces) == 0:
                continue
            n0 = len(mesh.faces)
            mesh = light_mesh(mesh)
            mesh.apply_transform(T)
            meta.append({"link": link.get("name"), "colour": [round(v, 3) for v in colour_of(visual, materials)],
                         "faces": int(len(mesh.faces)), "faces_upstream": int(n0)})
            verts.append(np.asarray(mesh.vertices, dtype=np.float32))
            faces.append(np.asarray(mesh.faces, dtype=np.uint32))
    return meta, verts, faces


def main() -> int:
    ap = argparse.ArgumentParser(description="Light per-link meshes of the R1 + BrainCo model for the headset's lab view.")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta, verts, faces = build(a.urdf)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    arrays = {f"v{i}": v for i, v in enumerate(verts)}
    arrays.update({f"f{i}": f for i, f in enumerate(faces)})
    np.savez_compressed(a.out, meta=json.dumps(meta), **arrays)
    total = sum(m["faces"] for m in meta)
    upstream = sum(m["faces_upstream"] for m in meta)
    print(f"[build_vr_meshes] {len(meta)} meshes on {len({m['link'] for m in meta})} links: {total} triangles "
          f"(upstream {upstream}) -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    if not meta or total > 120000:
        print("[build_vr_meshes] unexpected result", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
