from __future__ import annotations

import math

import numpy as np


def hull_mesh(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import ConvexHull

    pts = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(pts)
    used = np.unique(hull.simplices)
    remap = -np.ones(len(pts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts = pts[used]
    faces = remap[hull.simplices]
    centre = verts.mean(axis=0)
    for f in faces:
        a, b, c = verts[f]
        if np.dot(np.cross(b - a, c - a), a - centre) < 0.0:
            f[1], f[2] = f[2], f[1]
    return verts, faces


def prism_points(poly_xy, z0: float, z1: float) -> np.ndarray:
    p = np.asarray(poly_xy, dtype=np.float64)
    return np.vstack([np.c_[p, np.full(len(p), z0)], np.c_[p, np.full(len(p), z1)]])


def regular_polygon(n: int, apothem: float, phase: float = 0.0) -> np.ndarray:
    circum = apothem / math.cos(math.pi / n)
    a = phase + math.pi / n + 2.0 * math.pi * np.arange(n) / n
    return np.c_[circum * np.cos(a), circum * np.sin(a)]


def box_points(size, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    sx, sy, sz = (s / 2.0 for s in size)
    c = np.asarray(center, dtype=np.float64)
    return np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]) + c


def _ray_to_polygon(direction: np.ndarray, poly: np.ndarray) -> np.ndarray:
    t_best = np.inf
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        normal = np.array([b[1] - a[1], a[0] - b[0]])
        den = float(normal @ direction)
        if den > 1e-12:
            t_best = min(t_best, float(normal @ a) / den)
    return direction * t_best


def plate_with_hole(hole_poly, half: float | None, z0: float, z1: float, chamfer: float = 0.0,
                    center=(0.0, 0.0), outer_poly=None) -> list[np.ndarray]:
    hole = np.asarray(hole_poly, dtype=np.float64)
    if outer_poly is None:
        outer_poly = [[half, half], [-half, half], [-half, -half], [half, -half]]
    outer = np.asarray(outer_poly, dtype=np.float64)
    e0, e1 = outer[1] - outer[0], outer[2] - outer[1]
    if e0[0] * e1[1] - e0[1] * e1[0] < 0:
        outer = outer[::-1]
    corner_ang = np.arctan2(outer[:, 1], outer[:, 0])
    n = len(hole)
    pieces = []
    for k in range(n):
        p0, p1 = hole[k], hole[(k + 1) % n]
        a0, a1 = math.atan2(p0[1], p0[0]), math.atan2(p1[1], p1[0])
        span = (a1 - a0) % (2.0 * math.pi)
        rim = [_ray_to_polygon(p0, outer), _ray_to_polygon(p1, outer)]
        rim += [c for c, ca in zip(outer, corner_ang) if 0.0 < (ca - a0) % (2.0 * math.pi) < span]
        rim = np.array(rim)
        pts = [prism_points(rim, z0, z1)]
        if chamfer > 0.0:
            zc = max(z0, z1 - chamfer)
            pts.append(prism_points(np.array([p0, p1]), z0, zc))
            grow = lambda p: p * (1.0 + chamfer / np.linalg.norm(p))
            pts.append(np.array([[*grow(p0), z1], [*grow(p1), z1]]))
        else:
            pts.append(prism_points(np.array([p0, p1]), z0, z1))
        piece = np.vstack(pts)
        piece[:, :2] += np.asarray(center, dtype=np.float64)
        pieces.append(piece)
    return pieces


def ring_pieces(r_in: float, r_out: float, z0: float, z1: float, n: int = 16) -> list[np.ndarray]:
    inner = regular_polygon(n, r_in)
    outer = regular_polygon(n, r_out)
    pieces = []
    for k in range(n):
        quad = np.array([inner[k], inner[(k + 1) % n], outer[(k + 1) % n], outer[k]])
        pieces.append(prism_points(quad, z0, z1))
    return pieces


def gear_pieces(module: float, teeth: int, bore_apothem: float, z0: float, z1: float) -> list[np.ndarray]:
    rp = module * teeth / 2.0
    r_root, r_tip = rp - 1.25 * module, rp + module
    pitch_ang = 2.0 * math.pi / teeth
    pieces = []
    r_bore = bore_apothem / math.cos(pitch_ang / 2.0)
    bore = [(r_bore * math.cos((k - 0.5) * pitch_ang), r_bore * math.sin((k - 0.5) * pitch_ang)) for k in range(teeth)]
    for k in range(teeth):
        c = k * pitch_ang
        a0, a1 = c - pitch_ang / 2.0, c + pitch_ang / 2.0
        r_hub = r_root / math.cos(pitch_ang / 2.0)
        sector = np.array([bore[k], bore[(k + 1) % teeth], [r_hub * math.cos(a1), r_hub * math.sin(a1)],
                           [r_hub * math.cos(a0), r_hub * math.sin(a0)]])
        pieces.append(prism_points(sector, z0, z1))
        w_root, w_tip = 0.30 * pitch_ang, 0.13 * pitch_ang
        r_base = r_root * 0.98
        tooth = np.array([[r_base * math.cos(c - w_root), r_base * math.sin(c - w_root)],
                          [r_tip * math.cos(c - w_tip), r_tip * math.sin(c - w_tip)],
                          [r_tip * math.cos(c + w_tip), r_tip * math.sin(c + w_tip)],
                          [r_base * math.cos(c + w_root), r_base * math.sin(c + w_root)]])
        pieces.append(prism_points(tooth, z0, z1))
    return pieces


def thread_ridge_pieces(r_root: float, r_crest: float, pitch: float, z_start: float, turns: float,
                        seg_deg: float = 10.0, z_min: float = -np.inf, z_max: float = np.inf) -> list[np.ndarray]:
    pieces = []
    n = int(round(turns * 360.0 / seg_deg))
    for k in range(n):
        pts = []
        for phi in (math.radians(k * seg_deg), math.radians((k + 1) * seg_deg)):
            zc = z_start + pitch * phi / (2.0 * math.pi)
            c, s_ = math.cos(phi), math.sin(phi)
            for r, half in ((r_root, 0.375 * pitch), (r_crest, 0.125 * pitch)):
                pts += [[r * c, r * s_, zc - half], [r * c, r * s_, zc + half]]
        pts = np.array(pts)
        if pts[:, 2].min() >= z_min and pts[:, 2].max() <= z_max:
            pieces.append(pts)
    return pieces


def thread_profile(u: np.ndarray) -> np.ndarray:
    u = np.mod(u, 1.0)
    return np.clip(np.minimum((u - 0.125) / 0.25, (0.875 - u) / 0.25), 0.0, 1.0)


def threaded_rod(r_minor: float, depth: float, pitch: float, z0: float, z1: float, lead_in: float = 0.004,
                 n_theta: int = 72, per_pitch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    m = max(2, int(math.ceil((z1 - z0) / pitch * per_pitch)))
    th = 2.0 * np.pi * np.arange(n_theta) / n_theta
    z = np.linspace(z0, z1, m + 1)
    T, Z = np.meshgrid(th, z)
    r = r_minor + depth * thread_profile(Z / pitch - T / (2.0 * np.pi))
    r = np.minimum(r, r_minor + np.maximum(z1 - Z, 0.0) / lead_in * depth)
    verts = np.c_[(r * np.cos(T)).ravel(), (r * np.sin(T)).ravel(), Z.ravel()]
    faces = []
    idx = lambda j, i: j * n_theta + (i % n_theta)
    for j in range(m):
        for i in range(n_theta):
            a, b, c, d = idx(j, i), idx(j, i + 1), idx(j + 1, i + 1), idx(j + 1, i)
            faces += [(a, b, c), (a, c, d)]
    bottom, top = len(verts), len(verts) + 1
    verts = np.vstack([verts, [0.0, 0.0, z0], [0.0, 0.0, z1]])
    for i in range(n_theta):
        faces.append((bottom, idx(0, i + 1), idx(0, i)))
        faces.append((top, idx(m, i), idx(m, i + 1)))
    return verts, np.array(faces, dtype=np.int64)


def threaded_nut(r_minor: float, depth: float, pitch: float, height: float, across_flats: float,
                 clearance: float, countersink: float = 0.002, n_theta: int = 72, per_pitch: int = 16
                 ) -> tuple[np.ndarray, np.ndarray]:
    if n_theta % 12:
        raise ValueError("n_theta must be a multiple of 12 (the hex corners are sampled)")
    m = max(2, int(math.ceil(height / pitch * per_pitch)))
    th = 2.0 * np.pi * np.arange(n_theta) / n_theta
    z = np.linspace(0.0, height, m + 1)
    T, Z = np.meshgrid(th, z)
    r = r_minor + clearance + depth * thread_profile(Z / pitch - T / (2.0 * np.pi))
    r_major = r_minor + clearance + depth
    edge = np.minimum(Z, height - Z)
    r = np.maximum(r, r_major + countersink - edge)
    inner = np.c_[(r * np.cos(T)).ravel(), (r * np.sin(T)).ravel(), Z.ravel()]
    apothem = across_flats / 2.0
    rel = np.mod(th + math.pi / 6.0, math.pi / 3.0) - math.pi / 6.0
    r_hex = apothem / np.cos(rel)
    outer = np.vstack([np.c_[r_hex * np.cos(th), r_hex * np.sin(th), np.full(n_theta, zz)] for zz in (0.0, height)])
    verts = np.vstack([inner, outer])
    n_in = len(inner)
    idx = lambda j, i: j * n_theta + (i % n_theta)
    out = lambda k, i: n_in + k * n_theta + (i % n_theta)
    faces = []
    for j in range(m):
        for i in range(n_theta):
            a, b, c, d = idx(j, i), idx(j, i + 1), idx(j + 1, i + 1), idx(j + 1, i)
            faces += [(a, c, b), (a, d, c)]
    for i in range(n_theta):
        a, b, c, d = out(0, i), out(0, i + 1), out(1, i + 1), out(1, i)
        faces += [(a, b, c), (a, c, d)]
        faces += [(idx(0, i), idx(0, i + 1), out(0, i + 1)), (idx(0, i), out(0, i + 1), out(0, i))]
        faces += [(idx(m, i), out(1, i + 1), idx(m, i + 1)), (idx(m, i), out(1, i), out(1, i + 1))]
    return verts, np.array(faces, dtype=np.int64)
