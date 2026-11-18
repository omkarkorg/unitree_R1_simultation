from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import assembly_geometry as g

DENSITY = 1000.0
PART_FRICTION = 0.5
THREAD_FRICTION = 0.25
SIDE_TABLE_X = (-0.25, 0.45)
SIDE_TABLE_Y = (-0.95, -0.55)
SIDE_TABLE_THICKNESS = 0.03
SIDE_TABLE_LEG = 0.04
SIDE_TABLE_COLOUR = (0.62, 0.64, 0.66)
SIDE_ROWS_Y = (-0.65, -0.80)

PEG_SIZE = 0.029
PEG_LENGTH = 0.060
PEG_CLEARANCE = 0.002
PEG_BOARD_THICKNESS = 0.035
PEG_CELL = 0.068
PEG_CHAMFER = 0.003
GEAR_MODULE = 0.004
GEAR_TEETH = (20, 12, 16)
GEAR_THICKNESS = 0.015
GEAR_SHAFT_R = 0.007
GEAR_BORE_CLEARANCE = 0.001
GEAR_BACKLASH = 0.0008
GEAR_SHAFT_HEIGHT = 0.045
PLATE_THICKNESS = 0.012
RING_OUTER = (0.042, 0.035, 0.028)
RING_INNER = 0.014
RING_THICKNESS = 0.014
POST_R = 0.008
POST_HEIGHT = 0.075
POST_SPACING = 0.095
THREAD_R_MINOR = 0.012
THREAD_DEPTH = 0.003
THREAD_PITCH = 0.008
THREAD_CLEARANCE = 0.001
THREAD_SEGMENT_DEG = 15.0
BOLT_CORE_HEIGHT = 0.050
BOLT_THREAD_TURNS = 4.0
NUT_ACROSS_FLATS = 0.046
NUT_HEIGHT = 0.020
NUT_THREAD_TURNS = 2.5

COLOURS = {
    "peg_square": (0.85, 0.20, 0.15), "peg_round": (0.15, 0.45, 0.85), "peg_hex": (0.95, 0.75, 0.10),
    "gear_large": (0.20, 0.65, 0.30), "gear_small": (0.90, 0.45, 0.10), "gear_medium": (0.55, 0.30, 0.75),
    "ring_large": (0.85, 0.15, 0.25), "ring_medium": (0.95, 0.60, 0.10), "ring_small": (0.25, 0.70, 0.80),
    "nut": (0.72, 0.72, 0.75),
    "peg_board": (0.30, 0.30, 0.34), "gear_plate": (0.35, 0.33, 0.30), "hanoi_base": (0.42, 0.30, 0.22),
    "bolt": (0.60, 0.60, 0.63),
}


@dataclass
class Part:
    name: str
    pieces: list
    start: list
    colour: tuple
    thread: bool = False
    mass: float = field(default=0.0)
    piece_names: list | None = None

    def __post_init__(self):
        if not self.mass:
            self.mass = DENSITY * sum(_hull_volume(p) for p in self.pieces)


@dataclass
class Fixture:
    name: str
    pieces: list
    colour: tuple
    thread: bool = False
    piece_names: list | None = None


def _hull_volume(points) -> float:
    from scipy.spatial import ConvexHull

    return float(ConvexHull(np.asarray(points)).volume)


def _translate(pieces, offset):
    off = np.asarray(offset, dtype=np.float64)
    return [np.asarray(p, dtype=np.float64) + off for p in pieces]


def _quat_z(angle: float) -> list:
    return [0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)]


def table_layout(table_z: float) -> dict:
    return {"peg_board": (0.40, 0.31), "bolt": (0.40, 0.135), "gear_plate": (0.42, -0.045), "hanoi_base": (0.40, -0.34)}


def _gear_shaft_offsets() -> list[float]:
    radii = [GEAR_MODULE * z / 2.0 for z in GEAR_TEETH]
    ys = [0.0]
    for a, b in zip(radii, radii[1:]):
        ys.append(ys[-1] + a + b + GEAR_BACKLASH)
    return ys


def targets(table_z: float) -> dict:
    lay = table_layout(table_z)
    bx, by = lay["peg_board"]
    out = {f"peg_{k}": (bx, by + (i - 1) * PEG_CELL) for i, k in enumerate(("square", "round", "hex"))}
    px, py = lay["gear_plate"]
    ys = _gear_shaft_offsets()
    tip = [GEAR_MODULE * (z / 2.0 + 1.0) for z in GEAR_TEETH]
    y_first = py - (ys[-1] + tip[0] + tip[-1]) / 2.0 + tip[0]
    for name, y in zip(("gear_large", "gear_small", "gear_medium"), ys):
        out[name] = (px, y_first + y)
    hx, hy = lay["hanoi_base"]
    for k in (-1, 0, 1):
        out[f"post_{k + 1}"] = (hx, hy + k * POST_SPACING)
    out["bolt"] = lay["bolt"]
    out.update(peg_board_top=table_z + PEG_BOARD_THICKNESS, plate_top=table_z + PLATE_THICKNESS,
               shaft_top=table_z + PLATE_THICKNESS + GEAR_SHAFT_HEIGHT, post_top=table_z + PLATE_THICKNESS + POST_HEIGHT,
               bolt_top=table_z + PLATE_THICKNESS + BOLT_CORE_HEIGHT, gear_plate_length=ys[-1] + tip[0] + tip[-1] + 0.012)
    return out


def fixtures(table_z: float) -> list[Fixture]:
    lay = table_layout(table_z)
    at = targets(table_z)
    z0 = table_z
    out = []
    apothem = (PEG_SIZE + PEG_CLEARANCE) / 2.0
    holes = {"square": g.regular_polygon(4, apothem), "round": g.regular_polygon(24, apothem),
             "hex": g.regular_polygon(6, apothem)}
    pieces = []
    for kind, hole in holes.items():
        pieces += g.plate_with_hole(hole, PEG_CELL / 2.0, z0, z0 + PEG_BOARD_THICKNESS, chamfer=PEG_CHAMFER,
                                    center=at[f"peg_{kind}"])
    out.append(Fixture("peg_board", pieces, COLOURS["peg_board"]))
    px, py = lay["gear_plate"]
    pieces = [g.box_points((0.11, at["gear_plate_length"], PLATE_THICKNESS), (px, py, z0 + PLATE_THICKNESS / 2.0))]
    for name in ("gear_large", "gear_small", "gear_medium"):
        pieces.append(g.prism_points(g.regular_polygon(16, GEAR_SHAFT_R) + list(at[name]), z0, at["shaft_top"]))
    out.append(Fixture("gear_plate", pieces, COLOURS["gear_plate"]))
    hx, hy = lay["hanoi_base"]
    pieces = [g.box_points((0.10, 2 * POST_SPACING + 2 * RING_OUTER[0] + 0.02, PLATE_THICKNESS),
                           (hx, hy, z0 + PLATE_THICKNESS / 2.0))]
    for k in range(3):
        pieces.append(g.prism_points(g.regular_polygon(16, POST_R) + list(at[f"post_{k}"]), z0, at["post_top"]))
    out.append(Fixture("hanoi_base", pieces, COLOURS["hanoi_base"]))
    tx, ty = lay["bolt"]
    zb = z0 + PLATE_THICKNESS
    pieces = [g.box_points((0.07, 0.07, PLATE_THICKNESS), (tx, ty, z0 + PLATE_THICKNESS / 2.0)),
              g.prism_points(g.regular_polygon(24, THREAD_R_MINOR) + [tx, ty], zb, zb + BOLT_CORE_HEIGHT)]
    thread_top = zb + BOLT_CORE_HEIGHT - 0.004 - 0.375 * THREAD_PITCH
    ridge = _translate(g.thread_ridge_pieces(THREAD_R_MINOR - 0.0005, THREAD_R_MINOR + THREAD_DEPTH, THREAD_PITCH,
                                             thread_top - BOLT_THREAD_TURNS * THREAD_PITCH, BOLT_THREAD_TURNS,
                                             THREAD_SEGMENT_DEG), (tx, ty, 0.0))
    out.append(Fixture("bolt", pieces + ridge, COLOURS["bolt"], thread=True,
                       piece_names=["base", "core"] + [f"thread_{i:03d}" for i in range(len(ridge))]))
    return out


def _peg_pieces(kind: str) -> list:
    half = PEG_SIZE / 2.0
    n = {"square": 4, "round": 32, "hex": 6}[kind]
    return [g.prism_points(g.regular_polygon(n, half), -PEG_LENGTH / 2.0, PEG_LENGTH / 2.0)]


def _nut_pieces() -> tuple[list, list]:
    hexagon = g.regular_polygon(6, NUT_ACROSS_FLATS / 2.0)
    bore = g.regular_polygon(24, THREAD_R_MINOR + THREAD_DEPTH + THREAD_CLEARANCE)
    h = NUT_HEIGHT
    body = g.plate_with_hole(bore, None, -h / 2.0, h / 2.0, outer_poly=hexagon)
    ridge = g.thread_ridge_pieces(THREAD_R_MINOR + THREAD_DEPTH + THREAD_CLEARANCE + 0.0005,
                                  THREAD_R_MINOR + THREAD_CLEARANCE, THREAD_PITCH, -h / 2.0 + 0.002 + THREAD_PITCH / 2.0,
                                  NUT_THREAD_TURNS, THREAD_SEGMENT_DEG, z_min=-h / 2.0, z_max=h / 2.0)
    return body + ridge, [f"body_{i:03d}" for i in range(len(body))] + [f"thread_{i:03d}" for i in range(len(ridge))]


def parts(table_z: float) -> list[Part]:
    top = table_z + 0.002
    near, far = SIDE_ROWS_Y
    out = [Part(f"peg_{k}", _peg_pieces(k), [x, near, top + PEG_LENGTH / 2.0, 0, 0, 0, 1], COLOURS[f"peg_{k}"])
           for k, x in (("square", -0.15), ("round", -0.07), ("hex", 0.01))]
    place = {"gear_large": (-0.13, far), "gear_small": (0.20, near), "gear_medium": (0.32, near)}
    for name, teeth in zip(("gear_large", "gear_small", "gear_medium"), GEAR_TEETH):
        pieces = g.gear_pieces(GEAR_MODULE, teeth, GEAR_SHAFT_R + GEAR_BORE_CLEARANCE, -GEAR_THICKNESS / 2.0,
                               GEAR_THICKNESS / 2.0)
        out.append(Part(name, pieces, [*place[name], top + GEAR_THICKNESS / 2.0, 0, 0, 0, 1], COLOURS[name]))
    for name, r_out, x in zip(("ring_large", "ring_medium", "ring_small"), RING_OUTER, (0.0, 0.12, 0.24)):
        pieces = g.ring_pieces(RING_INNER, r_out, -RING_THICKNESS / 2.0, RING_THICKNESS / 2.0)
        out.append(Part(name, pieces, [x, far, top + RING_THICKNESS / 2.0, 0, 0, 0, 1], COLOURS[name]))
    nut_pieces, nut_names = _nut_pieces()
    out.append(Part("nut", nut_pieces, [0.10, near, top + NUT_HEIGHT / 2.0, *_quat_z(0.0)], COLOURS["nut"],
                    thread=True, piece_names=nut_names))
    return out


THREAD_CONTACTS = {"nut": "nut", "bolt": "bolt", "filtered": {"body_": ("core", "thread_"), "thread_": ("core",)}}


def side_table_boxes(table_z: float) -> list[dict]:
    cx, cy = sum(SIDE_TABLE_X) / 2.0, sum(SIDE_TABLE_Y) / 2.0
    lx, ly = SIDE_TABLE_X[1] - SIDE_TABLE_X[0], SIDE_TABLE_Y[1] - SIDE_TABLE_Y[0]
    boxes = [{"name": "side_table", "size": [lx, ly, SIDE_TABLE_THICKNESS],
              "center": [cx, cy, table_z - SIDE_TABLE_THICKNESS / 2.0], "colour": SIDE_TABLE_COLOUR}]
    h = table_z - SIDE_TABLE_THICKNESS
    inset = 0.03 + SIDE_TABLE_LEG / 2.0
    for name, sx, sy in (("FL", +1, +1), ("FR", +1, -1), ("BL", -1, +1), ("BR", -1, -1)):
        boxes.append({"name": f"side_table_leg_{name}", "size": [SIDE_TABLE_LEG, SIDE_TABLE_LEG, h],
                      "center": [cx + sx * (lx / 2.0 - inset), cy + sy * (ly / 2.0 - inset), h / 2.0],
                      "colour": SIDE_TABLE_COLOUR})
    return boxes


def visual_mesh(pieces) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, n = [], [], 0
    for p in pieces:
        v, f = g.hull_mesh(p)
        verts.append(v)
        faces.append(f + n)
        n += len(v)
    return np.vstack(verts), np.vstack(faces)
