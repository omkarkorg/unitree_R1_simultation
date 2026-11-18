from __future__ import annotations

from . import loco_spec

LAB_STATE_PORT = 55570
LAB_STATE_HZ = 120.0

TABLE_TOP_Z = 0.58
TABLE_SIZE = (0.60, 1.00)
TABLE_FRONT_X = 0.20
TABLE_TOP_THICKNESS = 0.04
TABLE_LEG = 0.05
TABLE_COLOUR = (0.56, 0.41, 0.27)
TABLE_LEG_COLOUR = (0.25, 0.17, 0.10)
TRAY_SIZE = (0.12, 0.08, 0.01)
TRAY_CENTER_XY = (0.36, 0.0)
TRAY_COLOUR = (0.15, 0.35, 0.80)
CUBE_SIZE = 0.05
CYL_RADIUS, CYL_HEIGHT = 0.025, 0.12
OBJECT_Y = 0.085
OBJECT_X = 0.32
FLOOR_SIZE = (8.0, 8.0, 0.10)
FLOOR_COLOUR = (0.35, 0.36, 0.38)

ROOM_X0, ROOM_X1, ROOM_Y0, ROOM_Y1, ROOM_HEIGHT = -1.2, 1.8, -1.5, 1.5, 2.8
ROOM_WALL = 0.10
ROOM_WALL_COLOUR = (0.80, 0.81, 0.82)
ROOM_FLOOR_COLOUR = (0.56, 0.57, 0.58)
ROOM_CEILING_COLOUR = (0.88, 0.88, 0.88)

OBJECTS = (
    {"name": "red_cube", "shape": "box", "size": [CUBE_SIZE, CUBE_SIZE, CUBE_SIZE], "colour": (0.85, 0.10, 0.10)},
    {"name": "green_cylinder", "shape": "cylinder", "radius": CYL_RADIUS, "height": CYL_HEIGHT,
     "colour": (0.10, 0.65, 0.20)},
)
OBJECT_NAMES = tuple(o["name"] for o in OBJECTS)
SCENES = ("assembly", "pick_place")


def table_x() -> float:
    return TABLE_FRONT_X + TABLE_SIZE[0] / 2.0


def object_start_positions(table_z: float = TABLE_TOP_Z) -> dict:
    return {"red_cube": (OBJECT_X, OBJECT_Y, table_z + CUBE_SIZE / 2.0 + 0.002),
            "green_cylinder": (OBJECT_X, -OBJECT_Y, table_z + CYL_HEIGHT / 2.0 + 0.002)}


def table_boxes(table_z: float = TABLE_TOP_Z, tray: bool = True) -> list[dict]:
    tx = table_x()
    boxes = [{"name": "table", "size": [TABLE_SIZE[0], TABLE_SIZE[1], TABLE_TOP_THICKNESS],
              "center": [tx, 0.0, table_z - TABLE_TOP_THICKNESS / 2.0], "colour": TABLE_COLOUR}]
    inset = 0.04 + TABLE_LEG / 2.0
    height = table_z - TABLE_TOP_THICKNESS
    for name, sx, sy in (("FL", +1, +1), ("FR", +1, -1), ("BL", -1, +1), ("BR", -1, -1)):
        boxes.append({"name": f"table_leg_{name}", "size": [TABLE_LEG, TABLE_LEG, height],
                      "center": [tx + sx * (TABLE_SIZE[0] / 2.0 - inset), sy * (TABLE_SIZE[1] / 2.0 - inset), height / 2.0],
                      "colour": TABLE_LEG_COLOUR})
    if tray:
        boxes.append({"name": "tray", "size": list(TRAY_SIZE),
                      "center": [TRAY_CENTER_XY[0], TRAY_CENTER_XY[1], table_z + TRAY_SIZE[2] / 2.0],
                      "colour": TRAY_COLOUR})
    return boxes


def floor_box() -> dict:
    return {"name": "floor", "size": list(FLOOR_SIZE), "center": [0.0, 0.0, -FLOOR_SIZE[2] / 2.0], "colour": FLOOR_COLOUR}


def room_boxes(with_ceiling: bool = False) -> list[dict]:
    cx, cy, z = (ROOM_X0 + ROOM_X1) / 2.0, (ROOM_Y0 + ROOM_Y1) / 2.0, ROOM_HEIGHT / 2.0
    lx, ly, w = ROOM_X1 - ROOM_X0 + 2 * ROOM_WALL, ROOM_Y1 - ROOM_Y0 + 2 * ROOM_WALL, ROOM_WALL
    boxes = [
        {"name": "room_floor", "size": [ROOM_X1 - ROOM_X0, ROOM_Y1 - ROOM_Y0, 0.004], "center": [cx, cy, 0.002],
         "colour": ROOM_FLOOR_COLOUR},
        {"name": "wall_front", "size": [w, ly, ROOM_HEIGHT], "center": [ROOM_X1 + w / 2, cy, z], "colour": ROOM_WALL_COLOUR},
        {"name": "wall_back", "size": [w, ly, ROOM_HEIGHT], "center": [ROOM_X0 - w / 2, cy, z], "colour": ROOM_WALL_COLOUR},
        {"name": "wall_left", "size": [lx, w, ROOM_HEIGHT], "center": [cx, ROOM_Y1 + w / 2, z], "colour": ROOM_WALL_COLOUR},
        {"name": "wall_right", "size": [lx, w, ROOM_HEIGHT], "center": [cx, ROOM_Y0 - w / 2, z], "colour": ROOM_WALL_COLOUR},
    ]
    if with_ceiling:
        boxes.append({"name": "ceiling", "size": [lx, ly, w], "center": [cx, cy, ROOM_HEIGHT + w / 2],
                      "colour": ROOM_CEILING_COLOUR})
    return boxes


def lab_joint_names() -> list[str]:
    from .joint_map import HAND_SIDES, body_joint_names, hand_joint_names
    return body_joint_names() + [n for side in HAND_SIDES for n in hand_joint_names(side)]


def scene_boxes(scene: str, table_z: float) -> list[dict]:
    if scene not in SCENES:
        raise ValueError(f"scene must be one of {SCENES}, got {scene!r}")
    if scene == "assembly":
        from .assembly import side_table_boxes
        return table_boxes(table_z, tray=False) + side_table_boxes(table_z)
    return table_boxes(table_z)


def scene_objects(scene: str, table_z: float) -> list[dict]:
    if scene not in SCENES:
        raise ValueError(f"scene must be one of {SCENES}, got {scene!r}")
    if scene == "assembly":
        from .assembly import parts
        return [{"name": p.name, "shape": "convex", "pieces": p.pieces, "colour": p.colour, "start": list(p.start),
                 "mass": p.mass, "thread": p.thread, "piece_names": p.piece_names} for p in parts(table_z)]
    start = object_start_positions(table_z)
    return [{**o, "start": [*start[o["name"]], 0.0, 0.0, 0.0, 1.0]} for o in OBJECTS]


def scene_fixtures(scene: str, table_z: float) -> list:
    if scene == "assembly":
        from .assembly import fixtures
        return fixtures(table_z)
    return []


def _mesh_json(pieces) -> tuple[list, list]:
    from .assembly import visual_mesh
    v, f = visual_mesh(pieces)
    return [[round(float(c), 5) for c in p] for p in v], [[int(i) for i in t] for t in f]


def lab_description(table_z: float, pelvis_height: float, room: bool, scene: str = "pick_place",
                    floating: bool = False, body: str | None = None) -> dict:
    objects = []
    for o in scene_objects(scene, table_z):
        o = {k: v for k, v in o.items() if k not in ("start", "mass", "thread", "piece_names")}
        if o["shape"] == "convex":
            v, f = _mesh_json(o.pop("pieces"))
            o.update(shape="mesh", vertices=v, faces=f)
        objects.append(o)
    mode = body or ("balance" if floating else "fixed")
    meshes = []
    for fx in scene_fixtures(scene, table_z):
        v, f = _mesh_json(fx.pieces)
        meshes.append({"name": fx.name, "vertices": v, "faces": f, "colour": fx.colour})
    return {
        "state_port": LAB_STATE_PORT,
        "state_hz": LAB_STATE_HZ,
        "frame": "x forward, y left, z up (m); poses x y z qx qy qz qw",
        "scene": scene,
        "robot_root": [0.0, 0.0, float(pelvis_height), 0.0, 0.0, 0.0, 1.0],
        "floating_base": bool(floating),
        "body": {"mode": mode, "h_nom": loco_spec.H_NOM, "pitch_max": loco_spec.PITCH_MAX,
                 "h_min": loco_spec.USE_H_MIN if mode == "balance" else loco_spec.H_MIN},
        "joint_names": lab_joint_names(),
        "boxes": scene_boxes(scene, table_z) + [floor_box()] + (room_boxes(with_ceiling=True) if room else []),
        "meshes": meshes,
        "objects": objects,
    }
