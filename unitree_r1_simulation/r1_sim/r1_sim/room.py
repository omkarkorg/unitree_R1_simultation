from __future__ import annotations

import isaaclab.sim as sim_utils

from .layout import room_boxes

ROOT = "/World/Room"


def spawn_room() -> None:
    for b in room_boxes(with_ceiling=False):
        cfg = sim_utils.CuboidCfg(size=tuple(b["size"]),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(b["colour"]), roughness=0.9))
        cfg.func(f"{ROOT}/{b['name']}", cfg, translation=tuple(b["center"]), orientation=(0.0, 0.0, 0.0, 1.0))
