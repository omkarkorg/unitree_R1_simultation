from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd_file
from isaaclab.sim.utils import clone

from r1_sim.robot_usd import convert_robot_urdf


@clone
def spawn_r1(prim_path, cfg, translation=None, orientation=None, **kwargs):
    usd = convert_robot_urdf(cfg.urdf_path, cfg.cache_dir, cfg.variant, log=lambda m: print(f"[r1_loco] {m}"))
    return spawn_from_usd_file(prim_path, usd, cfg, translation, orientation)
