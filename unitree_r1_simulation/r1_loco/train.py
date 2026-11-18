import os
import sys

import warp as wp

wp.config.enable_backward = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("R1_SIM_DIR", "/workspace/r1_sim"))

import r1_loco
from isaaclab_rl.entrypoints import run_train_cli

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--task" not in argv:
        argv = ["--task", r1_loco.TASK] + argv
    raise SystemExit(run_train_cli(["--rl_library", "rsl_rl"] + argv))
