#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_nvidia_runtime
compose run --rm -T --name r1-kit-test --entrypoint bash sim -c \
    "/workspace/isaaclab/isaaclab.sh -p /workspace/r1_sim/tests/assembly_physics_test.py --kit_args=--/log/channels/omni.physx.tensors.plugin=error $*" \
    2>&1 | grep -E "^(assembly|pegs|gears|rings|nut|nut/bolt)|  PASS|  FAIL|ASSEMBLY|Traceback|Error:" 
exit "${PIPESTATUS[0]}"
