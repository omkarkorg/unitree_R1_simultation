#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
docker ps --filter ancestor=unitree-r1/sim:6.1.0 --format '{{.Names}}' | grep -q . \
    || die "Isaac Sim is not running - start ./scripts/run_sim.sh first (and wait for 'running')"
compose run --rm -T --entrypoint python teleop /opt/r1_sim/tests/pick_parts_test.py "$@"
