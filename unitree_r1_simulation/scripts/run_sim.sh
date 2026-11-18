#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

require_nvidia_runtime
[[ "${SIM_HEADLESS:-0}" == "1" ]] || allow_x11
info "starting Isaac Sim (first start: URDF->USD conversion + shader compilation, several minutes)"
compose run --rm --name r1-sim sim "$@"
