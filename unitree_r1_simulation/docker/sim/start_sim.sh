#!/usr/bin/env bash
set -euo pipefail

args=()
if [[ "${SIM_HEADLESS:-0}" == "1" ]]; then
    if [[ "${SIM_LIVESTREAM:-0}" != "0" ]]; then
        args+=(--livestream "${SIM_LIVESTREAM}")
    fi
elif [[ -z "${DISPLAY:-}" ]]; then
    echo "[start_sim] DISPLAY is not set - starting headless (set DISPLAY and run 'xhost +local:' on the host for the GUI)"
else
    args+=(--viz kit)
fi

exec "${ISAACLAB_PATH:-/workspace/isaaclab}/isaaclab.sh" -p /workspace/r1_sim/run_sim.py "${args[@]}" ${SIM_EXTRA_ARGS:-} "$@"
