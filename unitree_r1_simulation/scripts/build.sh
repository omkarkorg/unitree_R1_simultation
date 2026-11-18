#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

services=("$@")
[[ ${#services[@]} -gt 0 ]] || services=(sim teleop ros2)
for s in "${services[@]}"; do
    case "${s}" in sim|teleop|ros2) ;; *) die "unknown image '${s}' (use sim, teleop, ros2)";; esac
    info "building ${s} ..."
    compose --profile teleop --profile ros2 build "${s}"
done
info "done: $(docker images --format '{{.Repository}}:{{.Tag}} ({{.Size}})' | grep unitree-r1 | tr '\n' ' ')"
