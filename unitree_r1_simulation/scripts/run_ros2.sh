#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

allow_x11
if [[ $# -eq 0 ]]; then
    compose run --rm ros2
else
    compose run --rm ros2 "$@"
fi
