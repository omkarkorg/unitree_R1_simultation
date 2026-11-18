#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
compose run --rm --name r1-mock-sim mock-sim "$@"
