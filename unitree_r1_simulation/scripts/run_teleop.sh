#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ ! -s certs/cert.pem || ! -s certs/key.pem || -z "${HOST_IP:-}" ]] \
        || ! host_has_ip "${HOST_IP}" || ! cert_covers_ip "${HOST_IP}"; then
    info "HOST_IP=${HOST_IP:-<unset>} / certificate do not match this PC's address - renewing them"
    ./scripts/gen_certs.sh
    HOST_IP="$(sed -n 's/^HOST_IP=//p' .env | tail -1)"
    export HOST_IP
fi
[[ -r certs/key.pem ]] || die "certs/key.pem is not readable by $(id -un) - re-run ./scripts/gen_certs.sh without sudo"

extra=()
for arg in "$@"; do
    if [[ "${arg}" == "--record" ]]; then export TELEOP_RECORD=1; else extra+=("${arg}"); fi
done
mkdir -p data
compose run --rm --name r1-teleop teleop "${extra[@]}"
