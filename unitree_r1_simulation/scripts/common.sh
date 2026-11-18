ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT_DIR
cd "${ROOT_DIR}" || exit 1

if [[ -f "${ROOT_DIR}/.env" ]]; then
    _caller_env="$(export -p | grep -v '^declare -[a-zA-Z]*r')"
    set -a
    source "${ROOT_DIR}/.env"
    set +a
    eval "${_caller_env}"
    unset _caller_env
fi

info()  { printf '\033[1;34m[r1]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[r1] WARNING:\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31m[r1] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

compose() { docker compose -f "${ROOT_DIR}/docker-compose.yml" "$@"; }

detect_host_ip() {
    local addr
    addr="$(ip -4 route get 1.1.1.1 2>/dev/null \
        | awk '{for (i = 1; i < NF; i++) if ($i == "src") {print $(i + 1); exit}}' || true)"
    if [[ -z "${addr}" ]]; then
        addr="$(ip -4 -o addr show scope global up 2>/dev/null \
            | awk '$2 !~ /^(docker|br-|veth|virbr|lxdbr|tailscale|wg|tun|zt)/ {split($4, a, "/"); print a[1]; exit}' || true)"
    fi
    printf '%s\n' "${addr}"
}

host_has_ip() {
    ip -4 -o addr show 2>/dev/null | awk '{split($4, a, "/"); print a[1]}' | grep -qxF -- "$1"
}

cert_covers_ip() {
    openssl x509 -in "${ROOT_DIR}/certs/cert.pem" -noout -ext subjectAltName 2>/dev/null \
        | tr ',' '\n' | sed 's/^ *//' | grep -qxF -- "IP Address:$1"
}

allow_x11() {
    if [[ -z "${DISPLAY:-}" ]]; then
        warn "DISPLAY is not set - GUI windows (Isaac Sim, RViz) cannot open."
        return 0
    fi
    if command -v xhost >/dev/null 2>&1; then
        xhost +local: >/dev/null 2>&1 || warn "xhost +local: failed - GUI windows may not open"
    else
        warn "xhost not found (sudo apt install x11-xserver-utils) - GUI windows may not open"
    fi
}

require_nvidia_runtime() {
    if ! docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
        die "Docker has no NVIDIA runtime. Run: sudo ./scripts/setup_host.sh"
    fi
}
