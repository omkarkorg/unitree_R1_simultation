#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "$0")/common.sh"

fails=0
ok()   { printf '  \033[1;32mOK\033[0m    %s\n' "$*"; }
fail() { printf '  \033[1;31mFAIL\033[0m  %s\n' "$*"; fails=$((fails + 1)); }
note() { printf '  \033[1;33mNOTE\033[0m  %s\n' "$*"; }

echo "Host checks"
if docker info >/dev/null 2>&1; then ok "docker $(docker version --format '{{.Server.Version}}')"; else fail "docker not usable by $(id -un) (add yourself to the docker group)"; fi
if docker compose version >/dev/null 2>&1; then ok "docker compose $(docker compose version --short)"; else fail "docker compose plugin missing"; fi

if command -v nvidia-smi >/dev/null 2>&1; then
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    gpu="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
    if [[ "$(printf '%s\n%s\n' "595.58.03" "${drv}" | sort -V | head -1)" == "595.58.03" ]]; then
        ok "NVIDIA driver ${drv} (${gpu})"
    else
        fail "NVIDIA driver ${drv} < 595.58.03 (minimum for Isaac Sim 6.1 on Linux)"
    fi
    vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
    [[ "${vram:-0}" -ge 15000 ]] || note "GPU has ${vram} MiB VRAM (NVIDIA lists 16 GB as minimum; this scene runs on 12 GB - keep SIM_STEREO=0 if VRAM runs out)"
else
    fail "nvidia-smi not found (NVIDIA driver missing)"
fi

if docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then ok "Docker NVIDIA runtime"; else fail "Docker NVIDIA runtime missing -> sudo ./scripts/setup_host.sh"; fi

mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
[[ ${mem_gb} -ge 30 ]] && ok "RAM ${mem_gb} GB" || note "RAM ${mem_gb} GB (Isaac Sim lists 32 GB as minimum)"
docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
free_gb=$(df -BG --output=avail "${docker_root}" 2>/dev/null | tail -1 | tr -dc '0-9')
if docker image inspect unitree-r1/sim:6.1.0 >/dev/null 2>&1; then
    ok "disk ${free_gb:-?} GB free for Docker"
else
    [[ -n "${free_gb}" && ${free_gb} -ge 80 ]] && ok "disk ${free_gb} GB free for Docker" \
        || note "disk ${free_gb:-?} GB free (the first build needs ~80 GB: images ~55 GB + build cache)"
fi

if [[ -n "${DISPLAY:-}" ]]; then ok "DISPLAY=${DISPLAY} (session: ${XDG_SESSION_TYPE:-?})"; else note "DISPLAY not set - use SIM_HEADLESS=1"; fi
command -v xhost >/dev/null 2>&1 && ok "xhost available" || note "xhost missing (sudo apt install x11-xserver-utils)"

ip="$(detect_host_ip)"
if [[ -n "${ip}" ]]; then ok "LAN IP ${ip} (Quest 3 must be on the same network)"; else fail "no LAN IP"; fi
if [[ -s certs/cert.pem && -s certs/key.pem && "${HOST_IP:-}" == "${ip}" ]] && cert_covers_ip "${ip}"; then
    ok "TLS certificate and .env HOST_IP match ${ip}"
else
    note "certificate / .env HOST_IP (${HOST_IP:-unset}) do not match ${ip} - run_teleop.sh renews them when it starts"
fi

if command -v ufw >/dev/null 2>&1 && systemctl is-active --quiet ufw; then
    note "ufw is active - 'sudo ufw status' must list 8012/tcp ALLOW (setup_host.sh adds it; ./scripts/check_quest.sh tests it)"
fi
if ss -ltn 2>/dev/null | grep -qE ':(8012|55555|60000)\b'; then
    note "a port of 8012/55555/60000 is already in use (old container still running?)"
fi

for img in unitree-r1/sim:6.1.0 unitree-r1/teleop:2.3 unitree-r1/ros2:humble; do
    if docker image inspect "${img}" >/dev/null 2>&1; then ok "image ${img}"; else fail "image ${img} not built -> ./scripts/build.sh"; fi
done
if [[ "${SIM_LOCOMOTION:-1}" == "1" && "${SIM_BODY:-exact}" == "balance" ]]; then
    if [[ -s r1_sim/policies/r1_wholebody.pt ]]; then ok "leg controller r1_sim/policies/r1_wholebody.pt"
    else fail "no leg controller r1_sim/policies/r1_wholebody.pt -> ./scripts/train_loco.sh, then --export (or SIM_BODY=exact)"; fi
fi

echo
if [[ ${fails} -eq 0 ]]; then info "all checks passed"; else warn "${fails} check(s) failed"; fi
exit $(( fails > 0 ))
