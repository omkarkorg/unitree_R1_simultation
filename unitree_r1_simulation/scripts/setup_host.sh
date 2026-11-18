#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "Please run with sudo:  sudo $0"
    exit 1
fi

echo "[setup] NVIDIA driver:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
    echo "[setup] nvidia-smi failed - install the NVIDIA driver first (>= 595.58.03 for Isaac Sim 6.1)"; exit 1; }
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
if [[ "$(printf '%s\n%s\n' "595.58.03" "${drv}" | sort -V | head -1)" != "595.58.03" ]]; then
    echo "[setup] WARNING: driver ${drv} is older than 595.58.03, the minimum for Isaac Sim 6.1 on Linux"
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo "[setup] installing NVIDIA Container Toolkit ..."
    apt-get update
    apt-get install -y --no-install-recommends curl gnupg ca-certificates
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
else
    echo "[setup] NVIDIA Container Toolkit already installed: $(nvidia-ctk --version | head -1)"
fi

echo "[setup] configuring the Docker runtime (restarts Docker: running containers are stopped) ..."
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "[setup] testing GPU access from a container ..."
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L

if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
    echo "[setup] ufw is active - allowing TCP 8012 (xr_teleoperate web server for the Quest 3)"
    ufw allow 8012/tcp comment 'unitree xr_teleoperate (Quest 3)'
fi

echo "[setup] done. Next (as your user, not sudo): ./scripts/check_host.sh"
