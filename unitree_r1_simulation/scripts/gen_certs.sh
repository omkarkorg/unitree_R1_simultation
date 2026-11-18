#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

[[ "${EUID}" -ne 0 ]] || die "run this as your normal user (not sudo): the teleop container reads the key as your user"
ip="${1:-$(detect_host_ip)}"
[[ -n "${ip}" ]] || die "could not detect the LAN IP; pass it: $0 <ip>"
[[ "${ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "'${ip}' is not an IPv4 address"

mkdir -p certs
openssl req -x509 -nodes -days 825 -newkey rsa:2048 \
    -keyout certs/key.pem -out certs/cert.pem \
    -subj "/CN=r1-teleop" \
    -addext "subjectAltName=IP:${ip},IP:127.0.0.1,DNS:localhost" >/dev/null 2>&1
chmod 644 certs/cert.pem
chmod 600 certs/key.pem

if grep -q '^HOST_IP=' .env; then
    sed -i "s/^HOST_IP=.*/HOST_IP=${ip}/" .env
else
    echo "HOST_IP=${ip}" >> .env
fi
sed -i "s/^HOST_UID=.*/HOST_UID=$(id -u)/; s/^HOST_GID=.*/HOST_GID=$(id -g)/" .env

info "certificate for ${ip} written to certs/ (valid 825 days)"
info "Quest 3 URL: https://${ip}:8012/?ws=wss://${ip}:8012"
