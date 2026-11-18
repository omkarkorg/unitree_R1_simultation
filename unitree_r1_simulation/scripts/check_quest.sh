#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "$0")/common.sh"

fails=0
ok()   { printf '  \033[1;32mOK\033[0m    %s\n' "$*"; }
fail() { printf '  \033[1;31mFAIL\033[0m  %s\n' "$*"; fails=$((fails + 1)); }
note() { printf '  \033[1;33mNOTE\033[0m  %s\n' "$*"; }

quest_ip="${1:-}"
[[ -z "${quest_ip}" || "${quest_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || die "give the headset's IP address as numbers, e.g. $0 192.168.1.57 (Quest: Settings -> Wi-Fi -> the connected network -> details)"

echo "Quest 3 connection checks"

ip="$(detect_host_ip)"
[[ -n "${ip}" ]] || { fail "this PC has no network address"; exit 1; }
dev="$(ip -4 -o addr show | awk -v a="${ip}" '{split($4, s, "/"); if (s[1] == a) {print $2; exit}}')"
net="$(nmcli -g GENERAL.CONNECTION device show "${dev}" 2>/dev/null || true)"
ok "this PC is ${ip} on ${dev}${net:+ (network \"${net}\")} - the headset must join the same network"
if [[ "${net,,}" =~ (guest|gæst|gäst|gast|gjest|vieras|visitor|besucher|invit) ]]; then
    note "\"${net}\" looks like a guest network: those often block traffic between devices (client isolation)"
fi

url_ip="${HOST_IP:-}"
if [[ -n "${url_ip}" ]] && host_has_ip "${url_ip}"; then
    if [[ "${url_ip}" == "${ip}" ]]; then ok ".env HOST_IP=${url_ip}"
    else note ".env HOST_IP=${url_ip} is another address of this PC (its default route uses ${ip})"; fi
    if cert_covers_ip "${url_ip}"; then ok "certificate covers ${url_ip}"
    else fail "certificate does not cover ${url_ip} -> restart ./scripts/run_teleop.sh (it renews it)"; fi
else
    fail ".env HOST_IP=${url_ip:-<unset>} is not this PC's address (now ${ip}) -> restart ./scripts/run_teleop.sh (it renews it)"
    url_ip="${ip}"
fi

if ss -ltn 2>/dev/null | grep -qE ':8012\b'; then
    if command -v curl >/dev/null 2>&1; then
        code="$(curl -sk -o /dev/null -w '%{http_code}' --max-time 5 "https://${url_ip}:8012/" 2>/dev/null || true)"
        if [[ "${code}" == "200" ]]; then ok "teleop web server answers at https://${url_ip}:8012"
        else fail "teleop web server on port 8012 does not answer (HTTP ${code:-none})"; fi
    fi
    served="$(openssl s_client -connect "${url_ip}:8012" </dev/null 2>/dev/null \
        | openssl x509 -noout -ext subjectAltName 2>/dev/null | tr ',' '\n' | sed -n 's/^ *IP Address://p' | head -1)"
    if [[ -n "${served}" && "${served}" != "${url_ip}" ]]; then
        fail "the running teleop was started for the old address ${served} - press q in its terminal and run ./scripts/run_teleop.sh again"
    fi
    img="$(sed -n '/^  teleop:/,/^  [a-z]/s/^    image: *//p' docker-compose.yml | head -1)"
    if docker image inspect "${img}" >/dev/null 2>&1; then
        if docker run --rm --network bridge --entrypoint python "${img}" \
                -c "import socket; socket.create_connection(('${url_ip}', 8012), timeout=3)" >/dev/null 2>&1; then
            ok "firewall lets other devices reach TCP 8012"
        else
            fail "TCP 8012 looks blocked by this PC's firewall -> sudo ufw allow 8012/tcp"
        fi
    fi
else
    note "nothing listens on port 8012 - start ./scripts/run_teleop.sh (after run_sim.sh), then run this again"
fi

page_open() {
    ss -Htn state established '( sport = :8012 )' 2>/dev/null | awk -v q="$1" \
        '{p = $NF; sub(/:[0-9]+$/, "", p); gsub(/[][]/, "", p); sub(/^::ffff:/, "", p); if (p == q) f = 1} END {exit !f}'
}
if [[ -n "${quest_ip}" ]]; then
    if ip -4 route get "${quest_ip}" 2>/dev/null | grep -q ' via '; then
        fail "${quest_ip} is not on this PC's network (${dev}) - connect the headset to the same Wi-Fi"
    else
        echo "  ...   waiting up to 60 s for ${quest_ip} - put the headset on now and keep it awake"
        answered=0; page=0; end=$((SECONDS + 60))
        while (( SECONDS < end )); do
            if page_open "${quest_ip}"; then page=1; answered=1; break; fi
            if ping -c 1 -W 2 "${quest_ip}" >/dev/null 2>&1; then answered=1; break; fi
            sleep 1
        done
        mac="$(ip neigh show "${quest_ip}" 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "lladdr") print $(i + 1)}')"
        vendor=""
        if [[ -n "${mac}" ]]; then
            vendor="$(systemd-hwdb query "OUI:$(tr -d ':' <<<"${mac:0:8}" | tr 'a-f' 'A-F')" 2>/dev/null \
                | sed -n 's/^ID_OUI_FROM_DATABASE=//p')"
        fi
        if [[ -n "${vendor}" && ! "${vendor}" =~ (Meta|Oculus|Facebook) ]]; then
            fail "the device at ${quest_ip} is made by ${vendor} - not a Quest; read the IP again on the headset (Settings -> Wi-Fi)"
        elif (( page )); then
            ok "the headset at ${quest_ip} has the teleop page open"
        elif (( answered )); then
            ok "headset ${quest_ip} answers - the network lets the two devices talk to each other"
        elif [[ -n "${mac}" ]]; then
            note "headset ${quest_ip} is on the network but does not answer ping (the path itself looks fine)"
        else
            fail "headset ${quest_ip} did not answer for 60 s. If it was awake and the IP is right, the Wi-Fi blocks traffic between devices (client isolation) - use a hotspot (README section 6)"
        fi
    fi
else
    note "to test the Wi-Fi itself, add the headset's IP (Quest: Settings -> Wi-Fi -> the connected network -> details), e.g. $0 192.168.1.57"
fi

echo
info "Quest 3 URL: https://${url_ip}:8012/?ws=wss://${url_ip}:8012"
if [[ ${fails} -eq 0 ]]; then info "no problem found"; else warn "${fails} check(s) failed - see README section 6"; fi
exit $(( fails > 0 ))
