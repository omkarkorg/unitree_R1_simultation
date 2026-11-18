#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
W=build/teleop_patch_work
P=docker/teleop/patches
XR_FILES=(teleop/teleop_hand_and_arm.py teleop/robot_control/robot_arm.py teleop/robot_control/robot_arm_ik.py
          teleop/robot_control/robot_hand_brainco.py)
TV_FILES=(src/televuer/televuer.py src/televuer/tv_wrapper.py src/televuer/lab_view.py)
IMAGE="${TELEOP_IMAGE:-unitree-r1/teleop:2.3}"

if [[ "${1:-}" == "extract" ]]; then
    [[ ! -e "${W}/new" ]] || { echo "${W}/new exists - remove it first (it may hold unsaved edits)" >&2; exit 1; }
    mkdir -p "${W}"
    docker run --rm -v "${PWD}/${W}:/out" --entrypoint bash "${IMAGE}" -c "
set -e
cd /opt/xr_teleoperate
for f in ${XR_FILES[*]}; do mkdir -p /out/new/xr/\$(dirname \$f) /out/orig/xr/\$(dirname \$f)
  cp \$f /out/new/xr/\$f; git show HEAD:\$f > /out/orig/xr/\$f; done
cd teleop/televuer
for f in ${TV_FILES[*]}; do mkdir -p /out/new/tv/\$(dirname \$f) /out/orig/tv/\$(dirname \$f)
  cp \$f /out/new/tv/\$f; git show HEAD:\$f > /out/orig/tv/\$f 2>/dev/null || rm -f /out/orig/tv/\$f; done
chown -R $(id -u):$(id -g) /out"
    echo "extracted into ${W}/{orig,new}"
    exit 0
fi

header() {
    awk '/^--- /{exit} {print}' "$1"
}

make_patch() {
    local out="$1" tree="$2"; shift 2
    local tmp
    tmp="$(mktemp)"
    header "${out}" >"${tmp}"
    for f in "$@"; do
        local a="${W}/orig/${tree}/${f}" b="${W}/new/${tree}/${f}"
        if [[ -f "${a}" ]]; then
            diff -u --label "a/${f}" --label "b/${f}" "${a}" "${b}" >>"${tmp}" || [[ $? -eq 1 ]]
        else
            diff -u --label /dev/null --label "b/${f}" /dev/null "${b}" >>"${tmp}" || [[ $? -eq 1 ]]
        fi
    done
    mv "${tmp}" "${out}"
    echo "wrote ${out} ($(wc -l <"${out}") lines)"
}

make_patch "${P}/r1_teleop.patch" xr "${XR_FILES[@]}"
make_patch "${P}/televuer_r1.patch" tv "${TV_FILES[@]}"
