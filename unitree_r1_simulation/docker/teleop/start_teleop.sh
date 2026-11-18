#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${HOME}" "${MPLCONFIGDIR:-${HOME}/.config/matplotlib}"

if [[ ! -s "${XR_TELEOP_CERT}" || ! -s "${XR_TELEOP_KEY}" ]]; then
    echo "[start_teleop] TLS certificate missing (${XR_TELEOP_CERT}, ${XR_TELEOP_KEY})."
    echo "               Run ./scripts/gen_certs.sh on the host first."
    exit 1
fi

if [[ "${DDS_DOMAIN_ID:-1}" != "1" ]]; then
    echo "[start_teleop] xr_teleoperate --sim always uses DDS domain 1 (DDS_DOMAIN_ID=${DDS_DOMAIN_ID})."
    exit 1
fi

python /opt/teleop/wait_for_sim.py --host 127.0.0.1 --domain 1 --interface "${DDS_INTERFACE:-lo}"

args=(--arm=R1_A5 --ee=brainco --sim --headless
      --input-mode=hand
      --display-mode="${TELEOP_DISPLAY_MODE:-immersive}"
      --img-server-ip=127.0.0.1
      --network-interface="${DDS_INTERFACE:-lo}"
      --wait-for-hands="${TELEOP_WAIT_FOR_HANDS:-1}"
      --start-delay="${TELEOP_START_DELAY:-2}"
      --view="${TELEOP_VIEW:-lab}"
      --hand-mapping="${TELEOP_HAND_MAPPING:-exact}"
      --finger-mapping="${TELEOP_FINGER_MAPPING:-direct}"
      --vr-buttons="${TELEOP_VR_BUTTONS:-1}"
      --show-hands="${TELEOP_SHOW_HANDS:-1}"
      --ramp-time="${TELEOP_RAMP_TIME:-1}"
      --arm-speed-limit="${TELEOP_ARM_SPEED_LIMIT:-10}"
      --head-tracking="${TELEOP_HEAD_TRACKING:-1}"
      --head-speed-limit="${TELEOP_HEAD_SPEED_LIMIT:-3}"
      --immersive-view="${TELEOP_IMMERSIVE_VIEW:-1}"
      --whole-body="${TELEOP_WHOLE_BODY:-1}"
      --passthrough="${TELEOP_PASSTHROUGH:-1}"
      --world-scale="${TELEOP_WORLD_SCALE:-auto}")
if [[ "${TELEOP_RECORD:-0}" == "1" ]]; then
    args+=(--record --task-dir=/data --task-name="${TELEOP_TASK_NAME:-r1_pick_place}")
fi

ip="${HOST_IP:-<this-PC-LAN-IP>}"
cat <<EOF

=====================================================================================
 Meta Quest 3: open the Browser and go to
     https://${ip}:8012/?ws=wss://${ip}:8012
 Switch the Quest to hand tracking first: controllers lying still on a table (or tapped
 together twice), virtual hands visible. Accept the certificate warning (Advanced ->
 Proceed), press "$( [[ "${TELEOP_PASSTHROUGH:-1}" == "1" ]] && echo "Pass-through" || echo "Virtual Reality" )" and allow hand tracking.$( [[ "${TELEOP_PASSTHROUGH:-1}" == "1" ]] && printf '\n You then see your real room; the robot, the table and the side table are drawn into it.' )
 Whole body (TELEOP_WHOLE_BODY=1): stand where there is about 1.5 m free in front of you and on your right
 (the table goes in front of you, the parts' side table on your right) - the robot walks where you walk,
 turns where you turn, squats and leans with you, and stands still when you do.

 In the headset: press START (green button on your left) with a fingertip - or key r here.
   Once both hands are tracked a ${TELEOP_START_DELAY:-2} s countdown appears ("GET READY"): look at the table.
   At "GO" the table is placed in front of you, the robot's eyes where yours are and its hands where yours are.
   It holds (and stands still) whenever hand tracking is not OK (the headset says why).
   The parts are on the side table on your right; the peg board, gears, Tower of Hanoi and bolt on the table.
   REC (below START) = start/stop recording, like key s (only with TELEOP_RECORD=1 / --record).
   Above START: RESTART (key c) = get ready and GO again, which re-calibrates: the robot's eyes and hands
   are put on yours again (use it when the hands feel off); RESET (key x; press it twice) = the robot and
   every part back where they started; EXIT (key q; press it twice) = end the teleoperation.
 Keys in THIS terminal:  r = start   s = start/stop recording   c = restart   x = reset the scene
                         q = quit (arms return home first)
=====================================================================================

EOF

cd /opt/xr_teleoperate/teleop
exec python teleop_hand_and_arm.py "${args[@]}" "$@"
