#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

use_isaac=0
[[ "${1:-}" == "--isaac" ]] && use_isaac=1
[[ -s certs/cert.pem && -s certs/key.pem ]] || die "no certificate yet - run ./scripts/gen_certs.sh"

teleop_log="$(mktemp)"
rec_dir="data/test_stack"
cleanup() { docker rm -f r1-test-mock r1-test-teleop >/dev/null 2>&1 || true; rm -f "${teleop_log}"; rm -rf "${rec_dir}"; }
trap cleanup EXIT
cleanup

sim_ctr=""
loco_failed=0
sim_counter() {
    local text
    if [[ ${use_isaac} -eq 1 ]]; then
        text="$(docker exec "${sim_ctr}" cat /tmp/r1_sim.status 2>/dev/null || true)"
    else
        text="$(docker logs r1-test-mock 2>&1 || true)"
    fi
    grep -o "[\"']$1[\"']: [0-9]*" <<<"${text}" | tail -1 | grep -o '[0-9]*$' || echo 0
}
settle() {
    if [[ ${use_isaac} -eq 1 ]]; then sleep 6; else sleep 3; fi
}

if [[ ${use_isaac} -eq 0 ]]; then
    info "[1/6] starting the mock simulator"
    compose run -d --rm --name r1-test-mock mock-sim --duration 600 >/dev/null
    sleep 4
else
    info "[1/6] using the running Isaac Sim simulator"
    sim_ctr="$(docker ps --filter ancestor=unitree-r1/sim:6.1.0 --format '{{.Names}}' | head -1)"
    [[ -n "${sim_ctr}" ]] || die "Isaac Sim is not running - start ./scripts/run_sim.sh first"
    for _ in $(seq 1 120); do
        docker exec "${sim_ctr}" test -s /tmp/r1_sim.status 2>/dev/null && break
        sleep 5
    done
    docker exec "${sim_ctr}" test -s /tmp/r1_sim.status || die "the simulator in '${sim_ctr}' is not running its scene yet"
    info "      simulator container: ${sim_ctr}"
fi

info "[2/6] protocol test: CRC, camera stream, R1_A5 arm controller, BrainCo hands"
compose run --rm -T --entrypoint python teleop /opt/r1_sim/tests/test_protocol.py

sim_status() { docker exec "${sim_ctr}" cat /tmp/r1_sim.status 2>/dev/null || true; }
if [[ ${use_isaac} -eq 1 ]]; then
    scene="$(sim_status | grep -o '"scene": "[a-z_]*"' | cut -d'"' -f4)"
    free="$(sim_status | grep -o '"free_standing": [a-z]*' | awk '{print $2}')"
    body="$(sim_status | grep -o '"body": "[a-z]*"' | cut -d'"' -f4)"
    info "      scene: ${scene:-?}, robot $([[ "${free}" == "true" ]] && echo "standing, body: ${body:-?} (exact = placed, balance = leg controller)" || echo "pelvis fixed")"
    if [[ "${free}" == "true" ]]; then
        info "[3/6] leg controller: walk, turn, sideways, squat, lean on rt/sim/body_cmd (r1_sim/tests/loco_test.py)"
        compose run --rm -T --entrypoint python teleop /opt/r1_sim/tests/loco_test.py \
            || { loco_failed=1; info "      leg controller test FAILED - continuing with the other tests"; }
    fi
    if [[ "${scene}" == "pick_place" && "${free}" != "true" ]]; then
        info "[3/6] pick test: R1_A5 IK + BrainCo grasp and lift the green cylinder (frames in ./data/grasp_test)"
        mkdir -p data/grasp_test
        compose run --rm -T --entrypoint python teleop /opt/r1_sim/tests/grasp_test.py --save-frames /data/grasp_test
    else
        info "      pick test: only with SIM_SCENE=pick_place SIM_LOCOMOTION=0 (the assembly kit: ./scripts/test_assembly.sh)"
    fi
else
    info "[3/6] leg controller / pick test: skipped (need Isaac Sim physics - run with --isaac)"
fi

info "[4/6] ROS 2 Humble: rt/lowstate -> /joint_states, camera -> image topics, arm_demo -> rt/lowcmd"
settle
before=$(sim_counter lowcmd); before_hand=$(sim_counter hand_cmd)
compose run --rm -T ros2 bash -c '
set -e
ros2 run r1_ros2 state_bridge >/tmp/state_bridge.log 2>&1 &
ros2 run r1_ros2 camera_bridge >/tmp/camera_bridge.log 2>&1 &
sleep 6
echo "--- /lowstate (tick):"; timeout 15 ros2 topic echo --once /lowstate --field tick
n=$(timeout 20 python3 - <<PY
import rclpy
from sensor_msgs.msg import JointState

rclpy.init()
node = rclpy.create_node("r1_test_joint_states")
got = []
node.create_subscription(JointState, "/joint_states", lambda m: got.append(len(m.name)), 10)
while not got:
    rclpy.spin_once(node, timeout_sec=0.5)
print(got[0])
PY
)
echo "--- /joint_states: ${n} joints"; test "${n}" -eq 48
echo "--- camera width:"; timeout 15 ros2 topic echo --once /r1/head_camera/image_raw --field width
echo "--- arm_demo for 8 s"; timeout 8 ros2 run r1_ros2 arm_demo || test $? -eq 124
'
settle
after=$(sim_counter lowcmd); crc=$(sim_counter lowcmd_crc_errors); after_hand=$(sim_counter hand_cmd)
info "      from ROS 2: rt/lowcmd $((after - before)) (CRC errors: ${crc}), rt/brainco/*/cmd $((after_hand - before_hand))"
[[ $((after - before)) -gt 100 && "${crc}" == "0" ]] || die "ROS 2 arm_demo commands were not accepted"
[[ $((after_hand - before_hand)) -gt 100 ]] || die "ROS 2 arm_demo hand commands did not arrive"

info "[5/6] xr_teleoperate end-to-end (teleop_hand_and_arm.py --ipc) with a simulated headset: lab view, VR buttons, hands -> arms + fingers, head, whole body"
mkdir -p "${rec_dir}"
compose run -d --rm --name r1-test-teleop -e COLUMNS=300 teleop --ipc \
    --record --task-dir /data/test_stack --task-name rec_test >/dev/null
docker logs -f r1-test-teleop >"${teleop_log}" 2>&1 &
for _ in $(seq 1 90); do
    docker logs r1-test-teleop 2>&1 | grep -q "Press \[r\]" && break
    sleep 1
done
docker logs r1-test-teleop 2>&1 | grep -q "Press \[r\]" || { docker logs r1-test-teleop 2>&1 | tail -40; die "teleop did not become ready"; }
settle
arm_count() { echo $(( $(sim_counter lowcmd) + $(sim_counter arm_sdk) )); }
before=$(arm_count); before_hand=$(sim_counter hand_cmd); before_body=$(sim_counter body_cmd)
immersive_view="${TELEOP_IMMERSIVE_VIEW:-1}" view="${TELEOP_VIEW:-lab}"
whole_body="${TELEOP_WHOLE_BODY:-1}"
[[ "${SIM_LOCOMOTION:-1}" == "1" ]] || whole_body=0
[[ "${TELEOP_HAND_MAPPING:-exact}" == "exact" ]] || whole_body=0
[[ "${TELEOP_DISPLAY_MODE:-immersive}" == "immersive" ]] || { immersive_view=0; view=camera; }
compose run --rm -T --entrypoint python teleop /opt/r1_sim/tests/teleop_hand_test.py \
    --start-delay "${TELEOP_START_DELAY:-2}" --ramp-time "${TELEOP_RAMP_TIME:-1}" \
    --speed-limit "${TELEOP_ARM_SPEED_LIMIT:-10}" --hand-mapping "${TELEOP_HAND_MAPPING:-exact}" \
    --view "${view}" --vr-buttons "${TELEOP_VR_BUTTONS:-1}" --record "${TELEOP_VR_BUTTONS:-1}" \
    --head-tracking "${TELEOP_HEAD_TRACKING:-1}" --immersive-view "${immersive_view}" \
    --whole-body "${whole_body}" --body "${SIM_BODY:-exact}" || die "teleop hand path test failed"
for _ in $(seq 1 30); do docker ps --format '{{.Names}}' | grep -q '^r1-test-teleop$' || break; sleep 1; done
! docker ps --format '{{.Names}}' | grep -q '^r1-test-teleop$' || { tail -20 "${teleop_log}"; die "teleop did not exit after EXIT"; }
settle
after=$(arm_count); after_hand=$(sim_counter hand_cmd); after_body=$(sim_counter body_cmd)
info "      samples taken by the simulator during the teleoperation test: arms (rt/lowcmd + rt/arm_sdk) $((after - before)), rt/brainco/*/cmd $((after_hand - before_hand)), rt/sim/body_cmd $((after_body - before_body))"
[[ $((after - before)) -gt 500 ]] || die "xr_teleoperate did not stream arm commands"
[[ $((after_hand - before_hand)) -gt 100 ]] || die "xr_teleoperate did not stream BrainCo hand commands"
[[ "${whole_body}" != "1" || $((after_body - before_body)) -gt 100 ]] || die "xr_teleoperate did not stream body commands (whole body)"
[[ "$(sim_counter lowcmd_crc_errors)" == "0" ]] || die "CRC errors on rt/lowcmd"
log_text="$(tr -s ' ' <"${teleop_log}")"
for expected in "headset in VR: the robot's forward" "controllers in use" "hand tracking: both hands tracked" \
                "tracking jump" "hands not tracked" "headset in VR | hands"; do
    grep -qF -- "${expected}" <<<"${log_text}" || { tail -40 "${teleop_log}"; die "teleop log lacks \"${expected}\""; }
done
[[ "$(grep -o "controllers in use - " <<<"${log_text}" | wc -l)" -eq 1 ]] \
    || { grep "controllers in use" "${teleop_log}"; die "teleop reported controllers in use while the hands were tracked"; }
! grep -q "Traceback" "${teleop_log}" || { grep -A12 "Traceback" "${teleop_log}"; die "teleop raised an exception"; }
info "      teleop log reports: VR heading, controllers in use, hands tracked / not tracked, tracking jump, status line"
if [[ "${TELEOP_VR_BUTTONS:-1}" == "1" ]]; then
    expected_lines=("START pressed in the headset" "REC pressed in the headset" "New episode created"
                    "RESTART pressed in the headset" "RESET pressed in the headset" "RESET: the robot and every object go back"
                    "GO: the robot follows you again" "EXIT pressed in the headset" "EXIT: the teleoperation ends")
    [[ "${whole_body}" != "1" ]] || expected_lines+=("the simulator put the robot back at its start")
    for expected in "${expected_lines[@]}"; do
        grep -qF -- "${expected}" <<<"${log_text}" || { tail -40 "${teleop_log}"; die "teleop log lacks \"${expected}\""; }
    done
    frames="$(python3 - "${rec_dir}/rec_test/episode_0000" <<'PY'
import json, os, sys
d = json.load(open(os.path.join(sys.argv[1], "data.json")))
colors = [f for f in os.listdir(os.path.join(sys.argv[1], "colors"))]
print(len(d["data"]), len(colors))
PY
)" || die "the episode recorded with the REC button was not saved (${rec_dir}/rec_test/episode_0000)"
    read -r n_items n_images <<<"${frames}"
    info "      REC button: episode saved with ${n_items} frames, ${n_images} head-camera images"
    [[ "${n_items}" -ge 30 && "${n_images}" -ge 30 ]] || die "the recorded episode is too short (${n_items} frames)"
fi

[[ "${loco_failed}" == "0" ]] || die "the leg controller test (step 3) failed - everything else passed"
info "[6/6] all tests passed"
