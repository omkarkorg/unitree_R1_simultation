#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_nvidia_runtime

KIT_ARGS="--kit_args=--/log/channels/omni.physx.tensors.plugin=error"
RUNS=logs/rsl_rl/r1_wholebody
mkdir -p "${RUNS}"

newest_run() { ls -d "${RUNS}"/*/ 2>/dev/null | sort | tail -1 | sed 's#/$##'; }
newest_model() {
    if [[ -n "${2:-}" ]]; then echo "$1/model_$2.pt"; return; fi
    ls "$1"/model_*.pt 2>/dev/null | sed 's#.*model_\([0-9]*\)\.pt#\1 &#' | sort -n | tail -1 | cut -d' ' -f2
}

case "${1:-}" in
    --resume)
        run="${2:-$(newest_run)}"; model="$(newest_model "${run}")"
        [[ -f "${model}" ]] || die "no checkpoint in ${run}"
        iteration="$(basename "${model}" .pt | cut -d_ -f2)"
        info "resuming ${model} (iteration ${iteration}) up to iteration 4000"
        compose run --rm --name r1-train train --num_envs 4096 --max_iterations $((4000 - iteration)) \
            --run_name "$(basename "${run}" | cut -d_ -f3-)_resumed" --checkpoint "/opt/r1_train/${model}" "${KIT_ARGS}"
        ;;
    --export)
        run="${2:-$(newest_run)}"; model="$(newest_model "${run}" "${3:-}")"
        [[ -f "${model}" ]] || die "no checkpoint ${model}"
        info "exporting ${model} -> r1_sim/policies/r1_wholebody.pt"
        docker run --rm -u "$(id -u):$(id -g)" -v "${ROOT_DIR}/logs:/opt/r1_train/logs:ro" \
            -v "${ROOT_DIR}/r1_sim:/workspace/r1_sim" -v "${ROOT_DIR}/r1_loco:/opt/r1_train/r1_loco:ro" \
            --entrypoint /opt/isaaclab-venv/bin/python unitree-r1/sim:6.1.0 \
            /opt/r1_train/r1_loco/export.py "/opt/r1_train/${model}" --out /workspace/r1_sim/policies/r1_wholebody.pt
        ;;
    --evaluate)
        policy="${2:-r1_sim/policies/r1_wholebody.pt}"
        [[ -f "${policy}" ]] || die "no ${policy} (export one first: ./scripts/train_loco.sh --export)"
        compose run --rm -T --name r1-eval --entrypoint /workspace/isaaclab/isaaclab.sh train \
            -p /opt/r1_train/r1_loco/evaluate.py --policy "/workspace/${policy}" --num_envs 32 "${KIT_ARGS}"
        ;;
    "")
        info "training the whole-body leg controller (Isaac Lab, 4096 robots; Ctrl+C stops, --resume continues)"
        compose run --rm --name r1-train train --num_envs 4096 --max_iterations 4000 --run_name wholebody "${KIT_ARGS}"
        ;;
    *)
        die "usage: $0 [--resume <run dir> | --export [<run dir> [<iteration>]] | --evaluate [<policy .pt>]]"
        ;;
esac
