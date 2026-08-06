#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-real_world/configs/real_wm_infer.yaml}"
INPUT_VIDEO="${INPUT_VIDEO:-/path/to/real_trial.mp4}"
ACTIONS="${ACTIONS:-/path/to/real_trial_actions.npy}"
TASK="${TASK:-Put the red block into the box.}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/real_world_wm/demo_trial}"
DEVICE="${DEVICE:-cuda:0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"

EFFECTIVE_CONFIG="${CONFIG}"
CLI_SUPPORTS_NUM_INFERENCE_STEPS=0
if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  if python -m real_world.world_model.cli_predict --help 2>&1 | grep -q -- "--num-inference-steps"; then
    CLI_SUPPORTS_NUM_INFERENCE_STEPS=1
  else
    EFFECTIVE_CONFIG="${OUTPUT_DIR}/real_wm_infer_num_steps_${NUM_INFERENCE_STEPS}.yaml"
    python scripts/override_real_world_wm_config.py \
      --config "${CONFIG}" \
      --output "${EFFECTIVE_CONFIG}" \
      --num-inference-steps "${NUM_INFERENCE_STEPS}"
    echo "[real-world-wm] cli_predict does not expose --num-inference-steps; using config override ${EFFECTIVE_CONFIG}"
  fi
fi

predict_args=(
  --config "${EFFECTIVE_CONFIG}"
  --input-video "${INPUT_VIDEO}"
  --actions "${ACTIONS}"
  --task "${TASK}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
)

if [[ -n "${NUM_INFERENCE_STEPS}" && "${CLI_SUPPORTS_NUM_INFERENCE_STEPS}" == "1" ]]; then
  predict_args+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi

python -m real_world.world_model.cli_predict "${predict_args[@]}"
