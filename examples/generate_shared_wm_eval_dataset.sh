#!/usr/bin/env bash
# Mini/full monitoring sets may share trial prefixes; they are not independent.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the source policy checkpoint}"
: "${SHARED_WM_EVAL:?Set SHARED_WM_EVAL to a new dataset root}"
: "${EXPERIMENT:?Set a unique EXPERIMENT name}"
for split in mini full; do
  trials=2
  [[ "$split" != full ]] || trials=6
  python -m merl.launch --mode MFRL --job collect \
    --sft-checkpoint "$SFT_CHECKPOINT" --shared-wm-eval "$SHARED_WM_EVAL" \
    --experiment "${EXPERIMENT}_${split}" --split "wm_eval_fixed_${split}" \
    --actor-gpus "${ACTOR_GPUS:-1}" --trials "$trials" "$@"
done
