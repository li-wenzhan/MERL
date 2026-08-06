#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

REAL_DATA_ROOT="${REAL_DATA_ROOT:-/path/to/real_hdf5_or_folder}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/real_world_wm_finetune/red_block_box}"
SVD_MODEL_PATH="${SVD_MODEL_PATH:-/path/to/models/stable-video-diffusion-img2vid}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/path/to/models/clip-vit-base-patch32}"
CTRL_WORLD_CKPT="${CTRL_WORLD_CKPT:-/path/to/Ctrl_World/checkpoint.pt}"
RESUME_CKPT="${RESUME_CKPT:-}"
TASK="${TASK:-Put the red block into the box.}"

IMAGE_KEY="${IMAGE_KEY:-observations/images/cam_high}"
ACTION_KEY="${ACTION_KEY:-action}"
ACTION_SLICE="${ACTION_SLICE:-right}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
ACTION_STRIDE="${ACTION_STRIDE:-${FRAME_STRIDE}}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-1}"
FPS="${FPS:-30}"
NUM_HISTORY="${NUM_HISTORY:-8}"
NUM_FUTURE="${NUM_FUTURE:-8}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-2}"
SAVE_FINAL="${SAVE_FINAL:-0}"
VALIDATION_STEPS="${VALIDATION_STEPS:-500}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
SELF_FORCING_WEIGHT="${SELF_FORCING_WEIGHT:-1.0}"
REWARD_LOSS_WEIGHT="${REWARD_LOSS_WEIGHT:-1.0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
SEED="${SEED:-1024}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29541}"
WANDB_MODE="${WANDB_MODE:-offline}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES WANDB_MODE

echo "==== Real-world Ctrl-World HDF5 fine-tune ===="
echo "REAL_DATA_ROOT=${REAL_DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SVD_MODEL_PATH=${SVD_MODEL_PATH}"
echo "CLIP_MODEL_PATH=${CLIP_MODEL_PATH}"
echo "CTRL_WORLD_CKPT=${CTRL_WORLD_CKPT}"
echo "RESUME_CKPT=${RESUME_CKPT}"
echo "TASK=${TASK}"
echo "IMAGE_KEY=${IMAGE_KEY}"
echo "ACTION_KEY=${ACTION_KEY}"
echo "ACTION_SLICE=${ACTION_SLICE}"
echo "FRAME_STRIDE=${FRAME_STRIDE}"
echo "ACTION_STRIDE=${ACTION_STRIDE}"
echo "SAMPLE_STRIDE=${SAMPLE_STRIDE}"
echo "FPS=${FPS}"
echo "NUM_HISTORY=${NUM_HISTORY}"
echo "NUM_FUTURE=${NUM_FUTURE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS}"
echo "MAX_KEEP_CKPTS=${MAX_KEEP_CKPTS}"
echo "SAVE_FINAL=${SAVE_FINAL}"
echo "VALIDATION_STEPS=${VALIDATION_STEPS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "==============================================="

train_args=(
  --data-root "${REAL_DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --svd-model-path "${SVD_MODEL_PATH}"
  --clip-model-path "${CLIP_MODEL_PATH}"
  --ckpt-path "${CTRL_WORLD_CKPT}"
  --instruction "${TASK}"
  --image-key "${IMAGE_KEY}"
  --action-key "${ACTION_KEY}"
  --action-slice "${ACTION_SLICE}"
  --num-history "${NUM_HISTORY}"
  --num-future "${NUM_FUTURE}"
  --frame-stride "${FRAME_STRIDE}"
  --action-stride "${ACTION_STRIDE}"
  --sample-stride "${SAMPLE_STRIDE}"
  --fps "${FPS}"
  --val-ratio "${VAL_RATIO}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --learning-rate "${LEARNING_RATE}"
  --max-train-steps "${MAX_TRAIN_STEPS}"
  --checkpointing-steps "${CHECKPOINTING_STEPS}"
  --max-keep-checkpoints "${MAX_KEEP_CKPTS}"
  --validation-steps "${VALIDATION_STEPS}"
  --validation-batches "${VALIDATION_BATCHES}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --mixed-precision "${MIXED_PRECISION}"
  --self-forcing-weight "${SELF_FORCING_WEIGHT}"
  --reward-loss-weight "${REWARD_LOSS_WEIGHT}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --seed "${SEED}"
)

if [[ -n "${RESUME_CKPT}" ]]; then
  train_args+=(--resume-from "${RESUME_CKPT}")
fi

if [[ "${SAVE_FINAL}" == "1" || "${SAVE_FINAL}" == "true" || "${SAVE_FINAL}" == "TRUE" ]]; then
  train_args+=(--save-final)
fi

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  real_world/world_model/train_hdf5_finetune.py \
  "${train_args[@]}"
