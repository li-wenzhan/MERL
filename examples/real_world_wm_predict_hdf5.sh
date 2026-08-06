#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-real_world/configs/real_wm_infer.yaml}"
HDF5="${HDF5:-tmp_files/hdf5/episode_25.hdf5}"
TASK="${TASK:-Put the red block into the box.}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/real_world_wm/episode_25_hdf5}"
DEVICE="${DEVICE:-cuda:0}"

IMAGE_KEY="${IMAGE_KEY:-observations/images/cam_high}"
ACTION_KEY="${ACTION_KEY:-action}"
ACTION_SLICE="${ACTION_SLICE:-right}"
WINDOW_MODE="${WINDOW_MODE:-predict}"
CURRENT_FRAME="${CURRENT_FRAME:-}"
PREDICT_FROM_FRAME="${PREDICT_FROM_FRAME:-${CURRENT_FRAME}}"
CURRENT_FRAME_RATIO="${CURRENT_FRAME_RATIO:-}"
NUM_HISTORY="${NUM_HISTORY:-8}"
NUM_FUTURE="${NUM_FUTURE:-8}"
INCLUDE_HISTORY_ACTIONS="${INCLUDE_HISTORY_ACTIONS:-1}"
FRAME_STRIDE="${FRAME_STRIDE:-8}"
ACTION_STRIDE="${ACTION_STRIDE:-${FRAME_STRIDE}}"
FPS="${FPS:-4}"
MAX_INPUT_FRAMES="${MAX_INPUT_FRAMES:-}"
VIDEO_STRIDE="${VIDEO_STRIDE:-1}"
NUM_FUTURE_FRAMES="${NUM_FUTURE_FRAMES:-${NUM_FUTURE}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
INPUT_MODE="${INPUT_MODE:-video}"
EXPORT_PRED_FRAMES="${EXPORT_PRED_FRAMES:-1}"
EXTRACT_ONLY="${EXTRACT_ONLY:-0}"
SLIDE_START_FRAME="${SLIDE_START_FRAME:-}"
SLIDE_END_FRAME="${SLIDE_END_FRAME:-}"
SLIDE_CHUNK_STEP="${SLIDE_CHUNK_STEP:-}"
SAVE_SLIDE_CHUNKS="${SAVE_SLIDE_CHUNKS:-1}"
SAVE_SLIDE_VIDEOS="${SAVE_SLIDE_VIDEOS:-0}"

EXTRACT_DIR="${EXTRACT_DIR:-${OUTPUT_DIR}/hdf5_inputs}"
HDF5_STEM="$(basename "${HDF5}")"
HDF5_STEM="${HDF5_STEM%.*}"
EXTRACTED_VIDEO="${EXTRACTED_VIDEO:-${EXTRACT_DIR}/${HDF5_STEM}_${ACTION_SLICE}.mp4}"
EXTRACTED_IMAGE_DIR="${EXTRACTED_IMAGE_DIR:-${EXTRACT_DIR}/${HDF5_STEM}_${ACTION_SLICE}_frames}"
EXTRACTED_CURRENT_IMAGE="${EXTRACTED_CURRENT_IMAGE:-${EXTRACT_DIR}/${HDF5_STEM}_${ACTION_SLICE}_current.png}"
EXTRACTED_GT_FUTURE_VIDEO="${EXTRACTED_GT_FUTURE_VIDEO:-${EXTRACT_DIR}/${HDF5_STEM}_${ACTION_SLICE}_gt_future.mp4}"
EXTRACTED_GT_FUTURE_IMAGE_DIR="${EXTRACTED_GT_FUTURE_IMAGE_DIR:-${EXTRACT_DIR}/${HDF5_STEM}_${ACTION_SLICE}_gt_future_frames}"
EXTRACTED_ACTIONS="${EXTRACTED_ACTIONS:-${EXTRACT_DIR}/${HDF5_STEM}_${ACTION_SLICE}_actions.npy}"
EXTRACTED_METADATA="${EXTRACTED_METADATA:-${EXTRACT_DIR}/${HDF5_STEM}_${ACTION_SLICE}_metadata.json}"

extract_args=(
  --hdf5 "${HDF5}"
  --output-actions "${EXTRACTED_ACTIONS}"
  --metadata-json "${EXTRACTED_METADATA}"
  --output-current-image "${EXTRACTED_CURRENT_IMAGE}"
  --image-key "${IMAGE_KEY}"
  --action-key "${ACTION_KEY}"
  --action-slice "${ACTION_SLICE}"
  --window-mode "${WINDOW_MODE}"
  --num-history "${NUM_HISTORY}"
  --num-future "${NUM_FUTURE}"
  --frame-stride "${FRAME_STRIDE}"
  --action-stride "${ACTION_STRIDE}"
  --fps "${FPS}"
)

if [[ "${INPUT_MODE}" == "video" ]]; then
  extract_args+=(--output-video "${EXTRACTED_VIDEO}")
  extract_args+=(--output-gt-future-video "${EXTRACTED_GT_FUTURE_VIDEO}")
elif [[ "${INPUT_MODE}" == "images" ]]; then
  extract_args+=(--output-image-dir "${EXTRACTED_IMAGE_DIR}")
  extract_args+=(--output-gt-future-image-dir "${EXTRACTED_GT_FUTURE_IMAGE_DIR}")
else
  echo "INPUT_MODE must be video or images, got ${INPUT_MODE}" >&2
  exit 2
fi

if [[ -n "${PREDICT_FROM_FRAME}" ]]; then
  extract_args+=(--current-frame "${PREDICT_FROM_FRAME}")
elif [[ -n "${CURRENT_FRAME_RATIO}" ]]; then
  extract_args+=(--current-frame-ratio "${CURRENT_FRAME_RATIO}")
fi

if [[ "${INCLUDE_HISTORY_ACTIONS}" == "1" || "${INCLUDE_HISTORY_ACTIONS}" == "true" ]]; then
  extract_args+=(--include-history-actions)
fi

echo "==== Real-world WM HDF5 predict configuration ===="
echo "CONFIG=${CONFIG}"
echo "HDF5=${HDF5}"
echo "TASK=${TASK}"
echo "IMAGE_KEY=${IMAGE_KEY}"
echo "ACTION_KEY=${ACTION_KEY}"
echo "ACTION_SLICE=${ACTION_SLICE}"
echo "WINDOW_MODE=${WINDOW_MODE}"
echo "PREDICT_FROM_FRAME=${PREDICT_FROM_FRAME:-auto}"
echo "CURRENT_FRAME_RATIO=${CURRENT_FRAME_RATIO:-auto}"
echo "NUM_HISTORY=${NUM_HISTORY}"
echo "NUM_FUTURE=${NUM_FUTURE}"
echo "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-config_default}"
echo "INCLUDE_HISTORY_ACTIONS=${INCLUDE_HISTORY_ACTIONS}"
echo "FRAME_STRIDE=${FRAME_STRIDE}"
echo "ACTION_STRIDE=${ACTION_STRIDE}"
echo "FPS=${FPS}"
echo "INPUT_MODE=${INPUT_MODE}"
echo "EXPORT_PRED_FRAMES=${EXPORT_PRED_FRAMES}"
echo "EXTRACT_ONLY=${EXTRACT_ONLY}"
echo "SLIDE_START_FRAME=${SLIDE_START_FRAME:-}"
echo "SLIDE_END_FRAME=${SLIDE_END_FRAME:-}"
echo "SLIDE_CHUNK_STEP=${SLIDE_CHUNK_STEP:-auto}"
echo "SAVE_SLIDE_VIDEOS=${SAVE_SLIDE_VIDEOS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DEVICE=${DEVICE}"
echo "=================================================="

if [[ -n "${SLIDE_START_FRAME}" || -n "${SLIDE_END_FRAME}" ]]; then
  if [[ -z "${SLIDE_START_FRAME}" || -z "${SLIDE_END_FRAME}" ]]; then
    echo "SLIDE_START_FRAME and SLIDE_END_FRAME must be set together." >&2
    exit 2
  fi

  slide_args=(
    --config "${CONFIG}"
    --hdf5 "${HDF5}"
    --output-dir "${OUTPUT_DIR}"
    --task "${TASK}"
    --device "${DEVICE}"
    --image-key "${IMAGE_KEY}"
    --action-key "${ACTION_KEY}"
    --action-slice "${ACTION_SLICE}"
    --start-frame "${SLIDE_START_FRAME}"
    --end-frame "${SLIDE_END_FRAME}"
    --num-history "${NUM_HISTORY}"
    --num-future "${NUM_FUTURE}"
    --frame-stride "${FRAME_STRIDE}"
    --action-stride "${ACTION_STRIDE}"
    --fps "${FPS}"
  )

  if [[ -n "${SLIDE_CHUNK_STEP}" ]]; then
    slide_args+=(--chunk-step "${SLIDE_CHUNK_STEP}")
  fi
  if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
    slide_args+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
  fi
  if [[ "${SAVE_SLIDE_CHUNKS}" == "1" || "${SAVE_SLIDE_CHUNKS}" == "true" ]]; then
    slide_args+=(--save-chunks)
  fi
  if [[ "${SAVE_SLIDE_VIDEOS}" == "1" || "${SAVE_SLIDE_VIDEOS}" == "true" ]]; then
    slide_args+=(--save-videos)
  fi
  if [[ "${EXPORT_PRED_FRAMES}" == "1" || "${EXPORT_PRED_FRAMES}" == "true" ]]; then
    slide_args+=(--export-frames)
  fi
  if [[ "${EXTRACT_ONLY}" == "1" || "${EXTRACT_ONLY}" == "true" ]]; then
    slide_args+=(--extract-only)
  fi

  python scripts/real_world_wm_hdf5_sliding_predict.py "${slide_args[@]}"
  exit 0
fi

python scripts/extract_real_world_wm_hdf5.py "${extract_args[@]}"

if [[ "${EXTRACT_ONLY}" == "1" || "${EXTRACT_ONLY}" == "true" ]]; then
  echo "[real-world-wm] EXTRACT_ONLY enabled; skip world model inference."
  echo "[real-world-wm] inspect ${EXTRACTED_CURRENT_IMAGE}, ${EXTRACTED_METADATA}, and extracted GT future media first."
  exit 0
fi

EFFECTIVE_CONFIG="${CONFIG}"
CLI_SUPPORTS_NUM_INFERENCE_STEPS=0
if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  if python -m real_world.world_model.cli_predict --help 2>&1 | grep -q -- "--num-inference-steps"; then
    CLI_SUPPORTS_NUM_INFERENCE_STEPS=1
  else
    EFFECTIVE_CONFIG="${EXTRACT_DIR}/real_wm_infer_num_steps_${NUM_INFERENCE_STEPS}.yaml"
    python scripts/override_real_world_wm_config.py \
      --config "${CONFIG}" \
      --output "${EFFECTIVE_CONFIG}" \
      --num-inference-steps "${NUM_INFERENCE_STEPS}"
    echo "[real-world-wm] cli_predict does not expose --num-inference-steps; using config override ${EFFECTIVE_CONFIG}"
  fi
fi

predict_args=(
  --config "${EFFECTIVE_CONFIG}"
  --actions "${EXTRACTED_ACTIONS}"
  --task "${TASK}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --input-fps "${FPS}"
  --video-stride "${VIDEO_STRIDE}"
  --num-future-frames "${NUM_FUTURE_FRAMES}"
)

if [[ -n "${NUM_INFERENCE_STEPS}" && "${CLI_SUPPORTS_NUM_INFERENCE_STEPS}" == "1" ]]; then
  predict_args+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi

if [[ "${INPUT_MODE}" == "video" ]]; then
  predict_args+=(--input-video "${EXTRACTED_VIDEO}")
else
  predict_args+=(--input-dir "${EXTRACTED_IMAGE_DIR}")
fi

if [[ -n "${MAX_INPUT_FRAMES}" ]]; then
  predict_args+=(--max-input-frames "${MAX_INPUT_FRAMES}")
fi

python -m real_world.world_model.cli_predict "${predict_args[@]}"

PRED_VIDEO="${OUTPUT_DIR}/pred_future.mp4"
PRED_FRAMES_DIR="${PRED_FRAMES_DIR:-${OUTPUT_DIR}/pred_future_frames}"
if [[ "${EXPORT_PRED_FRAMES}" == "1" || "${EXPORT_PRED_FRAMES}" == "true" ]]; then
  python scripts/export_real_world_wm_video_frames.py \
    --video "${PRED_VIDEO}" \
    --output-dir "${PRED_FRAMES_DIR}"
fi
