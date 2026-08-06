#!/usr/bin/env bash
# Batch entry for current MERL/MBRL/MFRL resumed logs.
# Usage:
#   bash scripts/plot_from_log_wm.sh ./checkpoints/MERL/EXP_DIR
# bash scripts/plot_from_log_wm.sh ./checkpoints/MERL/train_openvla-oft-SFT-libero_10-debug-3gpu-512x-loss_w-pro_MERL_0511
# MODE=mbrl bash scripts/plot_from_log_wm.sh ./checkpoints/MBRL/your_exp_dir
# MODE=mfrl bash scripts/plot_from_log_wm.sh ./checkpoints/MFRL/your_exp_dir
# Env:
#   MODE=merl|mbrl|mfrl OUT_DIR=... EXTRA_LOG=... SMOOTH=1 AUTO_GROUPS=true PYTHON_BIN=python

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="${1:-${LOG_DIR:-}}"
MODE="${MODE:-merl}"
MODE_UPPER="$(printf '%s' "${MODE}" | tr '[:lower:]' '[:upper:]')"

find_latest_experiment_dir() {
  local base_dir="$1"
  if [[ ! -d "${base_dir}" ]]; then
    return 1
  fi
  find "${base_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%T@\t%p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -f2-
}

DEFAULT_MODE_DIR="$(find_latest_experiment_dir "${REPO_ROOT}/checkpoints/${MODE_UPPER}" || true)"

if [[ -z "${TARGET}" ]]; then
  if [[ -n "${DEFAULT_MODE_DIR}" && -d "${DEFAULT_MODE_DIR}" ]]; then
    TARGET="${DEFAULT_MODE_DIR}"
  else
    TARGET="${REPO_ROOT}/tmp_files/ai_files"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SMOOTH="${SMOOTH:-1}"
AUTO_GROUPS="${AUTO_GROUPS:-true}"
LIST_KEYS="${LIST_KEYS:-false}"

if [[ -z "${OUT_DIR:-}" ]]; then
  if [[ -d "${TARGET}" ]]; then
    OUT_DIR="${TARGET}/plots_${MODE}"
  else
    OUT_DIR="$(dirname "${TARGET}")/plots_${MODE}"
  fi
fi

PLOT_ARGS=()
if [[ -d "${TARGET}" ]]; then
  PLOT_ARGS+=(--log_dir "${TARGET}" --recursive)
else
  PLOT_ARGS+=(--log_file "${TARGET}")
fi

if [[ -n "${EXTRA_LOG:-}" ]]; then
  PLOT_ARGS+=(--log_file "${EXTRA_LOG}")
fi

if [[ "${AUTO_GROUPS}" == "true" ]]; then
  PLOT_ARGS+=(--auto_prefix_groups)
fi

if [[ "${LIST_KEYS}" == "true" ]]; then
  PLOT_ARGS+=(--list_keys)
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/plot_log_metrics.py" \
  "${PLOT_ARGS[@]}" \
  --preset "${MODE}" \
  --out_dir "${OUT_DIR}" \
  --smooth "${SMOOTH}"

echo "plots saved to: ${OUT_DIR}"
