#!/usr/bin/env bash

# MERL Ray runtime patch:
# 1. Clean stale local Ray sessions before debug training.
# 2. Force local Ray unless explicitly disabled.
# 3. Use a short Ray temp dir and avoid Ray runtime_env agent by default.
# 4. Optionally pre-start a local Ray head via CLI for stable driver attach.

ensure_merl_ray_runtime() {
    local repo_root="${1:-${REPO_ROOT:-$(pwd)}}"
    local experiment_name="${2:-${EXPERIMENT_NAME:-default}}"

    export MERL_RAY_CLEAN_START="${MERL_RAY_CLEAN_START:-true}"
    export MERL_RAY_FORCE_LOCAL="${MERL_RAY_FORCE_LOCAL:-true}"
    export MERL_RAY_INCLUDE_DASHBOARD="${MERL_RAY_INCLUDE_DASHBOARD:-false}"
    export MERL_RAY_RUNTIME_ENV_MODE="${MERL_RAY_RUNTIME_ENV_MODE:-none}"
    export MERL_RAY_START_MODE="${MERL_RAY_START_MODE:-python}"
    export MERL_RAY_NODE_IP="${MERL_RAY_NODE_IP:-127.0.0.1}"
    export MERL_RAY_PORT="${MERL_RAY_PORT:-6379}"
    export MERL_RAY_INIT_TIMEOUT_S="${MERL_RAY_INIT_TIMEOUT_S:-180}"
    local ray_uid="${UID:-u}"
    export MERL_RAY_TMPDIR="${MERL_RAY_TMPDIR:-/tmp/merl_ray_${ray_uid}}"
    export RAY_TMPDIR="$MERL_RAY_TMPDIR"
    export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"
    export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
    export RAY_RUNTIME_ENV_LOG_TO_DRIVER="${RAY_RUNTIME_ENV_LOG_TO_DRIVER:-1}"
    export RAY_BACKEND_LOG_LEVEL="${RAY_BACKEND_LOG_LEVEL:-warning}"

    if [ "$MERL_RAY_FORCE_LOCAL" = "true" ]; then
        unset RAY_ADDRESS
        if [ "$MERL_RAY_START_MODE" != "cli" ]; then
            export MERL_RAY_ADDRESS=""
        fi
    fi

    if [ -z "${MERL_RAY_NUM_GPUS:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        export MERL_RAY_NUM_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
    fi

    mkdir -p "$MERL_RAY_TMPDIR"

    if [ "$MERL_RAY_CLEAN_START" = "true" ]; then
        echo "[preflight] ray clean start enabled: ray stop --force" >&2
        if command -v ray >/dev/null 2>&1; then
            ray stop --force || true
        else
            python -m ray.scripts.scripts stop --force || true
        fi
        case "$MERL_RAY_TMPDIR" in
            /tmp/merl_ray_*)
                find "$MERL_RAY_TMPDIR" -mindepth 1 -maxdepth 1 \( -name 'session_*' -o -name 'session_latest' \) -exec rm -rf {} + 2>/dev/null || true
                ;;
            *)
                echo "[preflight] skip Ray tmpdir cleanup for non-managed path: $MERL_RAY_TMPDIR" >&2
                ;;
        esac
    else
        echo "[preflight] ray clean start disabled" >&2
    fi

    if [ "$MERL_RAY_START_MODE" = "cli" ]; then
        if ! command -v ray >/dev/null 2>&1; then
            echo "[preflight] MERL_RAY_START_MODE=cli requires the ray command on PATH" >&2
            exit 1
        fi

        local start_args=(
            start
            --head
            --node-ip-address="$MERL_RAY_NODE_IP"
            --port="$MERL_RAY_PORT"
            --temp-dir="$MERL_RAY_TMPDIR"
            --include-dashboard="$MERL_RAY_INCLUDE_DASHBOARD"
            --disable-usage-stats
        )
        if [ -n "${MERL_RAY_NUM_GPUS:-}" ]; then
            start_args+=(--num-gpus="$MERL_RAY_NUM_GPUS")
        fi
        if [ -n "${MERL_RAY_NUM_CPUS:-}" ]; then
            start_args+=(--num-cpus="$MERL_RAY_NUM_CPUS")
        fi

        echo "[preflight] ray cli start: node_ip=${MERL_RAY_NODE_IP}, port=${MERL_RAY_PORT}, num_gpus=${MERL_RAY_NUM_GPUS:-auto}" >&2
        ray "${start_args[@]}"
        export MERL_RAY_ADDRESS="${MERL_RAY_ADDRESS:-${MERL_RAY_NODE_IP}:${MERL_RAY_PORT}}"
        export RAY_ADDRESS="$MERL_RAY_ADDRESS"
    fi

    echo "[preflight] Ray local runtime: tmpdir=${MERL_RAY_TMPDIR}, dashboard=${MERL_RAY_INCLUDE_DASHBOARD}, runtime_env_mode=${MERL_RAY_RUNTIME_ENV_MODE}, start_mode=${MERL_RAY_START_MODE}, address=${MERL_RAY_ADDRESS:-<python-local>}" >&2
}
