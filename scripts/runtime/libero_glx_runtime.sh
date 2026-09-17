#!/usr/bin/env bash
# MERL GLX runtime patch:
# 1. Reuse a healthy X display by default, or clean-restart Xvfb when requested.
# 2. Remove stale Xvfb lock files only after display health checks fail.
# 3. Keep outer GLX/Xvfb available, while LIBERO child envs default to EGL.

_merl_x_display_alive() {
    local display_num="$1"
    local display_ref=":${display_num}"
    if command -v xdpyinfo >/dev/null 2>&1; then
        xdpyinfo -display "$display_ref" >/dev/null 2>&1
        return $?
    fi

    local lock_file="/tmp/.X${display_num}-lock"
    local lock_pid=""
    if [ -f "$lock_file" ]; then
        lock_pid="$(tr -cd '0-9' < "$lock_file" || true)"
        if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

_merl_clean_x_display() {
    local display_num="$1"
    local lock_file="/tmp/.X${display_num}-lock"
    local socket_file="/tmp/.X11-unix/X${display_num}"
    local lock_pid=""
    local lock_args=""

    if [ -f "$lock_file" ]; then
        lock_pid="$(tr -cd '0-9' < "$lock_file" || true)"
    fi

    if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
        lock_args="$(ps -p "$lock_pid" -o args= 2>/dev/null || true)"
        case "$lock_args" in
            *Xvfb*" :${display_num}"*|*Xvfb*":${display_num} "*|*Xvfb*":${display_num}")
                echo "[preflight] clean-restart Xvfb on :${display_num} (pid=${lock_pid})" >&2
                kill "$lock_pid" 2>/dev/null || true
                local i
                for i in 1 2 3 4 5; do
                    if ! kill -0 "$lock_pid" 2>/dev/null; then
                        break
                    fi
                    sleep 1
                done
                if kill -0 "$lock_pid" 2>/dev/null; then
                    kill -9 "$lock_pid" 2>/dev/null || true
                fi
                ;;
            *)
                echo "[preflight] refusing to kill non-Xvfb process for :${display_num}: pid=${lock_pid}, args=${lock_args}" >&2
                exit 1
                ;;
        esac
    fi

    rm -f "$lock_file" "$socket_file"
}

_merl_ensure_x_display() {
    local display_num="$1"
    local display_ref=":${display_num}"

    if [ "${MERL_XVFB_CLEAN_START:-false}" = "true" ]; then
        _merl_clean_x_display "$display_num"
    fi

    if _merl_x_display_alive "$display_num"; then
        echo "[preflight] reuse active X display ${display_ref}" >&2
        return 0
    fi

    local lock_file="/tmp/.X${display_num}-lock"
    local socket_file="/tmp/.X11-unix/X${display_num}"
    local lock_pid=""
    if [ -f "$lock_file" ]; then
        lock_pid="$(tr -cd '0-9' < "$lock_file" || true)"
    fi

    if [ -z "$lock_pid" ] || ! kill -0 "$lock_pid" 2>/dev/null; then
        if [ -e "$lock_file" ] || [ -e "$socket_file" ]; then
            echo "[preflight] remove stale Xvfb files for ${display_ref}" >&2
            rm -f "$lock_file" "$socket_file"
        fi
    else
        echo "[preflight] X display ${display_ref} has live pid=${lock_pid} but is not reachable" >&2
        exit 1
    fi

    local xvfb_log="${MERL_XVFB_LOG_DIR:-/tmp}/merl_xvfb_${display_num}.log"
    Xvfb "$display_ref" -screen 0 "${MERL_XVFB_SCREEN:-1024x768x24}" >"$xvfb_log" 2>&1 &

    local ready=0
    local i
    for i in 1 2 3 4 5; do
        if _merl_x_display_alive "$display_num"; then
            ready=1
            break
        fi
        sleep 1
    done

    if [ "$ready" -ne 1 ]; then
        echo "[preflight] Xvfb failed to become ready on ${display_ref}" >&2
        if [ -f "$xvfb_log" ]; then
            tail -n 80 "$xvfb_log" >&2 || true
        fi
        exit 1
    fi
    echo "[preflight] Xvfb ready on ${display_ref}" >&2
}

_merl_visible_gpu_count() {
    local csv="${CUDA_VISIBLE_DEVICES:-}"
    if [ -z "$csv" ]; then
        echo 1
        return 0
    fi
    local old_ifs="$IFS"
    local -a items=()
    IFS=',' read -r -a items <<< "$csv"
    IFS="$old_ifs"
    echo "${#items[@]}"
}

ensure_merl_glx_runtime() {
    local display_default="${MERL_XVFB_DISPLAY:-:99}"
    export DISPLAY="${DISPLAY:-$display_default}"

    local display_num="${DISPLAY#:}"
    display_num="${display_num%%.*}"
    if [ -z "$display_num" ] || [ "$display_num" = "$DISPLAY" ]; then
        display_num=99
        export DISPLAY=":${display_num}"
    fi

    export MERL_XVFB_BASE_DISPLAY="${MERL_XVFB_BASE_DISPLAY:-$display_num}"
    export MERL_XVFB_DISPLAY_MODE="${MERL_XVFB_DISPLAY_MODE:-shared}"

    if [ "$MERL_XVFB_DISPLAY_MODE" = "per_actor" ] || [ "$MERL_XVFB_DISPLAY_MODE" = "per_rank" ]; then
        local display_count="${MERL_XVFB_DISPLAY_COUNT:-}"
        if [ -z "$display_count" ]; then
            display_count="$(_merl_visible_gpu_count)"
        fi
        if [ "$display_count" -lt 1 ]; then
            display_count=1
        fi
        export MERL_XVFB_DISPLAY_COUNT="$display_count"

        local idx=0
        while [ "$idx" -lt "$display_count" ]; do
            _merl_ensure_x_display "$((MERL_XVFB_BASE_DISPLAY + idx))"
            idx=$((idx + 1))
        done
        export DISPLAY=":${MERL_XVFB_BASE_DISPLAY}"
    else
        _merl_ensure_x_display "$display_num"
    fi

    local backend="${MUJOCO_GL:-${PYOPENGL_PLATFORM:-glx}}"
    if [ "$backend" = "osmesa" ] || [ "${PYOPENGL_PLATFORM:-}" = "osmesa" ]; then
        backend="osmesa"
    fi
    export MUJOCO_GL="$backend"
    export PYOPENGL_PLATFORM="$backend"
    export MERL_GLX_SOFTWARE="${MERL_GLX_SOFTWARE:-true}"
    if [ "$backend" = "glx" ] && [ "$MERL_GLX_SOFTWARE" = "true" ]; then
        export LIBGL_ALWAYS_SOFTWARE=1
        export LIBGL_DRI3_DISABLE=1
        export __GLX_VENDOR_LIBRARY_NAME=mesa
    else
        unset LIBGL_ALWAYS_SOFTWARE
        unset LIBGL_DRI3_DISABLE
        unset __GLX_VENDOR_LIBRARY_NAME
    fi
    export MERL_ENV_MP_START_METHOD="${MERL_ENV_MP_START_METHOD:-spawn}"
    export MERL_LIBERO_ENV_INIT_LOCK="${MERL_LIBERO_ENV_INIT_LOCK:-/tmp/merl_libero_glx_env_init.lock}"
    export MERL_LIBERO_ENV_INIT_LOCK_TIMEOUT_S="${MERL_LIBERO_ENV_INIT_LOCK_TIMEOUT_S:-300}"
    export MERL_LIBERO_ENV_INIT_LOCK_ENABLE="${MERL_LIBERO_ENV_INIT_LOCK_ENABLE:-auto}"
    export MERL_LIBERO_ENV_INIT_LOCK_SCOPE="${MERL_LIBERO_ENV_INIT_LOCK_SCOPE:-global}"
    export MERL_LIBERO_ENV_INIT_MAX_RETRY="${MERL_LIBERO_ENV_INIT_MAX_RETRY:-1}"
    export MERL_LIBERO_ENV_PARENT_MAX_RETRY="${MERL_LIBERO_ENV_PARENT_MAX_RETRY:-1}"
    export MERL_LIBERO_ENV_BACKEND="${MERL_LIBERO_ENV_BACKEND:-egl}"
    export MERL_LIBERO_RENDER_CUDA_VISIBLE_DEVICES="${MERL_LIBERO_RENDER_CUDA_VISIBLE_DEVICES:-${MERL_GLOBAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
    export MERL_LIBERO_ENV_SERVICE_ENABLE="${MERL_LIBERO_ENV_SERVICE_ENABLE:-true}"
    export MERL_LIBERO_PREFLIGHT="${MERL_LIBERO_PREFLIGHT:-true}"
    export MERL_LIBERO_EGL_DEVICE_ID="${MERL_LIBERO_EGL_DEVICE_ID:-0}"
    export MERL_LIBERO_GL_FALLBACK="${MERL_LIBERO_GL_FALLBACK:-none}"
    export MERL_LIBERO_FORCE_FALLBACK_AFTER_CRASH="${MERL_LIBERO_FORCE_FALLBACK_AFTER_CRASH:-true}"
    export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
    export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64"
}
