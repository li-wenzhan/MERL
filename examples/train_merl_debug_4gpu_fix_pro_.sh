set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ! Reader checklist:
# ! 1) Set CUDA_VISIBLE_DEVICES / NUM_GPUS / WM_GPU_IDX for your machine.
# ! 2) Set libero_pro_root in configs/evaluation_config.yaml; the rollout code now auto-derives the rest of the LIBERO_PRO subpaths.
# ! 3) Set SFT_MODEL_PATH / CKPT_PATH; ALIGN_PATH / world_model_config_path / libero_pro_eval_config_path default to repo-local files.
# ! 4) Optionally set LIBERO_PRO_ROOT only if you want an environment-variable fallback instead of config-driven import.
# ! 5) Run examples/generate_shared_wm_eval_dataset.sh before MERL so shared_wm_eval_root already contains .tar shards.

count_csv_items() {
    awk -F',' '{print NF}' <<< "$1"
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "[preflight] missing file: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [ ! -d "$1" ]; then
        echo "[preflight] missing directory: $1" >&2
        exit 1
    fi
}

require_divisible() {
    if [ "$2" -le 0 ] || [ $(( $1 % $2 )) -ne 0 ]; then
        echo "[preflight] $3: $1 is not divisible by $2" >&2
        exit 1
    fi
}

require_bool_flag() {
    case "$1" in
        true|false) ;;
        *)
            echo "[preflight] $2 must be 'true' or 'false', got '$1'" >&2
            exit 1
            ;;
    esac
}

require_tar_split() {
    local base_dir="$1"
    local split_name="$2"
    local split_dir="${base_dir%/}/${split_name}"
    local first_tar=""

    if [ ! -d "$split_dir" ]; then
        echo "[preflight] missing WM eval split directory: $split_dir" >&2
        echo "[preflight] run examples/generate_shared_wm_eval_dataset.sh first" >&2
        exit 1
    fi

    first_tar=$(find "$split_dir" -type f -name '*.tar' -print -quit 2>/dev/null || true)
    if [ -z "$first_tar" ]; then
        echo "[preflight] no WM eval shards found under: $split_dir" >&2
        echo "[preflight] run examples/generate_shared_wm_eval_dataset.sh first" >&2
        exit 1
    fi
}

has_resume_state() {
    local dir="$1"

    if [ -f "$dir/resume/resume_state_latest.json" ] || [ -f "$dir/resume_state_latest.json" ]; then
        return 0
    fi

    return 1
}

has_resume_artifacts() {
    local dir="$1"
    local first_log=""

    if has_resume_state "$dir"; then
        return 0
    fi

    first_log=$(find "$dir" -maxdepth 1 -type f -name 'run_*.log' -print -quit 2>/dev/null || true)
    if [ -n "$first_log" ] && has_valid_actor_checkpoint "$dir"; then
        return 0
    fi

    return 1
}

has_any_experiment_artifacts() {
    local dir="$1"
    local first_log=""

    if [ -d "$dir/actor" ] || [ -f "$dir/resume/resume_state_latest.json" ] || [ -f "$dir/resume_state_latest.json" ]; then
        return 0
    fi

    first_log=$(find "$dir" -maxdepth 1 -type f -name 'run_*.log' -print -quit 2>/dev/null || true)
    [ -n "$first_log" ]
}

has_model_or_resume_artifacts() {
    local dir="$1"
    local first_checkpoint_file=""

    if [ -d "$dir/actor" ] || [ -d "$dir/critic" ] || [ -d "$dir/world_model" ]; then
        return 0
    fi
    if [ -f "$dir/resume/resume_state_latest.json" ] || [ -f "$dir/resume_state_latest.json" ]; then
        return 0
    fi

    first_checkpoint_file=$(find "$dir" -maxdepth 4 -type f \( \
        -name 'checkpoint_meta.json' -o \
        -name 'rank_*.pt' -o \
        -name 'world_model.pth' -o \
        -name 'pytorch_model.bin' -o \
        -name 'model.safetensors' -o \
        -name 'adapter_model.bin' -o \
        -name 'adapter_model.safetensors' -o \
        -name 'resume_state_*.json' \
    \) -print -quit 2>/dev/null || true)
    [ -n "$first_checkpoint_file" ]
}

has_valid_actor_checkpoint() {
    local dir="$1"
    local actor_dir="$dir/actor"
    local first_checkpoint_file=""
    local meta_path=""
    local checkpoint_dir=""
    local first_rank_shard=""

    if [ ! -d "$actor_dir" ]; then
        return 1
    fi

    first_checkpoint_file=$(find "$actor_dir" -type f \( \
        -name 'pytorch_model.bin' -o \
        -name 'model.safetensors' -o \
        -name 'pytorch_model.bin.index.json' -o \
        -name 'model.safetensors.index.json' -o \
        -name 'adapter_model.bin' -o \
        -name 'adapter_model.safetensors' \
    \) -print -quit 2>/dev/null || true)
    if [ -n "$first_checkpoint_file" ]; then
        return 0
    fi

    while IFS= read -r meta_path; do
        checkpoint_dir=$(dirname "$meta_path")
        if grep -Eq '"format"[[:space:]]*:[[:space:]]*"fsdp_local_state_dict"' "$meta_path"; then
            continue
        fi
        first_rank_shard=$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'rank_*.pt' -print -quit 2>/dev/null || true)
        if [ -n "$first_rank_shard" ]; then
            return 0
        fi
    done < <(find "$actor_dir" -type f -name 'checkpoint_meta.json' -print 2>/dev/null)

    return 1
}

resolve_resume_enable() {
    case "$RESUME_ENABLE" in
        true|false)
            return 0
            ;;
        auto)
            if [ -d "$RESUME_DIR" ] && has_resume_artifacts "$RESUME_DIR"; then
                RESUME_ENABLE=true
                echo "[preflight] auto resume enabled from: $RESUME_DIR" >&2
            else
                RESUME_ENABLE=false
                echo "[preflight] auto resume disabled: no resumable state under $RESUME_DIR" >&2
            fi
            ;;
        *)
            echo "[preflight] RESUME_ENABLE must be 'true', 'false', or 'auto', got '$RESUME_ENABLE'" >&2
            exit 1
            ;;
    esac
}

# ! todo: expose the actor GPUs plus one dedicated WM GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MERL_GLOBAL_CUDA_VISIBLE_DEVICES="${MERL_GLOBAL_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"

# For NCCL runtime
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [ -z "${NCCL_DEBUG_SUBSYS:-}" ]; then
    unset NCCL_DEBUG_SUBSYS
else
    export NCCL_DEBUG_SUBSYS
fi
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
NCCL_IFNAME_FALLBACK="${NCCL_SOCKET_IFNAME:-}"
NCCL_IFNAME_FALLBACK="${NCCL_IFNAME_FALLBACK%%,*}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_IFNAME_FALLBACK}"
if [ -z "$GLOO_SOCKET_IFNAME" ]; then
    export GLOO_SOCKET_IFNAME=bond0
fi

# For headless GLX rendering
export MERL_XVFB_CLEAN_START="${MERL_XVFB_CLEAN_START:-true}"
export MERL_XVFB_DISPLAY_MODE="${MERL_XVFB_DISPLAY_MODE:-shared}"
export MERL_GLX_SOFTWARE="${MERL_GLX_SOFTWARE:-true}"
export MERL_LIBERO_ENV_BACKEND="${MERL_LIBERO_ENV_BACKEND:-egl}"
export MERL_LIBERO_RENDER_CUDA_VISIBLE_DEVICES="${MERL_LIBERO_RENDER_CUDA_VISIBLE_DEVICES:-$MERL_GLOBAL_CUDA_VISIBLE_DEVICES}"
export MERL_LIBERO_EGL_DEVICE_ID="${MERL_LIBERO_EGL_DEVICE_ID:-0}"
export MERL_LIBERO_GL_FALLBACK="${MERL_LIBERO_GL_FALLBACK:-none}"
source "${SCRIPT_DIR}/libero_glx_runtime.sh"
ensure_merl_glx_runtime
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64"

if [ -n "${LIBERO_PRO_ROOT:-}" ]; then
    export PYTHONPATH="${LIBERO_PRO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"  # optional fallback; prefer configs/evaluation_config.yaml:libero_pro_root
fi

# export NCCL_DEBUG=WARN
export WANDB_MODE=disabled
export WANDB_DISABLED=true
# export WANDB_API_KEY="86afa5b168a8fbf4a1dfd98145f0c5d133ca103b"  #! wandb api key
# export WANDB_MODE=offline

case "${PYTORCH_CUDA_ALLOC_CONF:-}" in
    *expandable_segments*)
        echo "[preflight] disable PYTORCH_CUDA_ALLOC_CONF expandable_segments for Ray+FSDP+Llama stability" >&2
        unset PYTORCH_CUDA_ALLOC_CONF
        ;;
esac
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM=true
# Debug-only CUDA checks; keep off for normal training unless debugging kernel failures.
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export TORCH_USE_CUDA_DSA=${TORCH_USE_CUDA_DSA:-0}

export ROBOT_PLATFORM=LIBERO  #! Use LIBERO: ROBOT_PLATFORM=LIBERO,  Use Robotwin ROBOT_PLATFORM=ALOHA

PROJECT_NAME="MERL"
SFT_MODEL_PATH="/mnt/afs/L202500276/model/openvla_oft_sft/Openvla-oft-SFT-libero10-trajall"  # ! server OpenVLA-OFT SFT checkpoint
CKPT_PATH="./checkpoints"  # todo: saved ckpt path, include policy_model, world_model

DATASET_NAME="libero_10"  #! change the dataset name
VLA_NAME="openvla-oft"
EXPERIMENT_DATE_TAG="0518"
EXPERIMENT_PROFILE_TAG="debug-3gpu-512x-loss_w-pro"
EXPERIMENT_BASE_NAME="train_${VLA_NAME}-SFT-${DATASET_NAME}-${EXPERIMENT_PROFILE_TAG}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_BASE_NAME}_${PROJECT_NAME}_${EXPERIMENT_DATE_TAG}}"
EXPERIMENT_OUTPUT_DIR="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$EXPERIMENT_OUTPUT_DIR/actor_assets}"
RESUME_ENABLE="${RESUME_ENABLE:-auto}"  # true / false / auto
RESUME_DIR="${RESUME_DIR:-$EXPERIMENT_OUTPUT_DIR}"
RESUME_REQUEST_MODE="$RESUME_ENABLE"
ALLOW_FRESH_WITH_STALE_LOGS="${ALLOW_FRESH_WITH_STALE_LOGS:-true}"
VLA_DATASET_STATS_PATH="$SFT_MODEL_PATH/dataset_statistics.json"  # ! requires dataset_statistics.json to exist under SFT_MODEL_PATH
NUM_GPUS=${NUM_GPUS:-3}  #! 总 GPU 数量 - 1; 剩 1 个给wm trainer
WM_GPU_IDX=${WM_GPU_IDX:-3}  #! dedicate the 3rd GPU to wm trainer
NUM_NODES=1
ALIGN_PATH="${REPO_ROOT}/configs/align.json"  # ! repo-local default; override only if your runtime env file lives elsewhere
AUTO_ADJUST_GPU_LAYOUT=${AUTO_ADJUST_GPU_LAYOUT:-true}
export MERL_RAY_START_MODE="${MERL_RAY_START_MODE:-cli}"
source "${SCRIPT_DIR}/ray_runtime.sh"
ensure_merl_ray_runtime "$REPO_ROOT" "$EXPERIMENT_NAME"

num_trials_per_task=6  #! 默认: 50, 3; 8
n_samples=4  #! 默认: 8, 需要 > micro_batch_size, 最好是; 1
action_chunks_len=8  #! 默认：8 (写死的？不要改)
#! traj_len = rollout_steps // action_chunks_len  -> 最好 % traj_mini_batch_size == 0，代码也支持自动降块
#! 其中：traj_mini_batch_size = train_batch_size * action_chunks_len（8）
#! rollout_steps(512) = N * train_batch_size * action_chunks_len**2（64）

train_batch_size=6  #! 默认: 64, 会被实际 //= NUM_GPUS (其结果需 > 0), 需要 == NUM_GPUS * N; < num_trials_per_task * 10, 应该 = log_prob_micro_batch_size / 2 * NUM_GPUS * N; 2
val_batch_size=3  #! fast/stable eval; one local env per actor GPU
rollout_batch_size=3
ppo_mini_batch_size=12  #! 用单个 global PPO minibatch 降低 12-sample 小批量下的高方差大步更新
micro_batch_size=1  # 默认: 1, for rollout train
val_micro_batch_size=1  #! 默认: 8, for rollout eval
wm_evolving_serial_samples=True  #! keep n_samples>1 globally, but run one LIBERO WM env per local call

log_prob_micro_batch_size=3  #! global value; FSDP divides by NUM_GPUS, so each GPU uses 1 for ref/old log-prob
#! 所以必须是: log_prob_micro_batch_size = N * NUM_GPUS
traj_mini_batch_size=12  #! keep divisible by NUM_GPUS for 3 actor GPUs
enable_gradient_checkpointing=True  #! 显著降低 actor 更新峰值显存
val_only=False  #! 仅 eval; 默认: False, 先测试后训练
val_before_train=False
wandb_mode="disabled"  #! 默认: online
# 原实现（带wandb）: trainer.logger=['console','wandb'] \
# wandb_mode=online

# actor_rollout_ref.actor.fsdp_config.grad_offload=True
# actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
# actor_rollout_ref.ref.fsdp_config.param_offload=True

#! added below
world_model_config_path="${REPO_ROOT}/configs/wm_online_config.py"  # ! repo-local default; override only if you use a custom wm_online_config.py
use_libero_pro=True  # ! keep True when using LIBERO_PRO as the online environment
libero_pro_eval_config_path="${REPO_ROOT}/configs/evaluation_config.yaml"  # ! repo-local default; override only if you use a custom evaluation_config.yaml

train_mode="MERL"  #! 改训练模式
world_model_enable=True
save_freq=5  #! save actor checkpoint every 5 global steps
save_freq_wm_outer=5
save_freq_wm_inner=1
wm_inner_steps=100
wm_eval_interval=5
test_freq=5
wm_weak_update_ratio_threshold=0.95
wm_weak_update_exit_ratio_threshold=0.90
wm_sparse_real_interval=5
wm_sparse_real_prompts=1
wm_fine_tune=True
strict_mode_assert=True
strict_validate_rollout=False
final_val_after_train=False
world_model_num_inference_steps=30
shared_wm_eval_root="./tmp_files/wm_eval_shared/$DATASET_NAME"  # ! todo: keep consistent with examples/generate_shared_wm_eval_dataset.sh
shared_wm_eval_global_steps=0
shared_wm_eval_full_interval=10
world_model_mini_eval_samples=24
world_model_full_eval_samples=48
replay_pool_capacity=128
video_save_interval=5
persist_imag_rollout_shards=False
persist_replay_pool=True
replay_pool_keep_last=1
replay_pool_max_gb=15

# Fast full-task experiment profile.
train_batch_size=3
wm_inner_steps=50
wm_eval_interval=10
test_freq=20
world_model_mini_eval_samples=12
world_model_full_eval_samples=24
allowed_task_ids="[0,1,2,3,4,5,6,7,8,9]"
train_max_steps=384
eval_max_steps=512
imag_horizon_max=192
rollout_temperature=0.8
max_wm_ratio=0.25
imag_ratio_max=0.25
imag_weight_min=0.0
wm_sample_weight_max=0.25
wm_ratio_rounding=carry
wm_ratio_ref_loss=0.50
wm_ratio_up_min=0.02
wm_ratio_up_max=0.08
wm_ratio_down_max=0.10
actor_lr=8e-7
actor_clip_ratio_high=0.08
actor_clip_ratio_low=0.05
actor_ppo_kl_hard_limit=0.12
# Temporary MERL-only reward guard. Remove once imagined reward routing is principled.
use_wm_reward_proxy=False
wm_real_anchor_reward=True
require_wm_anchor_reward=True
disable_wm_kl_penalty=False
wm_grpo_uid_mode=anchor
zero_unanchored_wm_weight=True
wm_actor_anchor_reward_min=1.0
wm_ratio_cooldown_steps=3
merl_imagined_reward_confidence=0.10
merl_imagined_reward_weight_cap=0.10
imag_advantage_abs_clip=2.0

require_equal() {
    if [ "$1" != "$2" ]; then
        echo "[preflight] $3: expected '$2' but got '$1'" >&2
        exit 1
    fi
}

require_contains() {
    case "$1" in
        *"$2"*) ;;
        *)
            echo "[preflight] $3: '$1' does not contain '$2'" >&2
            exit 1
            ;;
    esac
}

if [ "$n_samples" -le 1 ]; then
    echo "[preflight] GRPO requires n_samples > 1" >&2
    exit 1
fi

resolve_resume_enable
require_bool_flag "$RESUME_ENABLE" "RESUME_ENABLE"
require_bool_flag "$ALLOW_FRESH_WITH_STALE_LOGS" "ALLOW_FRESH_WITH_STALE_LOGS"

visible_gpu_count=$(count_csv_items "$CUDA_VISIBLE_DEVICES")
if [ "$AUTO_ADJUST_GPU_LAYOUT" = "true" ]; then
    if [ "$visible_gpu_count" -le 1 ]; then
        NUM_GPUS=1
        WM_GPU_IDX=0
        echo "[preflight] single-GPU fallback enabled: NUM_GPUS=$NUM_GPUS, WM_GPU_IDX=$WM_GPU_IDX" >&2
    elif [ "$visible_gpu_count" -lt $((NUM_GPUS + 1)) ]; then
        NUM_GPUS=$((visible_gpu_count - 1))
        WM_GPU_IDX=$((visible_gpu_count - 1))
        echo "[preflight] auto-adjust GPU layout: NUM_GPUS=$NUM_GPUS, WM_GPU_IDX=$WM_GPU_IDX" >&2
    fi
fi

if [ "$visible_gpu_count" -lt 1 ]; then
    echo "[preflight] CUDA_VISIBLE_DEVICES is empty" >&2
    exit 1
fi
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "[preflight] NUM_GPUS must be >= 1 after auto-adjust" >&2
    exit 1
fi
if [ "$WM_GPU_IDX" -lt 0 ] || [ "$WM_GPU_IDX" -ge "$visible_gpu_count" ]; then
    echo "[preflight] WM_GPU_IDX=$WM_GPU_IDX is out of range for CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

require_divisible "$train_batch_size" "$NUM_GPUS" "train_batch_size / NUM_GPUS"
require_divisible "$val_batch_size" "$NUM_GPUS" "val_batch_size / NUM_GPUS"
require_divisible "$ppo_mini_batch_size" "$NUM_GPUS" "ppo_mini_batch_size / NUM_GPUS"
require_divisible "$log_prob_micro_batch_size" "$NUM_GPUS" "log_prob_micro_batch_size / NUM_GPUS"
require_divisible "$train_max_steps" "$action_chunks_len" "train_max_steps / action_chunks_len"
require_divisible "$eval_max_steps" "$action_chunks_len" "eval_max_steps / action_chunks_len"
require_divisible "$imag_horizon_max" "$action_chunks_len" "imag_horizon_max / action_chunks_len"
if [ "$eval_max_steps" -lt "$train_max_steps" ]; then
    echo "[preflight] eval_max_steps must be >= train_max_steps" >&2
    exit 1
fi

require_equal "$PROJECT_NAME" "$train_mode" "PROJECT_NAME/train_mode mismatch"
require_contains "$EXPERIMENT_NAME" "$train_mode" "EXPERIMENT_NAME/train_mode mismatch"
require_equal "$world_model_enable" "True" "MERL requires world_model.enable=True"
require_equal "$wm_fine_tune" "True" "MERL requires wm_fine_tune=True"

require_file "$ALIGN_PATH"
require_file "$world_model_config_path"
require_file "$libero_pro_eval_config_path"
require_file "$VLA_DATASET_STATS_PATH"
require_dir "$SFT_MODEL_PATH"
if [ "$use_libero_pro" = "True" ] || [ "$use_libero_pro" = "true" ]; then
    python "$REPO_ROOT/scripts/preflight_libero_pro_compat.py" \
        --config "$libero_pro_eval_config_path"
    if [ "${MERL_LIBERO_PREFLIGHT:-true}" = "true" ]; then
        python "$REPO_ROOT/scripts/preflight_libero_env_service.py" \
            --config "$libero_pro_eval_config_path" \
            --task-suite "$DATASET_NAME" \
            --task-id 0 \
            --trial-id 0 \
            --num-steps-wait 1
    fi
fi
if [ "$world_model_enable" = "True" ] || [ "$world_model_enable" = "true" ]; then
    python "$REPO_ROOT/scripts/preflight_world_model_backbone.py" \
        --config "$world_model_config_path"
fi
require_tar_split "$shared_wm_eval_root" "wm_eval_fixed_mini"
require_tar_split "$shared_wm_eval_root" "wm_eval_fixed_full"

if [ "$RESUME_ENABLE" = "true" ]; then
    require_dir "$RESUME_DIR"
    if ! has_resume_artifacts "$RESUME_DIR"; then
        echo "[preflight] resume requested but no valid actor checkpoint was found under: $RESUME_DIR" >&2
        echo "[preflight] incomplete global_step_* directories are not resumable; use a valid RESUME_DIR or restart with RESUME_ENABLE=false" >&2
        exit 1
    fi
else
    if [ -d "$EXPERIMENT_OUTPUT_DIR" ] && has_any_experiment_artifacts "$EXPERIMENT_OUTPUT_DIR"; then
        if [ "$ALLOW_FRESH_WITH_STALE_LOGS" = "true" ] && ! has_model_or_resume_artifacts "$EXPERIMENT_OUTPUT_DIR"; then
            echo "[preflight] fresh run allowed under existing log-only directory: $EXPERIMENT_OUTPUT_DIR" >&2
            echo "[preflight] previous run_*.log files are kept for plotting/debug; no checkpoint/resume state will be loaded" >&2
        else
            echo "[preflight] existing model/resume artifacts found under: $EXPERIMENT_OUTPUT_DIR" >&2
            if [ "$RESUME_REQUEST_MODE" = "auto" ]; then
                echo "[preflight] auto did not find a valid resumable checkpoint; change RESUME_DIR/EXPERIMENT_NAME or set ALLOW_FRESH_WITH_STALE_LOGS=true only for log-only dirs" >&2
            else
                echo "[preflight] set RESUME_ENABLE=true to continue or change EXPERIMENT_NAME for a fresh run" >&2
            fi
            exit 1
        fi
    fi
fi

bash "$REPO_ROOT/examples/overwrite_vla_ckpt_utils.sh" "$SFT_MODEL_PATH" "$ACTOR_MODEL_PATH"
require_file "$ACTOR_MODEL_PATH/config.json"
require_file "$ACTOR_MODEL_PATH/dataset_statistics.json"

# HYDRA_FULL_ERROR=1 xvfb-run -a -s "-screen 0 1024x768x24" python -u -m verl.trainer.main_ppo \
HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=$num_trials_per_task \
    data.n_samples=$n_samples \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    +data.rollout_batch_size=$rollout_batch_size \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=$action_chunks_len \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=$actor_clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_low=$actor_clip_ratio_low \
    ++actor_rollout_ref.actor.ppo_kl_hard_limit=$actor_ppo_kl_hard_limit \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=$traj_mini_batch_size \
    actor_rollout_ref.model.enable_gradient_checkpointing=$enable_gradient_checkpointing \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=$val_micro_batch_size \
    actor_rollout_ref.rollout.temperature=$rollout_temperature \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=$micro_batch_size \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$ACTOR_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    "++actor_rollout_ref.rollout.allowed_task_ids=$allowed_task_ids" \
    ++actor_rollout_ref.rollout.train_max_steps=$train_max_steps \
    ++actor_rollout_ref.rollout.eval_max_steps=$eval_max_steps \
    ++actor_rollout_ref.rollout.save_training_videos=True \
    ++actor_rollout_ref.rollout.video_save_interval=$video_save_interval \
    ++actor_rollout_ref.rollout.max_env_videos_per_task=1 \
    ++actor_rollout_ref.rollout.max_wm_videos_per_task=1 \
    ++actor_rollout_ref.rollout.use_libero_pro=$use_libero_pro \
    ++actor_rollout_ref.rollout.env_service_enable=${MERL_LIBERO_ENV_SERVICE_ENABLE:-true} \
    ++actor_rollout_ref.rollout.wm_evolving_serial_samples=$wm_evolving_serial_samples \
    ++actor_rollout_ref.rollout.libero_pro_eval_config_path=$libero_pro_eval_config_path \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    ++actor_rollout_ref.ref.vla=$VLA_NAME \
    ++actor_rollout_ref.ref.action_token_len=7 \
    ++actor_rollout_ref.ref.action_chunks_len=$action_chunks_len \
    ++actor_rollout_ref.ref.use_proprio=False \
    ++actor_rollout_ref.ref.use_remove_padding=False \
    ++actor_rollout_ref.world_model.enable=$world_model_enable \
    ++actor_rollout_ref.world_model.config_path=$world_model_config_path \
    ++actor_rollout_ref.world_model.fine_tune=$wm_fine_tune \
    ++actor_rollout_ref.world_model.training_steps_per_epoch=5000 \
    ++actor_rollout_ref.world_model.wm_inner_steps=$wm_inner_steps \
    ++actor_rollout_ref.world_model.reward_model.reward_thr=0.5 \
    ++actor_rollout_ref.world_model.save_freq_wm_outer=$save_freq_wm_outer \
    ++actor_rollout_ref.world_model.save_freq_wm_inner=$save_freq_wm_inner \
    ++actor_rollout_ref.world_model.lr=1e-5 \
    ++actor_rollout_ref.world_model.batch_size=4 \
    ++actor_rollout_ref.world_model.num_workers=2 \
    ++actor_rollout_ref.world_model.num_inference_steps=$world_model_num_inference_steps \
    ++actor_rollout_ref.world_model.eval_num_inference_steps=$world_model_num_inference_steps \
    ++actor_rollout_ref.world_model.wm_eval_interval=$wm_eval_interval \
    ++actor_rollout_ref.world_model.wm_warmup_steps=4 \
    ++actor_rollout_ref.world_model.min_real_ratio=0.0 \
    ++actor_rollout_ref.world_model.max_wm_ratio=$max_wm_ratio \
    ++actor_rollout_ref.world_model.imag_ratio_min=0.15 \
    ++actor_rollout_ref.world_model.imag_ratio_max=$imag_ratio_max \
    ++actor_rollout_ref.world_model.imag_ratio_gamma=1.5 \
    ++actor_rollout_ref.world_model.imag_confidence_ema_alpha=0.8 \
    ++actor_rollout_ref.world_model.weak_update_enable=True \
    ++actor_rollout_ref.world_model.weak_update_ratio_threshold=$wm_weak_update_ratio_threshold \
    ++actor_rollout_ref.world_model.weak_update_exit_ratio_threshold=$wm_weak_update_exit_ratio_threshold \
    ++actor_rollout_ref.world_model.wm_sparse_real_interval=$wm_sparse_real_interval \
    ++actor_rollout_ref.world_model.wm_sparse_real_prompts=$wm_sparse_real_prompts \
    ++actor_rollout_ref.world_model.real_prompt_min_for_wm=1 \
    ++actor_rollout_ref.world_model.wm_rollout_n_samples_max=8 \
    ++actor_rollout_ref.world_model.imag_horizon_min=128 \
    ++actor_rollout_ref.world_model.imag_horizon_max=$imag_horizon_max \
    ++actor_rollout_ref.world_model.imag_obs_error_scale=8.0 \
    ++actor_rollout_ref.world_model.imag_done_error_scale=2.0 \
    ++actor_rollout_ref.world_model.imag_weight_min=$imag_weight_min \
    ++actor_rollout_ref.world_model.imag_weight_eta=1.5 \
    ++actor_rollout_ref.world_model.wm_sample_weight_max=$wm_sample_weight_max \
    ++actor_rollout_ref.world_model.wm_ratio_rounding=$wm_ratio_rounding \
    ++actor_rollout_ref.world_model.wm_ratio_ref_loss=$wm_ratio_ref_loss \
    ++actor_rollout_ref.world_model.wm_ratio_up_min=$wm_ratio_up_min \
    ++actor_rollout_ref.world_model.wm_ratio_up_max=$wm_ratio_up_max \
    ++actor_rollout_ref.world_model.wm_ratio_down_max=$wm_ratio_down_max \
    ++actor_rollout_ref.world_model.imag_priority_eps=0.001 \
    ++actor_rollout_ref.world_model.imag_priority_beta=1.0 \
    ++actor_rollout_ref.world_model.use_wm_reward_proxy=$use_wm_reward_proxy \
    ++actor_rollout_ref.world_model.wm_real_anchor_reward=$wm_real_anchor_reward \
    ++actor_rollout_ref.world_model.require_wm_anchor_reward=$require_wm_anchor_reward \
    ++actor_rollout_ref.world_model.disable_wm_kl_penalty=$disable_wm_kl_penalty \
    ++actor_rollout_ref.world_model.wm_grpo_uid_mode=$wm_grpo_uid_mode \
    ++actor_rollout_ref.world_model.zero_unanchored_wm_weight=$zero_unanchored_wm_weight \
    ++actor_rollout_ref.world_model.wm_actor_anchor_reward_min=$wm_actor_anchor_reward_min \
    ++actor_rollout_ref.world_model.wm_ratio_cooldown_steps=$wm_ratio_cooldown_steps \
    ++actor_rollout_ref.world_model.merl_imagined_reward_hard_constraint=True \
    ++actor_rollout_ref.world_model.merl_imagined_reward_confidence=$merl_imagined_reward_confidence \
    ++actor_rollout_ref.world_model.merl_imagined_reward_weight_cap=$merl_imagined_reward_weight_cap \
    ++actor_rollout_ref.world_model.imag_advantage_abs_clip=$imag_advantage_abs_clip \
    ++actor_rollout_ref.world_model.persist_imag_rollout_shards=$persist_imag_rollout_shards \
    ++actor_rollout_ref.world_model.fixed_eval_enabled=True \
    ++actor_rollout_ref.world_model.fixed_eval_root=$shared_wm_eval_root \
    ++actor_rollout_ref.world_model.fixed_eval_global_steps=$shared_wm_eval_global_steps \
    ++actor_rollout_ref.world_model.fixed_eval_full_interval=$shared_wm_eval_full_interval \
    ++actor_rollout_ref.world_model.fixed_eval_mini_batch_size=4 \
    ++actor_rollout_ref.world_model.fixed_eval_full_batch_size=4 \
    ++actor_rollout_ref.world_model.fixed_eval_mini_samples=$world_model_mini_eval_samples \
    ++actor_rollout_ref.world_model.fixed_eval_full_samples=$world_model_full_eval_samples \
    ++actor_rollout_ref.world_model.fixed_eval_mini_windows_per_episode=2 \
    ++actor_rollout_ref.world_model.fixed_eval_full_windows_per_episode=4 \
    ++actor_rollout_ref.world_model.fixed_eval_window_selection=uniform \
    ++actor_rollout_ref.rollout_base_dir=./tmp_files/rollout/$EXPERIMENT_NAME \
    ++actor_rollout_ref.wm_gpu_idx=$WM_GPU_IDX \
    ++prio_real_capacity=$replay_pool_capacity \
    ++prio_wm_capacity=$replay_pool_capacity \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    trainer.wandb_mode=$wandb_mode \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$EXPERIMENT_OUTPUT_DIR \
    ++trainer.resume.enable=$RESUME_ENABLE \
    ++trainer.resume.resume_dir=$RESUME_DIR \
    ++trainer.resume.persist_replay_pool=$persist_replay_pool \
    ++trainer.resume.replay_pool_keep_last=$replay_pool_keep_last \
    ++trainer.resume.replay_pool_max_gb=$replay_pool_max_gb \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=100 \
    trainer.val_only=$val_only \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    ++trainer.ray_startup_probe_timeout_s=180 \
    ++trainer.ray_tmpdir=$MERL_RAY_TMPDIR \
    ++trainer.ray_include_dashboard=$MERL_RAY_INCLUDE_DASHBOARD \
    ++trainer.ray_runtime_env_mode=$MERL_RAY_RUNTIME_ENV_MODE \
    ++trainer.ray_address=$MERL_RAY_ADDRESS \
    ++trainer.ray_node_ip_address=$MERL_RAY_NODE_IP \
    ++trainer.ray_init_timeout_s=$MERL_RAY_INIT_TIMEOUT_S \
    ++trainer.ray_num_gpus=$MERL_RAY_NUM_GPUS \
    trainer.val_before_train=$val_before_train \
    ++trainer.final_val_after_train=$final_val_after_train \
    ++trainer.strict_mode_assert=$strict_mode_assert \
    ++trainer.strict_validate_rollout=$strict_validate_rollout \
    ++trainer.train_mode=$train_mode \
