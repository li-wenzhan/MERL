set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ! Reader checklist:
# ! 1) Set CUDA_VISIBLE_DEVICES / NUM_GPUS / WM_GPU_IDX for your machine.
# ! 2) Set libero_pro_root in configs/evaluation_config.yaml; the rollout code now auto-derives the rest of the LIBERO_PRO subpaths.
# ! 3) Set SFT_MODEL_PATH / CKPT_PATH; ALIGN_PATH / world_model_config_path / libero_pro_eval_config_path default to repo-local files.
# ! 4) Optionally set LIBERO_PRO_ROOT only if you want an environment-variable fallback instead of config-driven import.
# ! 5) MFRL does not need shared WM eval data, but it still needs the same OpenVLA and LIBERO_PRO paths.

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
export CUDA_VISIBLE_DEVICES=0,1,2
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
source "${SCRIPT_DIR}/../libero_glx_runtime.sh"
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

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM=true
# Debug-only CUDA checks; keep off for normal training unless debugging kernel failures.
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export TORCH_USE_CUDA_DSA=${TORCH_USE_CUDA_DSA:-0}

export ROBOT_PLATFORM=LIBERO  #! Use LIBERO: ROBOT_PLATFORM=LIBERO,  Use Robotwin ROBOT_PLATFORM=ALOHA

PROJECT_NAME="MFRL"
SFT_MODEL_PATH="/path/to/Openvla-oft-SFT-libero10-trajall"  # ! todo: set to your local pretrained OpenVLA checkpoint directory
CKPT_PATH="./checkpoints"  # todo: saved ckpt path, include policy_model, world_model

DATASET_NAME="libero_10"  #! change the dataset name
VLA_NAME="openvla-oft"
EXPERIMENT_DATE_TAG="0426"
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
NUM_GPUS=2  #! 总 GPU 数量 - 1; 剩 1 个给wm trainer
WM_GPU_IDX=2  #! dedicate the 3rd GPU to wm trainer
NUM_NODES=1
ALIGN_PATH="${REPO_ROOT}/configs/align.json"  # ! repo-local default; override only if your runtime env file lives elsewhere
source "${SCRIPT_DIR}/../ray_runtime.sh"
ensure_merl_ray_runtime "$REPO_ROOT" "$EXPERIMENT_NAME"

num_trials_per_task=6  #! 默认: 50, 3; 8
n_samples=4  #! 默认: 8, 需要 > micro_batch_size, 最好是; 1
action_chunks_len=8  #! 默认：8 (写死的？不要改)
#! traj_len = rollout_steps // action_chunks_len  -> 最好 % traj_mini_batch_size == 0，代码也支持自动降块
#! 其中：traj_mini_batch_size = train_batch_size * action_chunks_len（8）
#! rollout_steps(512) = N * train_batch_size * action_chunks_len**2（64）

train_batch_size=4  #! 默认: 64, 会被实际 //= NUM_GPUS (其结果需 > 0), 需要 == NUM_GPUS * N; < num_trials_per_task * 10, 应该 = log_prob_micro_batch_size / 2 * NUM_GPUS * N; 2
val_batch_size=12  #! 默认: 496, 需要 == NUM_GPUS * N; < num_trials_per_task * 10
rollout_batch_size=1
ppo_mini_batch_size=4  #! 默认：128，会被实际 //= NUM_GPUS (其结果需 > 0)，且 % log_prob_micro_batch_size = 0
micro_batch_size=1  # 默认: 1, for rollout train
val_micro_batch_size=1  #! 默认: 8, for rollout eval

log_prob_micro_batch_size=4  #! 默认: 32, 是实际上训练的 micro_batch_size，注意：会被实际 //= NUM_GPUS (其结果需 > 0)
#! 所以必须是: log_prob_micro_batch_size = N * NUM_GPUS
traj_mini_batch_size=8  #! 2 张 actor 卡上更稳的轨迹切块；64-step trajectory -> 8 x 8
enable_gradient_checkpointing=True  #! 显著降低 actor 更新峰值显存
val_only=False  #! 仅 eval; 默认: False, 先测试后训练
val_before_train=True
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

train_mode="MFRL"
world_model_enable=False
save_freq=10
save_freq_wm_outer=10
save_freq_wm_inner=1
wm_inner_steps=100
wm_eval_interval=5
test_freq=5
wm_fine_tune=False
strict_mode_assert=True
replay_pool_capacity=128
video_save_interval=5
persist_replay_pool=False
replay_pool_keep_last=1
replay_pool_max_gb=15

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
if [ "$visible_gpu_count" -lt $((NUM_GPUS + 1)) ]; then
    echo "[preflight] CUDA_VISIBLE_DEVICES must expose NUM_GPUS actor GPUs plus one WM GPU" >&2
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

require_equal "$PROJECT_NAME" "$train_mode" "PROJECT_NAME/train_mode mismatch"
require_contains "$EXPERIMENT_NAME" "$train_mode" "EXPERIMENT_NAME/train_mode mismatch"
require_equal "$world_model_enable" "False" "MFRL requires world_model.enable=False"
require_equal "$wm_fine_tune" "False" "MFRL requires wm_fine_tune=False"

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
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=$traj_mini_batch_size \
    actor_rollout_ref.model.enable_gradient_checkpointing=$enable_gradient_checkpointing \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=$val_micro_batch_size \
    actor_rollout_ref.rollout.temperature=1.6 \
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
    ++actor_rollout_ref.rollout.save_training_videos=True \
    ++actor_rollout_ref.rollout.video_save_interval=$video_save_interval \
    ++actor_rollout_ref.rollout.max_env_videos_per_task=1 \
    ++actor_rollout_ref.rollout.max_wm_videos_per_task=1 \
    ++actor_rollout_ref.rollout.use_libero_pro=$use_libero_pro \
    ++actor_rollout_ref.rollout.env_service_enable=${MERL_LIBERO_ENV_SERVICE_ENABLE:-true} \
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
    ++actor_rollout_ref.world_model.wm_eval_interval=$wm_eval_interval \
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
    trainer.val_before_train=$val_before_train \
    ++trainer.strict_mode_assert=$strict_mode_assert \
    ++trainer.train_mode=$train_mode \
