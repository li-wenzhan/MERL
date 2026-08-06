set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# For single-machine multi-GPU mode
export CUDA_VISIBLE_DEVICES=0,1,2
export MERL_GLOBAL_CUDA_VISIBLE_DEVICES="${MERL_GLOBAL_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"

# For NCCL debug
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [ -z "${NCCL_DEBUG_SUBSYS:-}" ]; then
    unset NCCL_DEBUG_SUBSYS
else
    export NCCL_DEBUG_SUBSYS
fi
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# For headless GLX rendering
source "${SCRIPT_DIR}/libero_glx_runtime.sh"
ensure_merl_glx_runtime
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64

if [ -n "${LIBERO_PRO_ROOT:-}" ]; then
    export PYTHONPATH="${LIBERO_PRO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
fi

# export NCCL_DEBUG=WARN
export WANDB_MODE=disabled
export WANDB_DISABLED=true
# export WANDB_API_KEY="86afa5b168a8fbf4a1dfd98145f0c5d133ca103b"  #! wandb api key
# export WANDB_MODE=offline

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export ROBOT_PLATFORM=LIBERO  #! Use LIBERO: ROBOT_PLATFORM=LIBERO,  Use Robotwin ROBOT_PLATFORM=ALOHA

PROJECT_NAME="MeRL"
EXPERIMENT_NAME="train_Openvla-oft-SFT-libero_10-debug-3gpu-512x-loss_w-pro_0414_eval"  #! change the date
SFT_MODEL_PATH="/path/to/actor_checkpoint"  # ! todo: point to the actor checkpoint you want to evaluate
CKPT_PATH="./checkpoints"  # todo: saved ckpt path, include policy_model, world_model
EXPERIMENT_OUTPUT_DIR="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$EXPERIMENT_OUTPUT_DIR/actor_assets}"

DATASET_NAME="libero_10"  #! change the dataset name
VLA_NAME="openvla-oft"
NUM_GPUS=2  #! 总 GPU 数量 - 1; 剩 1 个给wm trainer
WM_GPU_IDX=0  #! set 0 as the wm trainer rank
NUM_NODES=1
ALIGN_PATH="${REPO_ROOT}/configs/align.json"

num_trials_per_task=8  #! 默认: 50, 3; 8
n_samples=2  #! 默认: 8, 需要 > micro_batch_size, 最好是; 1
action_chunks_len=8  #! 默认：8 (写死的？不要改)
#! traj_len = rollout_steps // action_chunks_len  -> traj_len 必须 % traj_mini_batch_size: == 0
#! 其中：traj_mini_batch_size = train_batch_size * action_chunks_len（8）
#! rollout_steps(512) = N * train_batch_size * action_chunks_len**2（64）

train_batch_size=4  #! 默认: 64, 会被实际 //= NUM_GPUS (其结果需 > 0), 需要 == NUM_GPUS * N; < num_trials_per_task * 10, 应该 = log_prob_micro_batch_size / 2 * NUM_GPUS * N; 2
val_batch_size=4  #! 默认: 496, 需要 == NUM_GPUS * N; < num_trials_per_task * 10
rollout_batch_size=1
ppo_mini_batch_size=2  #! 默认：128，会被实际 //= NUM_GPUS (其结果需 > 0)，且 % log_prob_micro_batch_size = 0
micro_batch_size=1  # 默认: 1, for rollout train
val_micro_batch_size=1  #! 默认: 8, for rollout eval

log_prob_micro_batch_size=2  #! 默认: 32, 是实际上训练的 micro_batch_size，注意：会被实际 //= NUM_GPUS (其结果需 > 0)
#! 所以必须是: log_prob_micro_batch_size = N * NUM_GPUS
val_only=True  #! 仅 eval; 默认: False, 先测试后训练
val_before_train=True  #! 默认: True
wandb_mode="disabled"  #! 默认: online
# 原实现（带wandb）: trainer.logger=['console','wandb'] \
# wandb_mode=online

# actor_rollout_ref.actor.fsdp_config.grad_offload=True
# actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
# actor_rollout_ref.ref.fsdp_config.param_offload=True

#! added below
world_model_config_path="${REPO_ROOT}/configs/wm_online_config.py"  # ! repo-local default; override only if you use a custom wm_online_config.py
use_libero_pro=True
libero_pro_eval_config_path="${REPO_ROOT}/configs/evaluation_config.yaml"  # ! repo-local default; override only if you use a custom evaluation_config.yaml

train_mode="MERL"
save_freq=100
save_freq_wm_outer=100
save_freq_wm_inner=1
wm_eval_interval=1
wm_fine_tune=True

bash "$REPO_ROOT/examples/overwrite_vla_ckpt_utils.sh" "$SFT_MODEL_PATH" "$ACTOR_MODEL_PATH"

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
    actor_rollout_ref.actor.traj_mini_batch_size=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
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
    +actor_rollout_ref.rollout.use_libero_pro=$use_libero_pro \
    +actor_rollout_ref.rollout.libero_pro_eval_config_path=$libero_pro_eval_config_path \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    +actor_rollout_ref.world_model.enable=True \
    +actor_rollout_ref.world_model.config_path=$world_model_config_path \
    +actor_rollout_ref.world_model.fine_tune=$wm_fine_tune \
    +actor_rollout_ref.world_model.training_steps_per_epoch=5000 \
    +actor_rollout_ref.world_model.reward_model.reward_thr=0.5 \
    +actor_rollout_ref.world_model.save_freq_wm_outer=$save_freq_wm_outer \
    +actor_rollout_ref.world_model.save_freq_wm_inner=$save_freq_wm_inner \
    +actor_rollout_ref.world_model.lr=1e-5 \
    +actor_rollout_ref.world_model.batch_size=4 \
    +actor_rollout_ref.world_model.num_workers=2 \
    +actor_rollout_ref.world_model.wm_eval_interval=$wm_eval_interval \
    +actor_rollout_ref.rollout_base_dir=./tmp_files/rollout/$EXPERIMENT_NAME \
    +actor_rollout_ref.wm_gpu_idx=$WM_GPU_IDX \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.wandb_mode=$wandb_mode \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$EXPERIMENT_OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=$save_freq \
    trainer.test_freq=4 \
    trainer.total_epochs=100 \
    trainer.val_only=$val_only \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.val_before_train=$val_before_train \
    +trainer.train_mode=$train_mode \
