set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ! Legacy convenience wrapper.
# ! Prefer examples/train_merl_debug_3gpu_fix_pro_.sh for the maintained MERL launcher.
# ! If you still use this file, update CUDA_VISIBLE_DEVICES, SFT_MODEL_PATH, CKPT_PATH, and world_model_config_path first.

export CUDA_VISIBLE_DEVICES=6,7
export MERL_GLOBAL_CUDA_VISIBLE_DEVICES="${MERL_GLOBAL_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
export MUJOCO_GL=glx
export PYOPENGL_PLATFORM=glx
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64

export NCCL_DEBUG=WARN
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
EXPERIMENT_NAME="train_Openvla-oft-SFT-libero10-debug-120x"  #! change the date
SFT_MODEL_PATH="/path/to/Openvla-oft-SFT-libero10-trajall"  # ! todo: set to your local pretrained OpenVLA checkpoint directory
CKPT_PATH="./checkpoints"  #! saved ckpt path, include policy_model, world_model
EXPERIMENT_OUTPUT_DIR="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$EXPERIMENT_OUTPUT_DIR/actor_assets}"

DATASET_NAME="libero_10"  #! change the dataset name
VLA_NAME="openvla-oft"
NUM_GPUS=2  #! GPU 数量
NUM_NODES=1
ALIGN_PATH="${REPO_ROOT}/configs/align.json"

num_trials_per_task=2  #! 默认: 50
n_samples=2  #! 默认: 8, 需要 > micro_batch_size
action_chunks_len=8  #! 默认：8 (写死的？)
#! traj_len = rollout_steps // action_chunks_len  -> traj_len 必须 % traj_mini_batch_size: == 0
#! 其中：traj_mini_batch_size = train_batch_size * action_chunks_len（8）
#! rollout_steps(512) = N * train_batch_size * action_chunks_len**2（64）
train_batch_size=4  #! 默认: 64, 需要 == NUM_GPUS * N; < num_trials_per_task * 10
val_batch_size=4  #! 默认: 496, 需要 == NUM_GPUS * N; < num_trials_per_task * 10
rollout_batch_size=128
micro_batch_size=1  # 默认: 1, for rollout train
val_micro_batch_size=4  #! 默认: 8, for rollout eval
log_prob_micro_batch_size=4  #! 默认: 32, 注意：会被实际 //= NUM_GPUS (结果需 > 0)
#! 所以必须是: log_prob_micro_batch_size = N * NUM_GPUS
val_only=False  #! 仅 eval; 默认: False, 先测试后训练
val_before_train=False  #! 默认: True
wandb_mode="disabled"  #! 默认: online
# 原实现（带wandb）: trainer.logger=['console','wandb'] \
# wandb_mode=online
# actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

#! added below
world_model_config_path="${REPO_ROOT}/configs/wm_online_config.py"


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
    actor_rollout_ref.model.action_chunks_len=${action_chunks_len} \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.world_model.enable=True \
    +actor_rollout_ref.world_model.config_path=$world_model_config_path \
    +actor_rollout_ref.world_model.fine_tune=True \
    +actor_rollout_ref.world_model.training_steps_per_epoch=10000 \
    +actor_rollout_ref.world_model.reward_model.reward_thr=0.8 \
    +actor_rollout_ref.world_model.lr=1e-5 \
    +actor_rollout_ref.world_model.batch_size=4 \
    +actor_rollout_ref.world_model.num_workers=2 \
    +actor_rollout_ref.rollout_base_dir=./tmp_files/rollout/$EXPERIMENT_NAME \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.wandb_mode=$wandb_mode \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$EXPERIMENT_OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=25 \
    trainer.test_freq=4 \
    trainer.total_epochs=100 \
    trainer.val_only=$val_only \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.val_before_train=$val_before_train \
