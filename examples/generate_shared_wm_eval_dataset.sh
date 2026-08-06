set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ! Reader checklist:
# ! 1) Set CUDA_VISIBLE_DEVICES to a GPU that can run one rollout worker.
# ! 2) Set SFT_MODEL_PATH / WORLD_MODEL_CONFIG_PATH / LIBERO_PRO_EVAL_CONFIG_PATH / ALIGN_PATH.
# ! 3) Keep DATASET_NAME and SHARED_WM_EVAL_ROOT consistent with MERL / MBRL launchers.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # ! todo: choose one visible GPU for shared-eval generation
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=true
export ROBOT_PLATFORM=LIBERO  # ! keep LIBERO when generating the shared WM eval dataset
source "${SCRIPT_DIR}/libero_glx_runtime.sh"
ensure_merl_glx_runtime

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version


def _version_tuple(raw):
    parts = []
    for item in raw.split("."):
        digits = "".join(ch for ch in item if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple((parts + [0, 0, 0])[:3])


try:
    hub_version = version("huggingface-hub")
except PackageNotFoundError as exc:
    raise SystemExit(
        "[preflight] missing huggingface-hub. Run: "
        "pip install 'huggingface-hub>=0.34.0,<1.0'"
    ) from exc

if not ((0, 34, 0) <= _version_tuple(hub_version) < (1, 0, 0)):
    raise SystemExit(
        "[preflight] incompatible huggingface-hub=="
        f"{hub_version}; transformers requires >=0.34.0,<1.0. "
        "Run: pip install --force-reinstall 'huggingface-hub>=0.34.0,<1.0'"
    )
PY

SFT_MODEL_PATH=${SFT_MODEL_PATH:-/path/to/Openvla-oft-SFT-libero10-trajall}  # ! todo: set to your local OpenVLA SFT checkpoint directory
WORLD_MODEL_CONFIG_PATH=${WORLD_MODEL_CONFIG_PATH:-$REPO_ROOT/configs/wm_online_config.py}  # ! repo-local default; override only if you use a custom wm_online_config.py
LIBERO_PRO_EVAL_CONFIG_PATH=${LIBERO_PRO_EVAL_CONFIG_PATH:-$REPO_ROOT/configs/evaluation_config.yaml}  # ! repo-local default; override only if you use a custom evaluation_config.yaml

DATASET_NAME=${DATASET_NAME:-libero_10}
VLA_NAME=${VLA_NAME:-openvla-oft}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-./tmp_files/actor_assets/shared_wm_eval_${DATASET_NAME}}
NUM_GPUS=${NUM_GPUS:-1}
NUM_NODES=${NUM_NODES:-1}
ALIGN_PATH=${ALIGN_PATH:-$REPO_ROOT/configs/align.json}  # ! repo-local default; override only if your runtime env file lives elsewhere
SHARED_WM_EVAL_ROOT=${SHARED_WM_EVAL_ROOT:-./tmp_files/wm_eval_shared/${DATASET_NAME}}  # ! todo: reuse this same root in MERL / MBRL scripts

export SFT_MODEL_PATH
export ACTOR_MODEL_PATH
export WORLD_MODEL_CONFIG_PATH
export LIBERO_PRO_EVAL_CONFIG_PATH
export ALIGN_PATH
export MERL_ENV_MP_START_METHOD

python - <<'PY'
import os

import yaml


def require_path(label: str, path: str, *, is_dir: bool = False) -> None:
    normalized = os.path.abspath(os.path.expanduser(path))
    exists = os.path.isdir(normalized) if is_dir else os.path.exists(normalized)
    if not exists:
        raise SystemExit(f"[preflight] missing {label}: {normalized}")


require_path("SFT_MODEL_PATH", os.environ["SFT_MODEL_PATH"], is_dir=True)
require_path("WORLD_MODEL_CONFIG_PATH", os.environ["WORLD_MODEL_CONFIG_PATH"])
require_path("LIBERO_PRO_EVAL_CONFIG_PATH", os.environ["LIBERO_PRO_EVAL_CONFIG_PATH"])
require_path("ALIGN_PATH", os.environ["ALIGN_PATH"])

with open(os.environ["LIBERO_PRO_EVAL_CONFIG_PATH"], "r", encoding="utf-8") as file_obj:
    evaluation_cfg = yaml.safe_load(file_obj) or {}

libero_pro_root = str(evaluation_cfg.get("libero_pro_root") or "").strip()
if not libero_pro_root or libero_pro_root.startswith("/path/to/"):
    raise SystemExit(
        "[preflight] configs/evaluation_config.yaml:libero_pro_root is not configured. "
        "Set it to your local LIBERO_PRO clone root."
    )

libero_pro_root = os.path.abspath(os.path.expanduser(libero_pro_root))
require_path("LIBERO_PRO root", libero_pro_root, is_dir=True)
require_path(
    "LIBERO_PRO benchmark root",
    os.path.join(libero_pro_root, "libero", "libero"),
    is_dir=True,
)
require_path(
    "LIBERO_PRO bddl_files",
    os.path.join(libero_pro_root, "libero", "libero", "bddl_files"),
    is_dir=True,
)
require_path(
    "LIBERO_PRO generate_init_states.py",
    os.path.join(libero_pro_root, "notebooks", "generate_init_states.py"),
)

print("[preflight] shared eval paths OK")
PY

python "$REPO_ROOT/scripts/preflight_libero_pro_compat.py" \
    --config "$LIBERO_PRO_EVAL_CONFIG_PATH"

bash "$REPO_ROOT/examples/overwrite_vla_ckpt_utils.sh" "$SFT_MODEL_PATH" "$ACTOR_MODEL_PATH"

run_shared_eval_generation() {
    local split_name="$1"
    local num_trials_per_task="$2"
    local experiment_name="$3"

    HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
        data.task_suite_name=$DATASET_NAME \
        data.num_trials_per_task=$num_trials_per_task \
        data.n_samples=1 \
        data.train_batch_size=1 \
        data.val_batch_size=1 \
        +data.rollout_batch_size=1 \
        data.max_prompt_length=256 \
        data.max_response_length=128 \
        actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
        actor_rollout_ref.model.vla=$VLA_NAME \
        actor_rollout_ref.model.action_token_len=7 \
        actor_rollout_ref.model.action_chunks_len=8 \
        actor_rollout_ref.actor.optim.lr=5e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=1 \
        actor_rollout_ref.actor.ppo_micro_batch_size=1 \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.grad_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.num_images_in_input=1 \
        actor_rollout_ref.rollout.num_images_in_input=1 \
        actor_rollout_ref.rollout.use_proprio=False \
        actor_rollout_ref.rollout.val_micro_batch_size=1 \
        actor_rollout_ref.rollout.temperature=0.0 \
        actor_rollout_ref.rollout.experiment_name=$experiment_name \
        actor_rollout_ref.rollout.micro_batch_size=1 \
        actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
        actor_rollout_ref.rollout.model_family=openvla \
        actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
        actor_rollout_ref.rollout.num_steps_wait=10 \
        actor_rollout_ref.rollout.pretrained_checkpoint=$ACTOR_MODEL_PATH \
        actor_rollout_ref.rollout.center_crop=True \
        actor_rollout_ref.rollout.max_prompt_length=512 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=hf \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        ++actor_rollout_ref.rollout.use_libero_pro=True \
        ++actor_rollout_ref.rollout.libero_pro_eval_config_path=$LIBERO_PRO_EVAL_CONFIG_PATH \
        actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
        ++actor_rollout_ref.ref.vla=$VLA_NAME \
        ++actor_rollout_ref.ref.action_token_len=7 \
        ++actor_rollout_ref.ref.action_chunks_len=8 \
        ++actor_rollout_ref.ref.use_proprio=False \
        ++actor_rollout_ref.ref.use_remove_padding=False \
        ++actor_rollout_ref.world_model.enable=False \
        ++actor_rollout_ref.world_model.fine_tune=False \
        ++actor_rollout_ref.world_model.config_path=$WORLD_MODEL_CONFIG_PATH \
        ++actor_rollout_ref.rollout_base_dir=$SHARED_WM_EVAL_ROOT \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['console'] \
        trainer.wandb_mode=disabled \
        trainer.project_name=WM_EVAL_DATA \
        trainer.experiment_name=$experiment_name \
        trainer.default_local_dir=./checkpoints/WM_EVAL_DATA/$experiment_name \
        trainer.n_gpus_per_node=$NUM_GPUS \
        trainer.nnodes=$NUM_NODES \
        trainer.save_freq=1000 \
        trainer.test_freq=1000 \
        trainer.total_epochs=1 \
        trainer.val_only=False \
        algorithm.adv_estimator=grpo \
        algorithm.adv_params.verifier_gamma=1.0 \
        algorithm.adv_params.reward_model_gamma=1.0 \
        trainer.runtime_env=$ALIGN_PATH \
        trainer.val_before_train=False \
        ++trainer.strict_mode_assert=False \
        ++trainer.train_mode=MFRL \
        ++trainer.rollout_before_train=True \
        ++trainer.sim_rollout_epoch=1 \
        ++trainer.preserve_rollout_base_dir=True \
        ++trainer.rollout_train_split=$split_name \
        ++trainer.rollout_save_eval=False \
        ++trainer.rollout_save_to_hdfs=True \
        ++trainer.rollout_do_sample=False
}

run_shared_eval_generation wm_eval_fixed_mini 2 shared_wm_eval_mini_${DATASET_NAME}
run_shared_eval_generation wm_eval_fixed_full 8 shared_wm_eval_full_${DATASET_NAME}
