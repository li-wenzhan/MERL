SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_DATASET_ROOT=${LIBERO_DATASET_ROOT:-/path/to/regen_dataset_no_noops}  # ! todo: set to your regenerated LIBERO dataset root
VAL_MODEL_PATH=${VAL_MODEL_PATH:-/path/to/Ctrl_World/checkpoint.pt}  # ! todo: point to the Ctrl-World checkpoint you want to validate

# CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
#     accelerate launch \
#     --num_processes 1 \
#     --main_process_port 29501 \
#     scripts/train_wm_new.py \
#     --dataset droid \
#     --dataset_root_path /path/to/droid/dataset_example \
#     --dataset_meta_info_path /path/to/droid/dataset_meta_info \
#     --dataset_names droid_subset

# /path/to/libero_regen/dataset_noop

# 39G显存
# CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
#     accelerate launch \
#     --num_processes 1 \
#     --main_process_port 29501 \
#     scripts/train_wm_new.py \
#     --dataset libero \
#     --dataset_root_path /path/to/regen_dataset_no_noops \
#     --dataset_names libero_goal libero_object libero_spatial libero_10

CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
    accelerate launch \
    --num_processes 1 \
    --main_process_port 29501 \
    "$SCRIPT_DIR/scripts/validate_new.py" \
    --val_model_path "$VAL_MODEL_PATH" \
    --dataset libero \
    --dataset_root_path "$LIBERO_DATASET_ROOT" \
    --num_steps 20 \
    --dirname samples_val_1222_e120000
