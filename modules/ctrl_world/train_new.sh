# ! Reader checklist:
# ! 1) Set CUDA_VISIBLE_DEVICES to the GPU you reserve for offline Ctrl-World pretraining.
# ! 2) Set --dataset_root_path to your local regenerated LIBERO dataset root.
# ! 3) Keep --dataset_names aligned with the datasets you actually prepared.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_DATASET_ROOT=${LIBERO_DATASET_ROOT:-/path/to/regen_dataset_no_noops}  # ! todo: set to your regenerated LIBERO dataset root

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
# ! todo: change your CUDA device for offline Ctrl-World pretraining
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
    accelerate launch \
    --num_processes 1 \
    --main_process_port 29501 \
    "$SCRIPT_DIR/scripts/train_new.py" \
    --dataset libero \
    --dataset_root_path "$LIBERO_DATASET_ROOT" \
    --dataset_names libero_goal libero_object libero_spatial libero_10
