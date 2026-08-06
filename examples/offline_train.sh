# ! Legacy convenience wrapper.
# ! Prefer modules/ctrl_world/train_new.sh for current offline Ctrl-World pretraining.
# ! If you still use this file, update CUDA_VISIBLE_DEVICES and --dataset_root_path first.

LIBERO_DATASET_ROOT=${LIBERO_DATASET_ROOT:-/mnt/afs/L202500276/data/LIBERO/regen_dataset_no_noops}  # ! todo: set to your regenerated LIBERO dataset root

CUDA_VISIBLE_DEVICES=3 WANDB_MODE=offline \
    accelerate launch \
    --num_processes 1 \
    --main_process_port 29501 \
    modules/ctrl_world/scripts/train_new.py \
    --dataset libero \
    --dataset_root_path "$LIBERO_DATASET_ROOT" \
    --dataset_names libero_goal libero_object libero_spatial libero_10
