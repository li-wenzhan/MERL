DATASET_ROOT_PATH=${DATASET_ROOT_PATH:-/path/to/regen_dataset_no_noops}  # ! todo: set to your regenerated LIBERO dataset root

python ./dataset/libero/prepare_data_samples_new.py \
    --dataset-root-path "$DATASET_ROOT_PATH" \
    --dataset-names "libero_goal" "libero_object" "libero_spatial" "libero_10" \
    --save-path "./dataset/libero" \
    --num-history 8 \
    --num-frames 8 \
    --window-sample 8 \
    --down-sample 2 \
    --positive-ratio 0.5 \
    --upsample-positive

# --window-sample 窗口采样间隔
# --down-sample 帧采样间距
