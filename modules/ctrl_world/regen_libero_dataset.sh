RAW_LIBERO_DATA_DIR=${RAW_LIBERO_DATA_DIR:-/path/to/LIBERO}  # ! todo: set to your raw LIBERO dataset root
REGEN_LIBERO_SAVE_DIR=${REGEN_LIBERO_SAVE_DIR:-/path/to/regen_dataset}  # ! todo: set to your regenerated LIBERO output directory

python dataset/libero/gen_libero_dataset_noop.py \
    --raw_data_dir "$RAW_LIBERO_DATA_DIR" \
    --save_dir "$REGEN_LIBERO_SAVE_DIR" \
    --libero_task_suites libero_spatial libero_object libero_goal libero_10 \
    --filter_static_actions
