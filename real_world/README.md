# Real-World MERL Branch

This branch is intentionally independent from Ray, LIBERO, and DataProto.

Stage 1 focuses on an offline real-world world-model interface:

1. Load a real camera video or image sequence.
2. Load a robot action sequence and a task instruction.
3. Run Ctrl-World once on a single GPU.
4. Save predicted future frames/video and per-frame reward scores.
5. Emit a structured `result.json` that can later be consumed by MERL replay or diagnostics.

The first runnable entrypoint is:

```bash
bash examples/real_world_wm_predict.sh
```

## Real-World HDF5 Fine-Tuning

Use this path when real robot data is stored as either:

- one directory containing multiple `.hdf5` / `.h5` files
- one merged `.h5` file containing multiple episode groups

The fine-tuning dataset reads only the top-view camera by default:

```text
observations/images/cam_high
```

and reads robot actions from:

```text
action
```

For bimanual 14D actions, the default `ACTION_SLICE=right` selects `action[:, 7:14]`.
The reward target is synthetic because real data has no reward: each training window has
reward shape `[T]`, with the last frame set to `1` and all earlier frames set to `0`.

Run a real-data fine-tune:

```bash
REAL_DATA_ROOT=/path/to/real_hdf5_or_folder TASK="Put the red block into the box." OUTPUT_DIR=./outputs/real_world_wm_finetune/red_block_box SVD_MODEL_PATH=/path/to/models/stable-video-diffusion-img2vid CLIP_MODEL_PATH=/path/to/models/clip-vit-base-patch32 CTRL_WORLD_CKPT=/path/to/Ctrl_World/checkpoint.pt CUDA_VISIBLE_DEVICES=0 bash real_world/train_real_world_wm_hdf5.sh
```

Useful overrides:

```text
FRAME_STRIDE=1 ACTION_STRIDE=1 FPS=30
NUM_HISTORY=8 NUM_FUTURE=8
BATCH_SIZE=1 MAX_TRAIN_STEPS=5000 CHECKPOINTING_STEPS=1000 MAX_KEEP_CKPTS=2
```

The script writes checkpoints under `OUTPUT_DIR` and also writes:

```text
OUTPUT_DIR/real_wm_infer_finetuned.yaml
```

Checkpoint policy:

- `checkpoint-N.pt` is a pure model `state_dict` and can be used directly by inference.
- `checkpoint-N.train_state.pt` is the sidecar training state for the same step and stores optimizer/global_step.
- By default the trainer saves every `1000` steps and keeps only the latest `2` numbered checkpoints.
- If `SAVE_FINAL=1`, an extra `checkpoint-final.pt` is written and is not counted by the numbered-checkpoint rotation.

Resume from any saved step:

```bash
REAL_DATA_ROOT=/path/to/real_hdf5_or_folder OUTPUT_DIR=./outputs/real_world_wm_finetune/red_block_box RESUME_CKPT=./outputs/real_world_wm_finetune/red_block_box/checkpoint-2000.pt MAX_TRAIN_STEPS=5000 CUDA_VISIBLE_DEVICES=0 bash real_world/train_real_world_wm_hdf5.sh
```

Passing `checkpoint-2000.pt` will automatically use `checkpoint-2000.train_state.pt` if it exists.
You can also pass the sidecar directly:

```bash
REAL_DATA_ROOT=/path/to/real_hdf5_or_folder OUTPUT_DIR=./outputs/real_world_wm_finetune/red_block_box RESUME_CKPT=./outputs/real_world_wm_finetune/red_block_box/checkpoint-2000.train_state.pt MAX_TRAIN_STEPS=5000 CUDA_VISIBLE_DEVICES=0 bash real_world/train_real_world_wm_hdf5.sh
```

`MAX_TRAIN_STEPS` is the absolute target global step. For example, if resuming from
`checkpoint-2000.pt` and you want 3000 more steps, set `MAX_TRAIN_STEPS=5000`.

Use that config directly with the existing real-world inference wrapper:

```bash
CONFIG=./outputs/real_world_wm_finetune/red_block_box/real_wm_infer_finetuned.yaml HDF5=/path/to/test_episode.hdf5 TASK="Put the red block into the box." bash examples/real_world_wm_predict_hdf5.sh
```
