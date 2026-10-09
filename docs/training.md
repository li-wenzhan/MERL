# Training MERL

## VLA initialization

Prepare the OpenVLA-OFT environment using its [setup guide](https://github.com/moojink/openvla-oft/blob/main/SETUP.md) and download the [LIBERO RLDS demonstrations](https://huggingface.co/datasets/openvla/modified_libero_rlds). In that checkout, run categorical action-chunk fine-tuning:

```bash
export ROBOT_PLATFORM=LIBERO
export WANDB_MODE=disabled
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path /models/openvla-7b \
  --data_root_dir /data/modified_libero_rlds \
  --dataset_name libero_10_no_noops --run_root_dir /outputs/vla_init \
  --use_l1_regression False --use_diffusion False --use_film False \
  --num_images_in_input 1 --use_proprio False \
  --batch_size 4 --learning_rate 5e-4 --lora_rank 32 \
  --max_steps 50000 --num_steps_before_decay 40000 --save_freq 5000 \
  --image_aug True --save_latest_checkpoint_only False
```

The [upstream trainer](https://github.com/moojink/openvla-oft/blob/main/vla-scripts/finetune.py) uses discrete next-token prediction when both continuous-head flags are disabled. Select LIBERO constants with chunk length 8 and action dimension 7. Use the upstream LoRA merge/export utilities to produce a complete model directory. Copy the training run's action statistics alongside the exported model.

The initialization directory contains:

```text
config.json
model.safetensors.index.json
model-*.safetensors
tokenizer.json
tokenizer_config.json
preprocessor_config.json
dataset_statistics.json
```

Set `VLA_INIT` to this directory and `UNNORM_KEY` to the suite's key in `dataset_statistics.json`. `--vla-init` is the policy initialization option; `--sft-checkpoint` is an equivalent alias. MERL creates run-local actor assets and preserves the source directory.

## Simulator initialization

Use `--job collect --mode MFRL --split wm_train --collection-dir /data/wm_init` to collect aligned NPZ trajectories. Select `configs/pretraining_config.yaml` for original environments. Six trials per task collect a 60-trajectory panel on LIBERO-10; use `--collection-trial-offset` to choose additional training state IDs. Keep those IDs disjoint from the evaluation panel.

The trajectory writer stores RGB `observations[T+1]`, executed `actions[T,7]`, instruction, outcome, task/trial IDs and provenance. The proxy target is discounted, horizon-truncated success-to-go. A failed trajectory has zero targets; a successful trajectory supplies positive supervision near its completion. Include both outcomes when preparing the simulator dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python -m merl.train_simulator \
  --data /data/wm_init/wm_train/trajectories --output /outputs/wm_init/run1 \
  --steps 5000 --learning-rate 1e-5 --save-every 500 --checkpoint-keep 2 --seed 0
```

This initializes the action-conditioned video predictor from SVD and the progress proxy from ImageNet/CLIP backbones. Frozen encoders stay in evaluation mode; trainable weights and optimizer storage use FP32 with BF16 computation. The loss combines visual denoising and masked progress classification. Each optimizer step samples one trajectory window and includes a valid partial final chunk when necessary.

To warm-start model weights, add `--from-checkpoint /models/simulator.pt`. To restore weights, optimizer and RNG into a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m merl.train_simulator \
  --data /data/wm_init/wm_train/trajectories --output /outputs/wm_init/run1_resumed \
  --resume-from /outputs/wm_init/run1/checkpoint-5000.train_state.pt \
  --steps 10000 --seed 0
```

`--steps` is the absolute target step. Dataset hashes, configuration and learning rate must match the resumed run. `latest.json` points to the most recent complete snapshot. `checkpoint-N.pt` loads directly through `merl.launch --wm-checkpoint`; the matching `.train_state.pt` restores offline training.

## Policy refinement and controls

The driver in `merl/trainer.py` performs grounded collection, optional simulator adaptation, stored calibration, recursive imagination and one mixed policy update per stage. `configs/merl.json` contains the algorithm parameters. `configs/launch_profiles.json` contains shared model, batch and Ray/FSDP settings.

| Mode | Update simulator | Schedule ratio/horizon | Trust replay |
| :--- | :---: | :---: | :---: |
| MFRL | — | — | — |
| MBRL | No | Fixed | No |
| STATIC_TRUST | No | Yes | Yes |
| ONLINE_MBRL | Yes | Fixed | No |
| MERL | Yes | Yes | Yes |

Run each control from the same initial VLA and simulator, with matching environment assets and seeds. MFRL uses three actor GPUs; model-based modes use the same actors and one simulator GPU. Use `--config` for a separate algorithm configuration or Hydra overrides after `--`, such as `merl.chunk_trust=false` or `merl.stage_trust=false`.

## Checkpoints and resume

```text
checkpoints/MERL/RUN/
├── actor/global_step_N/           # FSDP weights and actor runtime
├── world_model/global_step_N/     # Simulator weights, optimizer and residual predictor
├── training_state/               # Completed-stage snapshots, update metrics and trust decisions
├── grounded/stage_N/              # Aligned current-policy trajectories
├── imagination/                   # Recursive imagined trajectories
├── evaluation/                    # Real-environment episode artifacts
├── launch_manifest.json
├── run.log
└── ray_logs/
```

A completed-stage snapshot references both model directories and restores optimizer, RNG, scheduler, residual predictor and replay sampling state. `--resume-from` accepts the saved stage file, including files from earlier runs; it writes subsequent stages to a new experiment directory. Preserve the referenced model directories. Resume with the same mode, seed, initialization and actor rank count. `--checkpoint-keep 0` retains every checkpoint created by the new run; the default keeps the latest two.

Evaluation uses `--job evaluate --actor-checkpoint <actor directory>`. It executes the policy in the environment and uses environment success. The initialized policy is evaluated by omitting the actor checkpoint.
