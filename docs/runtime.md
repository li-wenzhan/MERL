# Runtime setup

Use Linux with Python 3.10, a CUDA-matched PyTorch/torchvision installation, NCCL, EGL and ffmpeg. `configs/runtime_versions.json` lists package versions. Install the remaining dependencies with `python -m pip install -r requirements.txt`.

## Local assets

Prepare assets on a network-enabled machine before submitting offline compute jobs:

- A categorical OpenVLA-OFT initialization with tokenizer, processor and action statistics, as described in [training](training.md).
- SVD and CLIP model directories, selected by `MERL_SVD_MODEL_PATH` and `MERL_CLIP_MODEL_PATH`.
- ImageNet ResNet-18 weights in the Torch Hub cache selected by `TORCH_HOME`, used when initializing a new progress proxy.
- The LIBERO-PRO checkout selected by `LIBERO_PRO_ROOT`, including robot assets, BDDL task files and matching `.pruned_init` states.

The standard launcher loads local model assets offline. Full simulator checkpoints include the progress backbone. Make the backbone directories, cache and datasets accessible on every compute node that runs the job.

`configs/pretraining_config.yaml` uses original environments. `configs/evaluation_config.yaml` enables the environment-shift panel. Select one perturbation at a time. Follow [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) to prepare the selected perturbation assets; environment shifts use suite names such as `libero_10_env`. Reuse the same generated assets and state files across methods.

## GPU layout and rendering

Model-based training allocates actor ranks to the first three visible GPUs and a single shared simulator to the fourth. MFRL uses the first three. Evaluation and trajectory collection can run on one GPU; FSDP actor checkpoint evaluation preserves the checkpoint's original rank count.

The launcher preserves scheduler-provided `CUDA_VISIBLE_DEVICES` and network interface variables. Headless rendering uses EGL and subprocess environment services by default. Set `MERL_LIBERO_ENV_BACKEND` and `MERL_LIBERO_EGL_DEVICE_ID` to select a rendering backend/device. `scripts/runtime/libero_glx_runtime.sh` provides an explicit GLX setup.

Validate asset layout and resolve all Hydra settings before allocating a training run:

```bash
python -m merl.launch --mode MERL --experiment assets_check \
  --vla-init "$VLA_INIT" --unnorm-key "$UNNORM_KEY" --wm-checkpoint "$WM_INIT" --check
```

`--render-check` also resets and steps an environment. `--dry-run` prints the command without model, asset or GPU access.

## Logs and run directories

```bash
ACP_LOG_DIR=/outputs/logs bash scripts/run_logged.sh python -m merl.launch \
  --mode MERL --experiment run1 --vla-init "$VLA_INIT" --wm-checkpoint "$WM_INIT"
```

The wrapper streams stdout/stderr and saves a UTC-named log and `.status.json` with command/logging exit codes and elapsed seconds. The default location is `tmp_files/acp_logs`. The launch manifest records its console-log path. Run-local `run.log` captures the trainer and `ray_logs/` preserves worker diagnostics.

An experiment name creates a fresh output directory. A failed run remains available for inspection; choose a new name for the next attempt. Launch manifests record the command, resolved settings, source hashes, model metadata, package versions, visible devices and final status. Compute jobs perform no source synchronization or Git operations.
