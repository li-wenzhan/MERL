# MERL: World Model-Evolving Reinforcement Learning with Trust-Calibrated Imagination

## Implementation status and maintained mechanisms

The production training path currently uses a grounded mirror for WM rollouts;
it is not yet the recursive no-oracle path described in the rebuttal. See the
[research-to-code audit](docs/implementation_audit.md) before interpreting results.
The isolated [no-oracle trust core](docs/no_oracle_trust.md) implements calibration,
frozen residual inference, recursive input boundaries and the Appendix A trust
equations. Its integration into the distributed training loop remains pending.

Run its CPU tests with `python -m unittest discover -s tests -v` (PyTorch and NumPy required).
For portable four-GPU command preparation, asset gates and qualitative comparison,
use the [H100 runbook](docs/h100_runbook.md) and `python -m merl.launch --help`.
The older guide below describes existing entrypoints and debug defaults, not a
verified reproduction of the paper's experimental protocol.

This repository contains one integrated workflow with four user-facing entrypoints:

- MERL online training: mixed real + imagined policy updates with online world-model calibration
- MBRL online training: pure imagined policy updates with world model enabled but not fine-tuned
- MFRL online training: real-only baseline without world model
- Ctrl-World offline pretraining: standalone world-model training under modules/ctrl_world

The recommended user entry scripts are:

- examples/train_merl_debug_3gpu_fix_pro_.sh
- examples/MBRL/train_mbrl_debug_3gpu_fix_pro_.sh
- examples/MFRL/train_mfrl_debug_3gpu_fix_pro_.sh
- examples/generate_shared_wm_eval_dataset.sh
- modules/ctrl_world/train_new.sh

The legacy wrappers examples/online_train.sh and examples/offline_train.sh are kept only for reference. For current runs, prefer the five scripts above.

## 1. What You Need Before Running

You need five things prepared before the first launch:

1. A Linux GPU machine with CUDA, NCCL, and headless OpenGL support.
2. A local OpenVLA SFT checkpoint directory that already contains dataset_statistics.json.
3. A local LIBERO checkout for offline Ctrl-World training and offline dataset tools.
4. A local LIBERO_PRO checkout for online rollout, shared WM eval generation, and MFRL / MBRL / MERL.
5. For MBRL, MERL, or Ctrl-World training, the local world-model backbones and optional warm-start checkpoints.

The default code paths are hardcoded to the original authors' machines. Before running anything, search for lines marked with `#!` or `# todo` in these files and replace them with your own paths:

- examples/train_merl_debug_3gpu_fix_pro_.sh
- examples/MBRL/train_mbrl_debug_3gpu_fix_pro_.sh
- examples/MFRL/train_mfrl_debug_3gpu_fix_pro_.sh
- examples/generate_shared_wm_eval_dataset.sh
- modules/ctrl_world/train_new.sh
- modules/ctrl_world/config.py
- configs/wm_online_config.py
- configs/wm_offline_config.py
- configs/evaluation_config.yaml

## 2. System Dependencies

Install the system packages below first.

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg \
  xvfb \
  libgl1-mesa-glx \
  libglu1-mesa \
  libxrender1 \
  libsm6 \
  libxext6
```

If you run on a headless server, start a virtual display in every new shell before online rollout, evaluation, or dataset generation:

```bash
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
export MUJOCO_GL=glx
export PYOPENGL_PLATFORM=glx
```

The three maintained online scripts already export these variables. The snippet above is still useful for manual debugging, notebooks, or custom launch commands.

## 3. Python Environment

The integrated MERL / MBRL / MFRL workflow is best kept on Python 3.10.

```bash
conda create -n merl python=3.10 -y
conda activate merl
python -m pip install --upgrade pip setuptools wheel
```

Install a CUDA-matched PyTorch build first. Replace `cu121` with the CUDA version that matches your machine.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install flash-attn. Do this after PyTorch is already available.

```bash
pip install flash-attn --no-build-isolation
```

Finally install the remaining Python packages from the root requirements file.

```bash
pip install -r requirements.txt
```

Notes:

- The root requirements.txt intentionally does not force one torch wheel, because the correct wheel depends on your CUDA runtime.
- The standard MERL / MBRL / MFRL path does not require modules/opensora. You can ignore that submodule unless you explicitly work on OpenSora utilities.

## 4. External Code And Assets

### 4.1 OpenVLA SFT checkpoint

All three online modes require one OpenVLA SFT checkpoint directory.

Required file:

- dataset_statistics.json

Required script variables:

- SFT_MODEL_PATH in the three online launch scripts
- VLA_DATASET_STATS_PATH is derived from SFT_MODEL_PATH and must resolve correctly

### 4.2 LIBERO for offline Ctrl-World training

Offline Ctrl-World training and offline LIBERO dataset regeneration use LIBERO, not LIBERO_PRO.

Clone your LIBERO repository and install it into the active Python environment with editable mode:

```bash
git clone <your-LIBERO-repo-url> /path/to/LIBERO
python -m pip install -e /path/to/LIBERO
```

Recommended smoke test:

```bash
python -c "import libero; print(libero.__file__)"
```

Then wire the offline-only paths:

- Set libero_root in configs/wm_offline_config.py to your LIBERO checkout root.
- Set LIBERO_DATASET_ROOT in modules/ctrl_world/train_new.sh to your regenerated offline dataset root.

These two paths are different on purpose:

- libero_root points to the LIBERO codebase.
- LIBERO_DATASET_ROOT or dataset_root_path points to the regenerated offline data directory.

### 4.3 LIBERO_PRO for online rollout and evaluation

Online rollout, shared WM eval dataset generation, and the maintained MFRL / MBRL / MERL launchers use LIBERO_PRO, not plain LIBERO.

Clone your LIBERO_PRO repository locally:

```bash
git clone <your-LIBERO_PRO-repo-url> /path/to/LIBERO_PRO
```

The current code path is config-first. Set line 4 of configs/evaluation_config.yaml so that it points to this checkout:

```yaml
libero_pro_root: "/path/to/LIBERO_PRO"
```

That single root is now the primary online setup path. The rollout code derives bddl, init-state script, init-files, and OOD yaml paths from that root automatically at runtime.

Optional environment-variable fallback:

```bash
export LIBERO_PRO_ROOT=/path/to/LIBERO_PRO
export PYTHONPATH="/path/to/LIBERO_PRO:${PYTHONPATH}"
```

Use the shell fallback only if you intentionally manage imports outside the repo config. The maintained scripts prefer the config-driven path above.

Current script-to-config wiring:

- examples/generate_shared_wm_eval_dataset.sh defaults LIBERO_PRO_EVAL_CONFIG_PATH to configs/evaluation_config.yaml.
- examples/train_merl_debug_3gpu_fix_pro_.sh defaults libero_pro_eval_config_path to configs/evaluation_config.yaml.
- examples/MBRL/train_mbrl_debug_3gpu_fix_pro_.sh defaults libero_pro_eval_config_path to configs/evaluation_config.yaml.
- examples/MFRL/train_mfrl_debug_3gpu_fix_pro_.sh defaults libero_pro_eval_config_path to configs/evaluation_config.yaml.

So in the normal workflow, all maintained online entrypoints read the same online root from configs/evaluation_config.yaml line 4.

### 4.4 World-model assets

For MERL, MBRL, and Ctrl-World offline pretraining, set these paths in configs/wm_online_config.py and configs/wm_offline_config.py:

- svd_model_path
- clip_model_path
- ckpt_path if you want warm-start initialization
- libero_root for offline LIBERO dataset regeneration tools
- dataset_sample_json_dir

## 5. One-Time Configuration Checklist

Before the first run, edit the fields marked with `#!` or `# todo`.

### 5.1 Online scripts

In the three maintained training scripts, the most important user-editable variables are:

- CUDA_VISIBLE_DEVICES
- SFT_MODEL_PATH
- CKPT_PATH
- ALIGN_PATH
- NUM_GPUS
- WM_GPU_IDX
- libero_pro_eval_config_path
- libero_pro_root in configs/evaluation_config.yaml
- world_model_config_path in MERL and MBRL
- shared_wm_eval_root in MERL and MBRL

Relationship between the online config and the maintained scripts:

- In the normal setup, keep libero_pro_eval_config_path at its default repo-local value.
- Change only configs/evaluation_config.yaml line 4, namely libero_pro_root: "/path/to/LIBERO_PRO".
- examples/generate_shared_wm_eval_dataset.sh, examples/train_merl_debug_3gpu_fix_pro_.sh, examples/MBRL/train_mbrl_debug_3gpu_fix_pro_.sh, and examples/MFRL/train_mfrl_debug_3gpu_fix_pro_.sh all read that same file by default.

### 5.2 Shared WM eval dataset generation

In examples/generate_shared_wm_eval_dataset.sh, update:

- CUDA_VISIBLE_DEVICES
- SFT_MODEL_PATH
- WORLD_MODEL_CONFIG_PATH
- LIBERO_PRO_EVAL_CONFIG_PATH
- ALIGN_PATH
- SHARED_WM_EVAL_ROOT

In the normal case, you do not need to change LIBERO_PRO_EVAL_CONFIG_PATH itself. Keep it pointing to the repo-local configs/evaluation_config.yaml, and edit only libero_pro_root there.

### 5.3 Ctrl-World offline pretraining

In modules/ctrl_world/train_new.sh and configs/wm_offline_config.py, update:

- CUDA_VISIBLE_DEVICES
- dataset_root_path
- dataset_names
- svd_model_path
- clip_model_path
- ckpt_path if used
- libero_root
- dataset_sample_json_dir
- tag

Relationship between the offline config and the maintained offline script:

- modules/ctrl_world/train_new.sh consumes the offline dataset root.
- configs/wm_offline_config.py:libero_root points to the LIBERO code checkout used by offline dataset tools.
- Offline Ctrl-World training does not read configs/evaluation_config.yaml and does not use LIBERO_PRO.

## 6. Recommended Run Order

### Step 1: Generate the shared WM eval dataset

This is required before MERL and MBRL if you want stable fixed-set WM evaluation during training.

```bash
bash examples/generate_shared_wm_eval_dataset.sh
```

Expected output root:

```text
./tmp_files/wm_eval_shared/<dataset_name>/wm_eval_fixed_mini/...
./tmp_files/wm_eval_shared/<dataset_name>/wm_eval_fixed_full/...
```

### Step 2: Run the MFRL baseline

```bash
bash examples/MFRL/train_mfrl_debug_3gpu_fix_pro_.sh
```

### Step 3: Run MBRL

```bash
bash examples/MBRL/train_mbrl_debug_3gpu_fix_pro_.sh
```

### Step 4: Run MERL

```bash
bash examples/train_merl_debug_3gpu_fix_pro_.sh
```

All three scripts support:

- RESUME_ENABLE=auto
- RESUME_ENABLE=true
- RESUME_ENABLE=false

If you want the simplest behavior, keep the default `auto` and do not reuse an old EXPERIMENT_NAME unless you intend to resume it.

## 7. Ctrl-World Offline Pretraining

The maintained offline entrypoint is:

```bash
bash modules/ctrl_world/train_new.sh
```

The script launches:

```text
modules/ctrl_world/scripts/train_new.py
```

You must first change the dataset_root_path in modules/ctrl_world/train_new.sh to your regenerated LIBERO dataset root.

If you prefer running it manually:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
accelerate launch \
  --num_processes 1 \
  --main_process_port 29501 \
  modules/ctrl_world/scripts/train_new.py \
  --dataset libero \
  --dataset_root_path /path/to/regen_dataset_no_noops \
  --dataset_names libero_goal libero_object libero_spatial libero_10
```

## 8. What Each Mode Actually Does

The training entrypoint is verl/trainer/main_ppo.py.

- MFRL uses `trainer.fit()` with `world_model.enable=False`
- MBRL uses `trainer.fit_wm_v5()` with `world_model.enable=True` and `wm_fine_tune=False`
- MERL uses `trainer.fit_wm_v5()` with `world_model.enable=True` and `wm_fine_tune=True`

In practice, that means:

- MFRL is the real-only baseline.
- MBRL uses the world model for imagined policy updates but does not fine-tune the world model online.
- MERL uses the world model and also performs online world-model calibration.

## 9. Common Failure Checks

If you want to avoid the most common first-run errors, verify the following before launch.

### 9.1 Missing OpenVLA stats

Make sure this file exists:

```text
${SFT_MODEL_PATH}/dataset_statistics.json
```

### 9.2 Missing shared WM eval dataset

MERL and MBRL preflight checks will fail if shared_wm_eval_root does not contain `.tar` shards under both fixed splits. Generate it first with examples/generate_shared_wm_eval_dataset.sh.

### 9.3 Broken LIBERO or LIBERO_PRO path

Online failures:

- If configs/evaluation_config.yaml line 4 still points to /path/to/LIBERO_PRO, online rollout and evaluation will fail during environment import or task preprocessing.
- If you choose the shell fallback path instead of the config-driven path, LIBERO_PRO_ROOT and PYTHONPATH must both resolve correctly.

Offline failures:

- If LIBERO was not installed with python -m pip install -e /path/to/LIBERO, offline Ctrl-World tools may fail on import libero.
- If configs/wm_offline_config.py:libero_root still points to /path/to/LIBERO, offline dataset regeneration tools will fail to find the LIBERO codebase.

### 9.4 Headless rendering not started

If you see display, OpenGL, or MuJoCo render errors, start Xvfb first and export DISPLAY, MUJOCO_GL, and PYOPENGL_PLATFORM.

### 9.5 GPU split mismatch

In MERL and MBRL:

- NUM_GPUS is the actor side GPU count
- WM_GPU_IDX must point to one extra visible GPU reserved for the world model trainer
- CUDA_VISIBLE_DEVICES must expose both groups at once

### 9.6 Resume confusion

The maintained launchers already fail fast if:

- RESUME_ENABLE=true but no valid checkpoint exists
- RESUME_ENABLE=false but the experiment directory already contains artifacts

Do not manually mix fresh and resumed runs in the same experiment directory.

## 10. Outputs And Metrics

Main outputs are written under:

- checkpoints/MERL
- checkpoints/MBRL
- checkpoints/MFRL
- tmp_files/rollout
- tmp_files/wm_eval_shared

For the current comparison logic, WM metrics, expected rankings, and result summarization, read:

- examples/README.md

That file documents:

- the expected MERL > MBRL > MFRL ranking
- the meaning of `val/test_score/all`, `train_reward/*`, and `wm/eval/*`
- the post-run comparison script

## 11. Minimal Successful Session

If you only want the shortest reliable path:

```bash
conda create -n merl python=3.10 -y
conda activate merl
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install -r requirements.txt

git clone <your-LIBERO-repo-url> /path/to/LIBERO
python -m pip install -e /path/to/LIBERO

git clone <your-LIBERO_PRO-repo-url> /path/to/LIBERO_PRO
# then set configs/evaluation_config.yaml line 4: libero_pro_root: "/path/to/LIBERO_PRO"

export LIBERO_PRO_ROOT=/path/to/LIBERO_PRO
# optional fallback only; the preferred setup is configs/evaluation_config.yaml:libero_pro_root

Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
export MUJOCO_GL=glx
export PYOPENGL_PLATFORM=glx

bash examples/generate_shared_wm_eval_dataset.sh
bash examples/MFRL/train_mfrl_debug_3gpu_fix_pro_.sh
bash examples/MBRL/train_mbrl_debug_3gpu_fix_pro_.sh
bash examples/train_merl_debug_3gpu_fix_pro_.sh
```

If each script still fails, the first thing to inspect is not the trainer internals but the user-editable path block marked with `#!` or `# todo` in the corresponding script or config file.
