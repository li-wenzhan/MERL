<div align="center">

<a href="https://merl-project.github.io/">
  <img src="assets/readme/banner.svg" width="100%" alt="MERL. Learning to trust imagination. Model-Evolving Reinforcement Learning, CoRL 2026.">
</a>

<h1>Trust-Calibrated VLA Policy Post-Training<br> with Evolving Imagination</h1>

<p>
  Wenzhan Li<sup>1,2</sup> &nbsp; Yiran Qin<sup>3,4,5</sup> &nbsp; Heng Zhou<sup>4,6</sup> &nbsp; Huirui Wang<sup>1</sup><br>
  Yulan Guo<sup>1,2</sup> &nbsp; Ruimao Zhang<sup>1 *</sup>
</p>
<p>
  <sup>1</sup>Sun Yat-sen University &nbsp; <sup>2</sup>Shenzhen Loop Area Institute<br>
  <sup>3</sup>The Chinese University of Hong Kong, Shenzhen &nbsp; <sup>4</sup>Shanghai AI Laboratory<br>
  <sup>5</sup>University of Oxford &nbsp; <sup>6</sup>University of Science and Technology of China
</p>

<p>
  <a href="https://www.corl.org/"><img src="https://img.shields.io/badge/CoRL-2026-BF5700?style=for-the-badge&amp;labelColor=71380D" height="28" alt="CoRL 2026"></a>
  <a href="https://arxiv.org/abs/2610.16517"><img src="https://img.shields.io/badge/arXiv-2610.16517-B31B1B?style=for-the-badge&amp;labelColor=6F1111&amp;logo=arxiv&amp;logoColor=white" height="28" alt="Paper: arXiv 2610.16517"></a>
  <a href="https://merl-project.github.io/"><img src="https://img.shields.io/badge/Project-Explore-92278F?style=for-the-badge&amp;labelColor=291C32" height="28" alt="Explore the interactive project page"></a>
  <a href="https://github.com/li-wenzhan/MERL"><img src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&amp;labelColor=30363D&amp;logo=github&amp;logoColor=white" height="28" alt="MERL implementation on GitHub"></a>
</p>

**A world model can imagine. MERL learns how much to trust it.**

<p>
  <a href="#overview">Overview</a> &nbsp;·&nbsp;
  <a href="#robot-demos">Robot demos</a> &nbsp;·&nbsp;
  <a href="#method">Method</a> &nbsp;·&nbsp;
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#training">Training</a> &nbsp;·&nbsp;
  <a href="#evaluation-and-visualization">Evaluation</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

</div>

## Overview

MERL uses grounded experience to improve both its policy and its simulator. It calibrates visual and progress errors on stored trajectories, then estimates the reliability of new imagined chunks with a frozen residual predictor. This supports recursive imagination as the policy changes, while concentrating policy updates on reliable experience.

The implementation combines tokenized [**OpenVLA-OFT**](https://github.com/moojink/openvla-oft), [**Ctrl-World**](https://github.com/Robert-gyj/Ctrl-World), [**LIBERO-PRO**](https://github.com/Zxy-MLlab/LIBERO-PRO), [Ray](https://docs.ray.io/en/latest/index.html) and [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html). This repository provides the training loop, component controls, evaluation, resumable checkpoints, and tools for robot videos and simulator comparisons.

<table align="center">
<tr>
<td align="center" width="33%"><h3>79.7%</h3><b>LIBERO-PRO success</b><br><sub>Four suites · three training seeds</sub></td>
<td align="center" width="34%"><h3>10.9% fewer</h3><b>High-threshold stages</b><br><sub>Versus the matched model-free control</sub></td>
<td align="center" width="33%"><h3>80.0%</h3><b>Observed robot success</b><br><sub>32 / 40 trials · fixed-data refinement</sub></td>
</tr>
</table>

**Try the research:** [explore the trust lab](https://merl-project.github.io/#trust), [inspect imagined futures](https://merl-project.github.io/#imagination), or [watch the robots](https://merl-project.github.io/#robots).

## Robot demos

Real MERL-Full executions on AgileX PiPER X. Click a rollout to open the full-resolution video gallery.

<table>
<tr><th align="center">Block-to-Box</th><th align="center">Lid-to-Pot</th></tr>
<tr>
<td width="50%"><a href="https://merl-project.github.io/#robots"><img src="assets/readme/block-merl.gif" width="100%" alt="MERL-Full closes its gripper around a block and places it in the box. Original two-times-speed physical rollout."></a></td>
<td width="50%"><a href="https://merl-project.github.io/#robots"><img src="assets/readme/lid-merl.gif" width="100%" alt="MERL-Full grasps a lid and places it onto the pot. Original two-times-speed physical rollout."></a></td>
</tr>
<tr><td align="center"><b>17 / 20</b> successful evaluation trials</td><td align="center"><b>15 / 20</b> successful evaluation trials</td></tr>
</table>

<sub>Qualitative examples at 2× physical-time playback; the original speed marks are retained. Evaluation uses held-out placements and colors after offline refinement from 50 demonstrations per task, without extra physical training rollouts.</sub>

## Method

**Update the simulator. Then calibrate how its imagination shapes learning.**

An adapting world model can still produce uneven imagined chunks. MERL turns estimated reliability into four explicit decisions:

<p align="center">
  <a href="https://merl-project.github.io/#trust"><picture><source media="(max-width: 640px)" srcset="assets/readme/trust-compass-mobile.svg"><img src="assets/readme/trust-compass.svg" width="100%" alt="Four trust decisions: stage-level loss mixing rho and rollout horizon h; chunk-level replay probability p and optimization weight w."></picture></a>
</p>

Stage-level trust controls the imagined loss contribution and horizon. Chunk-level trust controls selection and influence after sampling. A frozen residual predictor scores recursive imagination **without querying future ground truth**.

<details>
<summary><b>Open the complete pipeline and five-step refinement cycle</b></summary>

<p align="center">
  <a href="https://merl-project.github.io/#method"><img src="assets/merl_overview.png" width="100%" alt="Original MERL pipeline: grounded rollout, stored-action calibration, stage scheduling, chunk trust and mixed policy optimization."></a>
</p>

1. **Policy rollout.** Collect six grounded trajectories with the current VLA policy. Decode and denormalize categorical tokens into 8-step, 7-dimensional action chunks; store the executed commands with their preceding and resulting RGB observations.
2. **Stored grounded-imagined calibration.** Update the world model and progress proxy on stored grounded windows. Replay the same recorded actions to obtain matched predictions and residual targets, then refresh the residual predictor. Calibration uses the stored observations and masks without additional environment interaction.
3. **Stage-level trust.** Smooth the measured visual and proxy errors to schedule the real–imagined mixture ratio and the imagination horizon, between 8 and 32 environment steps.
4. **Chunk-level trust.** Re-query the policy on predicted RGB observations to imagine recursively. The frozen predictor estimates visual and proxy residuals from the anchor, imagined latents, executed actions, rollout depth and stage context. These errors determine replay probabilities and detached trust weights; partial chunks retain explicit masks.
5. **Mixed optimization.** Sample real and imagined chunks independently, then optimize `(1 − ρ) L_real + ρ L_imag`. Categorical token losses are summed within each valid chunk; the two branches are normalized by their own chunk counts. The simulator, proxy and residual predictor stay frozen during this policy update.

| Mode | Simulator | Stage scheduling | Imagined chunk sampling / weighting | Policy data |
| :--- | :--- | :--- | :--- | :--- |
| `MFRL` | Disabled | — | — | Real |
| `MBRL` | Frozen | Fixed ratio / horizon | Uniform / unit weight | Real + imagined |
| `STATIC_TRUST` | Frozen | Calibrated | Trust priority / weight | Real + imagined |
| `ONLINE_MBRL` | Online updated | Fixed ratio / horizon | Uniform / unit weight | Real + imagined |
| `MERL` | Online updated | Calibrated | Trust priority / weight | Real + imagined |

All modes share the grounded budget, actor optimizer, action representation and evaluation interface. See [training details](docs/training.md) and [no-oracle trust](docs/no_oracle_trust.md).

</details>

## Results

### LIBERO-PRO · same-protocol comparisons

Mean ± standard deviation over **three training seeds**, averaged across Spatial, Object, Goal and Long. Task semantics stay fixed while deployment conditions change.

| Method | Average SR (%) ↑ | AUC ↑ | S2T-H ↓ |
| :--- | ---: | ---: | ---: |
| SimpleVLA-RL | 77.1 ± 1.1 | 67.8 ± 1.0 | 104.1 ± 7.0 |
| WMPO† | 75.4 ± 1.2 | 63.2 ± 1.3 | 149.6 ± 10.2 |
| WoVR† | 77.8 ± 0.9 | 68.4 ± 0.9 | 101.3 ± 6.2 |
| **MERL** | **79.7 ± 0.7** | **70.6 ± 0.8** | **92.1 ± 4.8** |

<sub>† Baselines adapted to the same training/evaluation protocol; each WoVR seed has an independent simulator. AUC is normalized learning-curve area. S2T-H averages stages to 80% success on Spatial/Object/Goal and 65% on Long.</sub>

<details>
<summary><b>Matched component controls and budget accounting</b></summary>

Component comparison averaged over four perturbed task suites:

| Method | Average SR ↑ | AUC ↑ | S2T-H ↓ |
| :--- | ---: | ---: | ---: |
| MFRL | 77.2 | 67.9 | 103.4 |
| Static-MBRL | 73.3 | 59.8 | 168.9 |
| Static-MBRL + Trust | 75.1 | 63.5 | 145.8 |
| Online-Updated MBRL | 76.2 | 65.4 | 139.6 |
| MERL | **79.7** | **70.6** | **92.1** |

SR measures task success; AUC summarizes the success-rate learning curve; S2T-H counts refinement stages to the target success threshold with interpolated crossings. Each stage allows six grounded trajectories and at most 3,072 environment transitions.

The **10.9% reduction** is `(103.4 − 92.1) / 103.4` against the matched MFRL control. Imagination and simulator updates add model computation; fewer refinement stages do not imply the same percentage of wall-clock speedup.

</details>

### Physical deployment · fixed data, closed-loop evaluation

MERL-Full achieves **80.0%** observed average success (32/40; **95% CI [65.2, 89.5]**). MERL-Visual achieves 77.5% (31/40); overlapping confidence intervals do not resolve a proxy-only gain. These experiments cover two tabletop tasks and offline refinement.

[See the interactive results](https://merl-project.github.io/#results) · [Read the experimental protocol](https://arxiv.org/abs/2610.16517)

---

## Installation

<p>
  <img src="https://img.shields.io/badge/Python-3.10-92278F?logo=python&amp;logoColor=white&amp;labelColor=291C32" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-FSDP-C82E70?logo=pytorch&amp;logoColor=white&amp;labelColor=291C32" alt="PyTorch FSDP">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20CUDA-92278F?labelColor=291C32" alt="Linux and CUDA">
</p>

Use **Linux, Python 3.10, CUDA PyTorch/torchvision, NCCL, EGL and ffmpeg**. Policy training uses three actor GPUs; model-based modes add a fourth GPU for the shared simulator. A node with **4 × H100 80 GB** supports this layout. Simulator initialization and video comparison use one GPU.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
# Install the CUDA PyTorch/torchvision pair for your host, then:
python -m pip install -r requirements.txt
```

[Runtime versions](configs/runtime_versions.json) and [runtime setup](docs/runtime.md) provide dependency and headless-rendering details. OpenVLA-OFT uses eager attention in this implementation.

Prepare the [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) checkout and local [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) and [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) backbones before training:

```bash
export LIBERO_PRO_ROOT=/benchmark/LIBERO-PRO
export MERL_SVD_MODEL_PATH=/models/stable-video-diffusion-img2vid
export MERL_CLIP_MODEL_PATH=/models/clip-vit-base-patch32
export TORCH_HOME=/models/torch
# Cache the ImageNet initialization for the progress proxy during asset preparation.
python -c 'from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)'
```

Online refinement collects its own experience. Demonstrations in RLDS format are needed when preparing the initial VLA policy; simulator initialization below uses locally collected trajectories. `configs/pretraining_config.yaml` selects the original tasks, while `configs/evaluation_config.yaml` selects environment perturbations. Prepare matching BDDL and initial-state assets for each selected panel.

## Training

Start a fresh training run with the following sequence. MERL policy checkpoints are generated locally by these commands.

### 1. Prepare the VLA initialization

Train a categorical action-chunk policy from the public [OpenVLA base model](https://huggingface.co/openvla/openvla-7b) using the [OpenVLA-OFT fine-tuning workflow](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md). Select **discrete token prediction**, **one RGB image**, **no proprioception** and **8 × 7 actions**:

```text
--use_l1_regression False --use_diffusion False --use_film False
--num_images_in_input 1 --use_proprio False
```

A complete command and checkpoint preparation steps are in [the initialization guide](docs/training.md#vla-initialization). Export the merged model with its tokenizer, processor and `dataset_statistics.json`, then set:

```bash
export VLA_INIT=/outputs/vla_init/merged_model
export UNNORM_KEY=libero_10_no_noops
```

### 2. Collect simulator training trajectories

Collect real-environment trajectories with the initialized policy, without policy updates:

```bash
bash scripts/run_logged.sh python -m merl.launch \
  --mode MFRL --job collect --experiment wm_data_seed0 \
  --vla-init "$VLA_INIT" --unnorm-key "$UNNORM_KEY" \
  --eval-config configs/pretraining_config.yaml --actor-gpus 1 --trials 6 \
  --collection-dir data/wm_initialization --split wm_train
```

The NPZ files contain `T+1` RGB observations for `T` executed actions, task instructions and outcomes. Training and held-out evaluation collections use separate directories.

### 3. Train the initial simulator and proxy

Initialize from the public visual/text backbones and train on the collected trajectories:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_logged.sh python -m merl.train_simulator \
  --data data/wm_initialization/wm_train/trajectories \
  --output outputs/wm_init/seed0 --steps 5000 --save-every 500 --seed 0
export WM_INIT="$PWD/outputs/wm_init/seed0/checkpoint-5000.pt"
```

The trainer saves model weights, optimizer and RNG state, trajectory hashes and loss curves. `--from-checkpoint` loads an existing simulator initialization; `--resume-from` restores a saved training state. See [simulator initialization](docs/training.md#simulator-initialization).

### 4. Run policy refinement

```bash
bash scripts/run_logged.sh python -m merl.launch \
  --mode MERL --stages 100 --seed 0 --experiment merl_seed0 \
  --vla-init "$VLA_INIT" --unnorm-key "$UNNORM_KEY" --wm-checkpoint "$WM_INIT"
```

Choose `MFRL`, `MBRL`, `STATIC_TRUST` or `ONLINE_MBRL` to run the component controls. MFRL requires only the VLA initialization. For comparisons, reuse the same initialization, task panel and seed schedule across modes.

Algorithm settings live in [configs/merl.json](configs/merl.json). Additional Hydra overrides follow `--`, for example:

```bash
# Disable chunk trust while retaining online evolution and stage scheduling.
python -m merl.launch --mode MERL --experiment no_chunk_seed0 \
  --vla-init "$VLA_INIT" --unnorm-key "$UNNORM_KEY" --wm-checkpoint "$WM_INIT" \
  -- merl.chunk_trust=false
```

Completed checkpoints restore policy, simulator, optimizer, RNG, trust state and replay sampling state. Resume into a fresh experiment directory with an absolute target stage count:

```bash
python -m merl.launch --mode MERL --experiment merl_seed0_resumed --stages 100 \
  --vla-init "$VLA_INIT" --unnorm-key "$UNNORM_KEY" --wm-checkpoint "$WM_INIT" \
  --resume-from checkpoints/MERL/merl_seed0/training_state/completed_stage_000050.pt
```

## Evaluation and visualization

### Closed-loop robot behavior

Evaluate the saved actor on real simulator environments and export episode videos, keyframes, trajectories and success metrics:

```bash
bash scripts/run_logged.sh python -m merl.launch \
  --mode MERL --job evaluate --experiment merl_eval_seed0 \
  --vla-init "$VLA_INIT" --unnorm-key "$UNNORM_KEY" \
  --actor-checkpoint checkpoints/MERL/merl_seed0/actor/global_step_100 \
  --actor-gpus 3 --trials 6
```

FSDP checkpoint evaluation uses the same actor rank count as training. To evaluate the initialized policy on one GPU, omit `--actor-checkpoint` and use `--actor-gpus 1`.

For a compact sequential comparison on one four-GPU node:

```bash
bash examples/run_presentation.sh \
  --modes MFRL MBRL ONLINE_MBRL MERL \
  --vla-init "$VLA_INIT" --unnorm-key "$UNNORM_KEY" --wm-checkpoint "$WM_INIT" \
  --tasks 0 --trials 3 --steps 6 --training-minutes 15 \
  --output outputs/comparison_seed0
```

The comparison runner saves every requested trial, side-by-side videos, contact sheets, `scores.png`, `summary.json` and `trials.csv`. Its evaluation states start at offset 10; each mode records its actual interactions, optimizer updates and elapsed time.

### World-model imagination

Compare checkpoints against the same held-out observations and recorded actions:

```bash
bash scripts/run_logged.sh python -m merl.wm_visual_compare \
  --episode /outputs/heldout/episode.json \
  --checkpoint MBRL="$WM_INIT" \
  --checkpoint ONLINE_MBRL=/outputs/online/world_model/global_step_5/world_model.pth \
  --checkpoint MERL=/outputs/merl/world_model/global_step_5/world_model.pth \
  --start 64 --horizon 32 --inference-steps 8 --rollout recursive \
  --output outputs/wm_comparison_seed0
```

Outputs include GT and predicted videos, image panels, proxy traces, pixel MSE/PSNR and checkpoint/reference hashes. Models load sequentially on one GPU. Recursive prediction conditions on past and predicted RGB; `--rollout teacher_forced` provides a one-chunk diagnostic using stored real history. [Visualization guide](docs/visualization.md).

## Experiment outputs

Each experiment records its resolved configuration, input assets, source hashes, package versions, devices, command and final status. Stage logs include interaction counts, trust decisions, gradient norms and timings. Checkpoints and rollout artifacts are organized under `checkpoints/<MODE>/<EXPERIMENT>/`.

`scripts/run_logged.sh` streams stdout/stderr to the terminal and saves logs under `tmp_files/acp_logs/`, with exit codes and elapsed time in a matching `.status.json`. Set `ACP_LOG_DIR` to choose another persistent location.

```bash
python -m unittest discover -s tests -v
```

## Repository guide

| Path | Purpose |
| :--- | :--- |
| `merl/algorithm.py`, `merl/trainer.py` | Configuration, objectives and refinement stages |
| `merl/simulator.py`, `merl/imagined_rollout.py` | Simulator adaptation and recursive policy imagination |
| `merl/trust.py`, `merl/stored_calibration.py` | Residual prediction and stage/chunk trust |
| `merl/train_simulator.py` | Initial simulator and proxy training |
| `merl/launch.py`, `configs/` | Training, evaluation, collection and runtime configuration |
| `verl/workers/` | Distributed actors and shared simulator worker |
| `modules/ctrl_world/` | Video predictor and progress classifier |
| `real_world/` | Robot-video prediction and HDF5 simulator training |
| `examples/`, `scripts/` | Run wrappers, logging, asset preparation and analysis |
| `tests/`, `docs/` | Contract tests and usage guides |

## Citation

If you find MERL helpful in your research, please consider [giving this repository a ⭐](https://github.com/li-wenzhan/MERL) and citing our paper. Thank you for your interest and support!

```bibtex
@misc{li2026merl,
  title         = {Trust-Calibrated VLA Policy Post-Training with Evolving Imagination},
  author        = {Li, Wenzhan and Qin, Yiran and Zhou, Heng and Wang, Huirui and Guo, Yulan and Zhang, Ruimao},
  year          = {2026},
  eprint        = {2610.16517},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.16517}
}
```

The BibTeX entry above refers to the [arXiv preprint](https://arxiv.org/abs/2610.16517). We will add the official **CoRL 2026 / PMLR** citation once the proceedings are published.

## Acknowledgments

MERL builds on [VLA-RL](https://github.com/GuanxingLu/vlarl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World), [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO), [Ray](https://docs.ray.io/en/latest/index.html) and [PyTorch](https://pytorch.org/). We thank their authors and retain the original notices in vendored source. Refer to each upstream project's terms for its code, datasets and model weights.

<p align="center">
  <a href="https://merl-project.github.io/"><b>Explore MERL</b></a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2610.16517">Read the paper</a> &nbsp;·&nbsp;
  <a href="#installation">Start an experiment</a>
</p>
