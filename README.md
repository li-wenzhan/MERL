# MERL: World Model-Evolving Reinforcement Learning with Trust-Calibrated Imagination

MERL combines policy optimization, a learned simulator and online simulator updates
for robot learning. The repository also provides MFRL and frozen-simulator MBRL
execution modes, standalone Ctrl-World pretraining and real-robot WM inference.

## Implementation status

The distributed WM rollout currently uses a **grounded mirror**, not the recursive
no-oracle rollout in the rebuttal. The isolated [no-oracle core](docs/no_oracle_trust.md)
has calibration, frozen residual inference, recursive input boundaries and trust
equations; connecting its lifecycle and chunk weights to the production trainer
remains pending. See the [research-to-code audit](docs/implementation_audit.md).

The maintained launcher preserves legacy debug algorithm profiles, which are not
matched paper reproduction settings. A successful smoke test establishes no
performance claim.

## Repository map

| Location | Responsibility |
| --- | --- |
| `merl/` | Unified launcher and isolated trust/calibration mechanisms |
| `configs/` | Shared launch profiles, WM settings and LIBERO-PRO perturbations |
| `verl/trainer/` | Ray orchestration, rollout/update schedule and replay |
| `verl/workers/` | FSDP actors, simulator worker and environment services |
| `modules/ctrl_world/` | Simulator, reward proxy and offline WM training |
| `scripts/` | Asset preparation, preflight and result analysis |
| `examples/` | Short online examples and separate real-robot inference scripts |
| `tests/` | CPU research contracts and launcher regressions |
| `real_world/` | Separate real-robot workflows |

## Setup and assets

Use Linux with CUDA/NCCL, headless EGL, ffmpeg and compatible PyTorch/torchvision.
Install `requirements.txt` after selecting the CUDA build. OpenVLA-OFT uses
eager attention; other attention backends may need matching FlashAttention builds. The tested CCI environment is recorded in the
[runbook](docs/h100_runbook.md), not a universal lockfile. OpenSora is not required.

All online modes require an OpenVLA-OFT checkpoint with action statistics and a
[LIBERO-PRO checkout](https://github.com/Zxy-MLlab/LIBERO-PRO). Set `libero_pro_root`
in `configs/evaluation_config.yaml`. Verify the selected OOD task definitions and
initial states; original-task rendering does not validate perturbation assets.

Online rollout with pretrained models **does not require demonstration HDF5
datasets**. Demonstrations are needed for SFT or offline WM pretraining. Fixed WM
evaluation uses separately collected trajectories; keep them out of training/replay.

MERL/MBRL additionally require local SVD/CLIP backbones configured in
`configs/wm_online_config.py` and an explicit trained Ctrl-World `--wm-checkpoint`.
The launcher enables warm-start and strict checkpoint loading must succeed.

## One entrypoint

```bash
python -m merl.launch --help

# Asset/config checks on CCI; no Ray or training.
bash examples/run_libero.sh --mode MERL \
  --sft-checkpoint /models/openvla-oft --wm-checkpoint /models/ctrl-world.pt \
  --experiment merl_smoke_001 --check

# Run on ACP: three actors plus one WM trainer.
bash examples/run_libero.sh --mode MERL \
  --sft-checkpoint /models/openvla-oft --wm-checkpoint /models/ctrl-world.pt \
  --experiment merl_smoke_001 --smoke
```

Select `--mode MFRL` for real-only training (no WM checkpoint), or `--mode MBRL`
for a frozen simulator. Select `--job evaluate` for real-environment evaluation,
or `--job collect` with MFRL for fixed WM evaluation trajectories. `--dry-run`
prints commands without asset/GPU access. Extra Hydra `key=value` overrides follow
`--`; mode, checkpoint, output and layout invariants are protected.

Every run creates a fresh `checkpoints/<mode>/<experiment>` directory containing
`launch_manifest.json`, resolved configuration, selected source hashes and `run.log`. GPU
visibility is preserved; unrelated Ray jobs are never killed. The launcher does
not automatically resume. Full optimizer/RNG/replay restoration needs a separate audit.

ACP startup requires neither Git nor external network access. To capture the whole
console, including preflight failures, prefix a command with
`bash scripts/run_logged.sh`; logs and exit/timing summaries go to
`tmp_files/acp_logs/` (`ACP_LOG_DIR` overrides the location). Hugging Face loading
defaults to offline mode. Code synchronization is managed outside the job.

WM evaluation defaults to **off** while bringing up the pipeline. Collect both
fixed splits with `examples/generate_shared_wm_eval_dataset.sh`, then enable it
using `--wm-eval fixed --shared-wm-eval /data/wm_eval`. Missing data fails early.
See [examples](examples/README.md) and the [runbook](docs/h100_runbook.md).

## Validation and experiments

```bash
python -m unittest discover -s tests -v
PYTHONPATH=$PWD python scripts/verify_merl_memory_contract.py
```

Start with asset checks, environment reset/step, policy inference and one short
update before larger ACP runs. `--smoke` truncates tasks/horizons, so its success
rates are not benchmark results. Record actual transitions, seeds, failures,
checkpoints and budgets; evaluate methods under the same protocol. Do not assume
a ranking before collecting evidence.

Offline WM pretraining remains at `modules/ctrl_world/train_new.sh`. Configure
its regenerated dataset root and `configs/wm_offline_config.py` separately; this
uses offline LIBERO, not the online LIBERO-PRO evaluation configuration.

Private paper/rebuttal materials, ACP deployment files, credentials, outputs and
weights stay outside Git. Third-party licenses are preserved; the authors still
need to select a root license for MERL's contributions before public release.
