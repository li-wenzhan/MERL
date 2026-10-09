# Evaluation and visualization

## Robot behavior

`merl.launch --job evaluate` runs the actor in real LIBERO environments. Each saved episode contains the task/trial IDs, outcome, environment steps, MP4, keyframes, aligned executed-action trajectory and `episode.json`. Use identical evaluation configuration, task states and horizons across methods.

`examples/run_presentation.sh` trains selected modes sequentially and evaluates every requested trial. The default comparison uses task 0, three trials per task and evaluation states beginning at offset 10. `--tasks`, `--trials`, `--eval-offset`, `--steps` and `--training-minutes` set the panel and budget. Training and evaluation trial IDs must be disjoint within the fixed 50-state panel.

The runner writes `protocol.json` with asset hashes and trial IDs, a `run_info.json` per mode, and `report/` with summary CSV/JSON, score cards, side-by-side videos and contact sheets. Reports include completed updates, gradient norms, simulator updates and status. Incomplete panels retain their artifacts and do not report a success rate. `--continue-on-error` attempts the remaining modes after saving a failed mode's record.

To regenerate comparisons from saved artifacts:

```bash
python -m merl.presentation_report outputs/comparison_seed0
```

Fixed evaluation collection uses state IDs beginning at 10; simulator training collection starts at 0. `--collection-trial-offset` selects a different starting state.

## World-model imagination

Choose a held-out `episode.json` saved by real-environment evaluation. Pass the initial frozen simulator as `MBRL` and updated simulators as `ONLINE_MBRL` and `MERL`:

```bash
python -m merl.wm_visual_compare \
  --episode /outputs/heldout/episode.json \
  --checkpoint MBRL=/outputs/wm_init/checkpoint-5000.pt \
  --checkpoint ONLINE_MBRL=/outputs/online/world_model/global_step_5/world_model.pth \
  --checkpoint MERL=/outputs/merl/world_model/global_step_5/world_model.pth \
  --start 64 --horizon 32 --rollout recursive --inference-steps 8 \
  --output outputs/wm_comparison
```

Each model receives the same anchor, past actions, recorded future actions and diffusion seed. Models load sequentially on one GPU. `recursive` conditions later chunks on predicted RGB; `teacher_forced` refreshes history from the recorded observations at chunk boundaries. GT frames are used as comparison targets. Choose a window with sufficient history and valid recorded future actions.

Outputs include GT/predicted video panels, contact sheets, per-model RGB/proxy arrays, pixel MSE, PSNR and checkpoint/reference hashes. These metrics describe fixed-action prediction fidelity. Closed-loop task success is measured by robot evaluation.

## Learning curves

`scripts/plot_log_metrics.py` reads saved trainer logs and produces success-rate, AUC, interaction-budget and trust plots. `scripts/compare_mode_results.py` summarizes runs across modes. Use the same success threshold and evaluation schedule for convergence comparisons.

## Robot camera data

The [real-world tools](../real_world/README.md) accept video, frame directories and HDF5 trajectories. They export input histories, predicted frames, proxy scores and manifests; sliding prediction saves per-chunk GT and predicted panels.
