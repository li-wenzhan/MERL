# Minimal entrypoints

`run_libero.sh` forwards to `python -m merl.launch`: one entry for MERL and its
four component controls, with `--job train|evaluate|collect`. Devices and assets are arguments, not
separate scripts. Internal helpers now live in `scripts/`.

`run_presentation.sh` runs short training and complete real-environment evaluation
sequentially for all three modes on one four-GPU allocation. It saves checkpoints,
all requested episode videos, keyframes and comparison reports. See the
[presentation pilot](../docs/presentation_pilot.md) for budgets and limitations.

## Fixed WM evaluation data

```bash
SFT_CHECKPOINT=/models/openvla-oft \
SHARED_WM_EVAL=/data/wm_eval_run001 EXPERIMENT=wm_eval_run001 ACTOR_GPUS=1 \
bash examples/generate_shared_wm_eval_dataset.sh
```

Collection uses deterministic real-environment trajectories without policy updates.
Mini/full use 2/6 trials per task and may share initial-state prefixes: they are not
independent statistical replicates. Nonempty splits cannot be reused. Keep this
root separate from training data and record checkpoint, task/trial IDs and revision.

## Real-environment evaluation

```bash
bash examples/run_libero.sh --mode MERL --job evaluate \
  --sft-checkpoint /models/openvla-oft \
  --actor-checkpoint checkpoints/MERL/RUN/actor/global_step_100 \
  --experiment merl_eval_001 --actor-gpus 3 --trials 6
```

Supply base SFT assets including action statistics and a complete FSDP checkpoint
directory with its original rank count. For a full HF export, omit
`--actor-checkpoint` and supply that export as the SFT asset. Evaluation uses environment
success. Match evaluation configuration, initial states, sampling and horizons
across methods; retain failures and timeouts.

## Result inspection

```bash
python scripts/compare_mode_results.py \
  --mfrl checkpoints/MFRL/<run> --mbrl checkpoints/MBRL/<run> \
  --merl checkpoints/MERL/<run>
```

Inspect `val/test_score/all` for environment success; training proxy reward is not
a success label. With fixed WM evaluation enabled, inspect `wm/eval/*` plus
missing-data/error diagnostics. Camera-ready controls share optimizer and data
contracts. Legacy profiles differ and require separate labeling.
See [the runbook](../docs/h100_runbook.md).

The two `real_world_wm_predict*.sh` scripts handle separate video/HDF5 inputs.
Offline simulator training remains at `modules/ctrl_world/train_new.sh`.
