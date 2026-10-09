# Run entrypoints

| Script | Purpose |
| :--- | :--- |
| `run_libero.sh` | Forward arguments to `python -m merl.launch` for training, evaluation or collection |
| `run_presentation.sh` | Sequential mode comparison with robot videos and quantitative summaries |
| `generate_shared_wm_eval_dataset.sh` | Collect fixed mini/full evaluation panels |
| `real_world_wm_predict.sh` | Predict from camera video or frame directories |
| `real_world_wm_predict_hdf5.sh` | Extract HDF5 windows and predict their future frames |

Start with the [training guide](../docs/training.md), [runtime setup](../docs/runtime.md) and [visualization guide](../docs/visualization.md). Scripts preserve the arguments accepted by their Python entrypoints.

```bash
bash scripts/run_logged.sh bash examples/run_libero.sh \
  --mode MERL --experiment seed0 --vla-init "$VLA_INIT" --wm-checkpoint "$WM_INIT"
```

The log wrapper saves stdout/stderr and exit status under `tmp_files/acp_logs`; set `ACP_LOG_DIR` to change the destination.

Collect fixed evaluation trajectories in a separate directory:

```bash
SFT_CHECKPOINT="$VLA_INIT" SHARED_WM_EVAL=/data/wm_eval \
  EXPERIMENT=wm_eval ACTOR_GPUS=1 bash examples/generate_shared_wm_eval_dataset.sh
```

Mini/full panels contain 2/6 trials per task. Use `--split wm_train --collection-dir /data/wm_init` through `run_libero.sh` for simulator training data, then run `python -m merl.train_simulator`. Algorithm parameters live in `configs/merl.json`.
