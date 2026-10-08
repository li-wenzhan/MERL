# Legacy trainer memory contract

These replay-pool settings apply to `--protocol legacy`. The default camera-ready
driver uses chunk batches and complete-stage snapshots; see
[camera-ready protocol](camera_ready_protocol.md).

## New Dependencies

None. Use the existing MeRL training environment.

## New / Changed Arguments

- `++prio_real_capacity=128`
- `++prio_wm_capacity=128`
- `++actor_rollout_ref.rollout.video_save_interval=5`
- `++actor_rollout_ref.world_model.persist_imag_rollout_shards=False`
- `++actor_rollout_ref.world_model.fixed_eval_full_samples=100`
- `++trainer.resume.persist_replay_pool=True` for MERL / MBRL
- `++trainer.resume.replay_pool_keep_last=1`
- `++trainer.resume.replay_pool_max_gb=15`
- `++trainer.strict_mode_assert=True` for MERL imagined-contract fail-fast

## Verification

```bash
PYTHONPATH=$PWD python scripts/verify_merl_memory_contract.py
python -m py_compile \
  verl/trainer/ppo/ray_trainer.py \
  verl/trainer/ppo/dataproto_filter.py \
  verl/workers/actor/dp_rob.py \
  verl/workers/rollout/rob_rollout_wm_pro.py \
  scripts/verify_merl_memory_contract.py \
  scripts/plot_log_metrics.py
```

## Expected Result

- The verification script prints `PASS`.
- Replay pools keep at most 128 real and 128 WM samples.
- Returned/mixed replay DataProto objects do not contain `video`, `env_video`, or `env_dones`.
- PPO / WM-prompt batches keep `task_id`, `trial_id`, `action`, `wm_obs_error`, `wm_done_error`, and `wm_pred_valid`.
- Imagined samples with zero valid response tokens are filtered before PPO.
- MERL resume stores replay snapshots under `resume/replay_pool/`, keeps the latest one, and enforces the configured size cap.
