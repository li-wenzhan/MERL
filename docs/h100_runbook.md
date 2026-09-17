# Four-H100 preparation and qualitative comparison

## Current execution gate

The supplied workspace currently exposes one H100. On the inspected server,
`configs/evaluation_config.yaml` refers to a missing LIBERO-PRO checkout and the
four-GPU launchers' original VLA path is missing. SVD/CLIP backbone file preflight
passes, but the WM warm-start is disabled. These facts do not establish model
inference compatibility or training readiness.

Allocate four visible 80GB GPUs, provide the actual SFT checkpoint (including
`dataset_statistics.json`), the intended trained Ctrl-World checkpoint, and a
compatible LIBERO-PRO checkout. Confirm which simulator checkpoint contains a
trained reward proxy. A generic video backbone is not an adequate MBRL baseline.

## Portable entrypoint

`python -m merl.launch` wraps the existing four-GPU scripts. It removes the need
to edit their author-specific paths, fixes the 3-actor + 1-WM layout, rejects reuse
of experiment directories and writes a launch manifest with source hashes, Git
revision, package versions, GPU information and exit status. Existing scripts also
accept trailing Hydra overrides and environment overrides for asset/config paths.

This is an entrypoint to **legacy debug profiles**, not a matched paper reproduction.
It does not activate the new no-oracle module. See [implementation audit](implementation_audit.md).
The manifest records explicit overrides and the script hash; the existing trainer
prints its resolved Hydra config in the run log. Preserve both. It is not a complete
RNG/resume checkpoint and does not fix hardcoded internal seeds.

```bash
cd /path/to/MERL
source /path/to/merl/bin/activate

# Side-effect-free command review; works without a GPU or model files.
python -m merl.launch --mode MERL \
  --sft-checkpoint /models/openvla-oft \
  --eval-config /configs/libero-pro.yaml \
  --wm-config /configs/wm_online_config.py \
  --shared-wm-eval /data/wm_eval_shared/libero_10 \
  --experiment merl_smoke_001 --dry-run -- trainer.total_training_steps=1

# Run only after all gates below pass: omit --dry-run.
```

The same interface accepts `--mode MFRL` and `--mode MBRL`. The modes retain their
original hyperparameters, which are currently **not matched**. Training-step,
temperature and policy-update differences must be resolved before comparing gains.
The wrapper deliberately starts fresh; use the legacy entrypoint for an explicit
resume until the complete state-restoration audit is finished.

## Gates, in order

1. CPU contracts: `python -m unittest discover -s tests -v`.
2. GPU inventory: `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv`.
3. Asset/config checks:
   ```bash
   python scripts/preflight_world_model_backbone.py --config /configs/wm_online_config.py
   python scripts/preflight_libero_pro_compat.py --config /configs/libero-pro.yaml
   ```
4. One-task rendering: configure GLX/EGL using `examples/libero_glx_runtime.sh`,
   then run `scripts/preflight_libero_env_service.py --config /configs/libero-pro.yaml`.
5. Existing data/reward contracts: `PYTHONPATH=$PWD python scripts/verify_merl_memory_contract.py`.
6. One-stage four-device run. Verify finite loss/gradients, sampled-token coverage,
   actual real-environment transitions, completed trajectories, and checkpoint load.
7. Only then collect the common evaluation panel and larger experiments.

## Visualization protocol

Claim: compare closed-loop behavior of fixed MFRL/MBRL/MERL checkpoints under
identical perturbations; separately compare simulator predictions from identical
stored anchors/actions. These are different questions and need separate panels.

- Fix task IDs, initial states, perturbation config, evaluation seeds, action chunk,
  horizon, camera, sampling temperature and checkpoint selection rule in advance.
- Evaluate all methods using real environment observations and success labels.
  Never count a reward-proxy threshold as environment success.
- Save every selected trial, including failures/timeouts, with mode/checkpoint,
  task/trial/seed, actions, success, episode length, timestamps and video path.
- Choose a fixed trial-ID panel before inspecting outcomes; accompany selected
  qualitative examples with all-trial quantitative results.
- For imagined-versus-grounded panels, record whether actions are replayed exact
  stored actions or recursively policy-generated. Only the former gives a direct
  exact paired visual residual without additional environment interaction.
- Count grounded rollout, calibration, retry and evaluation transitions separately.
  Stage count alone cannot verify matched interaction budgets.

Do not reuse success-seeking training retries for evaluation. Do not infer Table II
trust validity or the paper's performance claims from these qualitative videos.
