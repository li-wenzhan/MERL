# Four-H100 preparation and qualitative comparison

## Current execution gate

The supplied workspace currently exposes one H100. LIBERO-PRO and the requested
OpenVLA-OFT SFT checkpoint have now been acquired. SVD/CLIP backbone file preflight
passes, but the WM warm-start is disabled. Single-device preflight does not establish
distributed training readiness. See the asset and validation notes below.

Allocate four visible 80GB GPUs and select the intended trained Ctrl-World
checkpoint. Confirm which simulator checkpoint contains a
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
   python scripts/preflight_openvla_oft.py --checkpoint /models/openvla-oft
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

## Acquired assets and single-device checks (2026-09-17)

- LIBERO-PRO: official `https://github.com/Zxy-MLlab/LIBERO-PRO.git`, revision
  `eafdb809426b13153aa1e4c42d6601844217dfec`, installed under
  `/mnt/afs/task3_2/L202500276_lwz/benchmark/LIBERO-PRO`.
- SFT: ModelScope `VesperLee/Openvla-oft-SFT-libero10-trajall`, installed under
  `/mnt/afs/task3_2/L202500276_lwz/models/Openvla-oft-SFT-libero10-trajall`.
  All 21 snapshot files downloaded; four safetensors shards match the index's
  982 tensor names. Normalization key: `libero_10_no_noops`.
- Single-image SFT inference passes on the real task-0 observation: finite `[8,7]`
  actions, no missing/unexpected parameters, 15,450,736,128 peak allocated GPU
  bytes. This used torch 2.12.0+cu130 / transformers 4.57.6; the vendored model
  warns that its original reference versions differ. Closed-loop behavior and
  FSDP remain separate validation gates.
- LIBERO-PRO compatibility passes with NumPy 1.26.4 and robosuite 1.4.1.
  A real `libero_10:0`, trial 0 environment resets, renders 256x256 RGB through
  EGL and executes two zero-motion actions. This checks the original task, not
  OOD asset generation, policy quality or success rate.
- Initial states use scoped NumPy reconstruction allowlisting with
  `weights_only=True`; MERL does not patch the external checkout or globally
  disable restricted loading. The compatibility preflight initializes the
  training stack before robosuite to avoid a reproduced native import-order
  crash in this torch 2.12 / Triton 3.7 / NumPy 1.26 environment.

Reproduce environment evidence into a **new** output directory:

```bash
ROBOT_PLATFORM=LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python scripts/preflight_libero_env_service.py \
  --config configs/evaluation_config.yaml --timeout-s 240 \
  --step-count 2 --output-dir outputs/env_smoke_001
```

The output contains rendered frames and `result.json`. Optionally test the SFT
model on the first saved frame, using the matching task instruction:

```bash
python scripts/preflight_openvla_oft.py --checkpoint /models/openvla-oft \
  --image outputs/env_smoke_001/frame_000.png \
  --instruction 'put both the alphabet soup and the tomato sauce in the basket' \
  --output outputs/vla_smoke_001.json
```

This uses local MERL model classes, one camera, discrete actions, eager attention,
BF16 and no proprioception, and leaves downloaded checkpoint files unchanged.
It checks finite `[8,7]` actions and refuses missing/mismatched model parameters;
it is not the distributed actor or a closed-loop policy evaluation.
