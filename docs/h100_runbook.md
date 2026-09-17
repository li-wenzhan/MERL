# CCI preparation and ACP execution

CCI is the single-H100 development environment. Use it for code, assets and small
runtime checks. Actual experiments run as ACP jobs on a node with four H100 80GB
GPUs. CCI GPU inventory is not an ACP resource-allocation problem.

## Unified launcher

`python -m merl.launch` composes Hydra directly from `configs/launch_profiles.json`.
Duplicated 1/3/4-GPU launch scripts were removed. Algorithm settings were extracted
from the former four-GPU scripts at revision `8de7bb6`; the profiles retain their
legacy differences (including learning rate, temperature and confidence guards).
They do **not** activate recursive no-oracle imagination or establish matched
paper budgets. See [implementation audit](implementation_audit.md).

MERL/MBRL reserve three actor GPUs plus one WM trainer GPU by default. MFRL uses
three actor GPUs, retaining the same actor parallelism; the fourth is unused.
WM inference copies may still occupy actor GPUs. `--actor-gpus` scales per-rank
batch settings and is useful for CCI real-only checks.

```bash
python -m merl.launch --mode MERL \
  --sft-checkpoint /models/openvla-oft --wm-checkpoint /models/ctrl-world.pt \
  --experiment merl_smoke_001 --check
```

`--check` verifies assets and resolved config without Ray/GPU allocation.
`--dry-run` only prints the command. `--render-check` additionally resets a real
environment. For ACP execution replace `--check` with `--smoke`, then remove
`--smoke` only after verifying a real update. Smoke means task 0, 16 environment
steps, one outer update, one WM inner step and two diffusion inference steps;
Accuracy filtering is disabled in smoke runs so all-failed short rollouts can
reach the update path; zero GRPO signal is possible. This tests plumbing, not
success or simulator fidelity.

Select `--job evaluate` for real-only evaluation of an exported actor or
`--job collect` with MFRL for fixed WM evaluation data. The default `--wm-eval off`
allows pipeline bring-up without fixed shards, and produces no fixed WM metrics.
Use `--wm-eval fixed --shared-wm-eval /data/wm_eval` after collecting both splits.

Each fresh run writes its resolved config, command, source hashes, Git revision,
package versions, GPU information and final status into `launch_manifest.json`,
plus a complete `run.log`. These are not a full RNG/optimizer resume checkpoint.
Source SFT weights are symlinked into run-specific actor assets, never overwritten.
Existing run directories and nonempty collection splits are rejected.

## Private ACP job payload

The deployment script is `tmp_files/acp_merl.sh` (intentionally ignored by Git).
It selects the supplied shared filesystem repo, environment and model paths, then
calls the maintained launcher. Submit it as the ACP job command; platform-specific
submission/resource syntax must come from the actual ACP configuration.

```bash
bash tmp_files/acp_merl.sh MERL train merl_smoke_001 --smoke
bash tmp_files/acp_merl.sh MBRL train mbrl_smoke_001 --smoke
bash tmp_files/acp_merl.sh MFRL train mfrl_smoke_001 --smoke
# CCI asset check; no four-GPU allocation needed.
bash tmp_files/acp_merl.sh MERL train asset_check_001 --check
```

Override `MERL_REPO`, `MERL_ENV`, `SFT_CHECKPOINT`, `WM_CHECKPOINT`, `OUTPUT_ROOT`,
`ACTOR_GPUS` or `SHARED_WM_EVAL` via environment variables. Never replace ACP's
`CUDA_VISIBLE_DEVICES` with physical indices from CCI. EGL is the default;
`scripts/runtime/libero_glx_runtime.sh` is available for explicit GLX setups.

## Validation gates

1. CPU contracts: `python -m unittest discover -s tests -v`.
2. `PYTHONPATH=$PWD python scripts/verify_merl_memory_contract.py`.
3. Asset/config checks with `--check`; strict simulator load with
   `scripts/preflight_world_model_backbone.py --config configs/wm_online_config.py
   --checkpoint /models/ctrl-world.pt --load-model`.
4. Environment reset/step and finite SFT actions on its saved observation.
5. Real-only closed-loop CCI smoke, then ACP actor/WM worker initialization.
6. One ACP update: finite loss/gradients, valid token masks, actual transition
   counts and checkpoint saving. A process starting is not an update passing.
7. Common evaluation panel, fixed WM dataset and larger experiments.

The supplied Ctrl-World checkpoint strictly matches all 2,665 state entries.
The WM trainer now raises on checkpoint mismatch instead of silently continuing.
This proves structural compatibility, not reward calibration or predictive quality.
When loading this full checkpoint, the reward backbone does not download redundant
ImageNet weights; its parameters come from the checkpoint.

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

## Evaluation integrity fixes

Task subsets are filtered in the dataset before dispatch; unexpected task IDs
raise instead of being silently relabeled. Evaluation order is deterministic and
the final partial batch is retained. Distributed padding results are excluded
from metrics and counted as `validation/padding_rollouts`. Invalid rollouts fail
the maintained evaluation job instead of silently reducing its denominator.
Collection requires a trial count divisible by actor ranks; use one actor for
small fixed datasets. These changes can change results relative to older scripts.
