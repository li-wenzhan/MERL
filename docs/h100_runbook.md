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
  warns that its original reference versions differ. Single-rank closed-loop
  validation is recorded below; multi-rank FSDP remains an ACP validation gate.
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

## Simulator forward smoke

```bash
python scripts/preflight_world_model_backbone.py \
  --config configs/wm_online_config.py --checkpoint /models/ctrl-world.pt \
  --image outputs/env_smoke_001/frame_000.png \
  --instruction 'put both the alphabet soup and the tomato sauce in the basket' \
  --output outputs/wm_forward_smoke_001.json
```

The command strictly loads the supplied checkpoint, repeats the input as history,
uses zero actions and runs two diffusion steps followed by reward prediction.
The image/instruction must match. This is a structural smoke test, not an aligned
stored trajectory or simulator-quality measurement. On the CCI checkpoint it
produced finite `[8,192,320,3]` frames and eight reward probabilities, with
5,295,523,840 peak allocated GPU bytes. `--load-model` alone keeps loading on CPU.

LIBERO-PRO's generated initial states use a NumPy-pickle ZIP format different from
original Torch archives. MERL supports both with scoped reconstruction allowlists;
no global unrestricted pickle loading is enabled. The official generator exposes
no seed argument: freeze and hash generated BDDL/init files for cross-mode reuse
instead of claiming byte-for-byte regeneration from a seed.

## Freeze the evaluation assets

After the selected perturbation assets are fully generated, validate and fingerprint
them once. Do not rerun generation independently for each mode.

```bash
python scripts/manifest_libero_assets.py \
  --bddl-dir /benchmark/LIBERO-PRO/libero/libero/bddl_files/libero_10_env \
  --init-dir /benchmark/LIBERO-PRO/libero/libero/init_files/libero_10_env \
  --expected-tasks 10 --min-trials 50 \
  --config configs/evaluation_config.yaml --source-root /benchmark/LIBERO-PRO \
  --output outputs/libero10_env_panel_001.json
```

The manifest records per-file hashes, task/state counts, unique state counts,
configuration hash and LIBERO-PRO Git revision. It rejects missing or extra tasks,
invalid state arrays and insufficient trials. Keep the manifest with all comparison
runs and verify hashes before reuse. A fingerprint records identity; it does not
make the external files immutable or prove the perturbation is a meaningful OOD shift.

## Dataset requirements

Online RL from the supplied SFT and Ctrl-World checkpoints does not require the
full LIBERO demonstration HDF5 dataset. It needs the simulator assets, BDDL tasks
and matching initial states. Demonstrations are needed when repeating SFT or
offline world-model pretraining. Fixed WM evaluation shards are another dataset:
collect them once through `--job collect`, freeze them and exclude them from
training. Default startup profiles leave fixed WM evaluation disabled.
The upstream warning about a missing `libero/datasets` directory refers to
demonstrations; the validated online environment path works without that directory.

The current `libero_10_env` panel contains ten task/state pairs with fifty valid
initial states each. `scripts/manifest_libero_assets.py` checked all 500 states
and recorded file hashes in
`outputs/preflight_20260917/pipeline/libero10_env_panel.json` on CCI.

## Closed-loop evidence and remaining gates

The ACP payload was also exercised on CCI with one actor:

```bash
ACTOR_GPUS=1 OUTPUT_ROOT=outputs/preflight_20260917/pipeline \
  bash tmp_files/acp_merl.sh MFRL evaluate cci_closed_loop_001 --smoke --trials 1
```

That run used clean revision `59efd6c`, exited with code zero and recorded one
valid real-environment trial, no invalid rollouts and an MP4. Its zero success
rate over sixteen steps is a plumbing result, not a benchmark estimate. Inspect
the run's `launch_manifest.json` and `run.log` for the resolved configuration and
provenance. Policy evaluation now uses the same real-environment path for all
three mode names, even when the training mode would enable the simulator.

The same command with `MBRL evaluate cci_mbrl_closed_loop_001` also passed at
clean revision `69cd2d5`: one valid trial, zero invalid rollouts, exit code zero
and `world_model.enable=false`. Both checks use the same SFT weights; they verify
mode dispatch and evaluation plumbing, not separately trained algorithms.

Both training loops now honor `trainer.total_training_steps`; `--smoke` caps the
run at one completed outer step. This limit counts completed loop iterations,
not gradient steps or environment interactions. For a checkpoint-saving smoke,
append `-- trainer.save_freq=1`. Record optimizer-step metrics as well as the
process exit status, since a guarded or zero-signal update can leave weights
unchanged. The regression suite currently contains 40 passing tests on both the
local CPU environment and CCI.

Actor updates now reject non-finite gradient norms and propagate optimizer errors.
AdamW state compatibility is handled before stepping; a potentially partial update
is never retried or silently reported as successful.

A CCI full-parameter training attempt (`cci_actor_update_001`, revision `a661e7d`)
exposed a missing MFRL horizon override: workers used 512 steps despite the smoke
configuration requesting 16. The driver was terminated intentionally; this run
is not a successful training check. Revision `4352241` forwards the configured
horizon, with a regression test at the actual metadata-dispatch boundary. The
corrected training path still needs an ACP runtime check.

Do not use single-rank CCI to validate the default full-parameter Adam update:
7.54B FP32 parameters, gradients and two optimizer moments require approximately
120 GB before reference weights and activations. Existing optimizer offload moves
state back to the GPU for updates; it is not CPU optimizer execution. Use the
three-actor ACP layout to validate the original precision and optimizer settings.

Before a full ACP run, verify multi-rank actor updates, the dedicated WM update
and synchronization, and checkpoint saving on the allocated four GPUs. The
one-step MERL smoke starts in the legacy warmup phase; it does not exercise every
imagined replay path. Recursive no-oracle trust is still an isolated tested core,
not an integrated production training mechanism. None of these smoke results
establishes the paper's comparative performance claims.
