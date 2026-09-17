# Research-to-code audit

Baseline inspected: `ceefa270d972c8ee2e466e1170cab689c8be008a`.
Sources: local submission PDF (28 pages) and one-page CoRL rebuttal under
`tmp_files/`. Those private materials are ignored by Git and are not republished.

## Critical discrepancies

| Design expectation | Current production implementation | Consequence / verification gate |
| --- | --- | --- |
| Recursive policy calls use imagined observations; calibration reuses stored data | `RobWMActorRolloutRefWorker.generate_sequences` routes `use_wm` to `_generate_minibatch_libero_evolving`; that method submits actions to the environment and constructs `new_inputs` from its results | This is a grounded mirror, not recursive imagination. Audit all environment transitions, including retries, before claiming the rebuttal's `6s` trajectory budget |
| No-oracle residual predictor frozen after simulator update | Existing `wm_obs_error` compares generated and environment frames; no fitted predictor was found in this path | New standalone core implements the boundary; production exporter, lifecycle and actor bridge remain required |
| Local proxy error against success-to-go targets | `wm_done_error` compares finish-step timing derived from the environment; this is not proxy MAE | Produce explicit aligned proxy targets and measured residuals on stored windows |
| Low-error chunk gets higher replay probability | `_compute_wm_rollout_confidence` increases priority with uncertainty | New core implements inverse-error probability. Legacy training remains different and cannot be represented as the paper implementation |
| Chunk admission and per-chunk loss weights | Existing pools store trajectories, with one confidence weight per sample, and general replay uses importance correction | Keep chunk identity through sampling, GRPO grouping, token masks and actor update; test gradients and admission separately |
| Stage scheduler: EMA error, inverse-power confidence, horizon `C..4C` | Existing path mixes training-noise loss, exponential confidence, cooldown and weak-update logic; four-GPU MERL launcher uses horizons 128..192 | Use the paper scheduler explicitly for reproduction; do not conflate debug defaults with paper settings |
| Full MERL uses imagined proxy progress | Four-GPU MERL launcher disables proxy rewards, requires successful real anchors and caps imagined weight | This is a conservative alternative behavior, not the full method described by the paper |
| Matched budgets and protocol | Mode scripts differ in batch sizes, optimizer/weight guards and collection paths | Build a shared resolved protocol and record actual environment interaction, not only online-step count |
| Reliable initialization and resume | Unified launcher now requires and enables an explicit WM checkpoint; strict load failures abort. `WorldModelTrainer.save_world_model` still saves model weights only | Verify actual loaded assets; optimizer/RNG/scheduler state need an explicit full-resume contract |

## Architecture map

- `verl/trainer/main_ppo.py`: Hydra entrypoint, mode selection, Ray orchestration.
- `verl/trainer/ppo/ray_trainer.py`: collection, WM updates, scheduling, replay,
  reward routing, advantages, policy updates and checkpoint orchestration. Its
  size and nested helper coupling make direct large-scale restructuring risky.
- `verl/workers/fsdp_workers.py`: actor/FSDP lifecycle plus a separate WM trainer.
  WM inference copies also exist in rollout workers; the reserved trainer GPU does
  not imply all WM computation and memory are isolated from actor GPUs.
- `verl/workers/rollout/rob_rollout_wm_pro.py`: environment service, real/mirror
  rollouts, policy input conversion, WM inference and video serialization.
- `modules/ctrl_world`: simulator, reward proxy, stored-trajectory datasets.
- `merl`: new isolated, dependency-light research mechanisms with tests.
- `real_world`: separate robot-data/inference workflows; not interchangeable with
  the LIBERO-PRO online experiment.

## Cleanup policy

The first patch also fixes environment-step versus policy-chunk token counting
in both WM output builders, aligns proxy reward placement with the valid prefix,
and resolves a conflicting rollout instruction merge exposed by the existing
contract verifier. These changes alter effective training signals; previous runs
are not directly comparable without recording the code revision.

Hardcoded W&B credentials were found in the tracked runtime JSON and launcher
scripts and removed from the current files. They remain in the pre-existing Git
history and require revocation/rotation by the account owner; no history rewrite
has been performed. Runtime credentials must come from the process environment.

Do not delete vendored `verl`/OpenSora modules or `*_old.py` merely because their
names look obsolete. Dynamic imports, config entrypoints and standalone offline
tools require dependency checks first. Existing upstream licenses must be retained.
The repository has third-party licenses but no root license selecting terms for
MERL's own contributions; the authors must choose that before an open-source release.

First consolidate the user-facing launch interface, document active versus legacy
paths, and protect behavior with tests. Then extract pure scheduler/replay/reward
functions from the trainer one at a time. Archive/remove a file only after checking
imports, launch/config references and a representative runtime regression.

## Hardware observation

On 2026-09-17 the supplied SSH workspace exposed one H100 80GB (`nvidia-smi`,
also only one numbered `/dev/nvidia*` device). It had the same clean baseline
commit and PyTorch `2.12.0+cu130`. The user confirmed this is CCI for development; actual four-GPU execution
runs on ACP. Single-device preflight cannot validate multi-rank FSDP collectives.
