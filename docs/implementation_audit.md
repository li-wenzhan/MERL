# Research-to-code audit

## Camera-ready release audit (2026-10-08)

The final camera-ready main text and appendix supersede the earlier submission
and rebuttal as the mechanism specification. The default public launcher now
selects the lifecycle described in [camera-ready protocol](camera_ready_protocol.md).
The discrepancy table below is retained as the **historical starting point**;
it describes the legacy implementation, not the new default.

Resolved discrepancies include recursive predicted-observation policy input,
stored-action calibration at depths 1–4, frozen residual inference, measured
stage error and 8–32-step horizons, inverse-error chunk replay, separately
normalized branch objectives, and one shared simulator GPU. Added the frozen
simulator plus trust control. The simulator's conditioning frame is now the
last history frame; the old first-future-frame index leaked a target into its
conditioning. Soft proxy supervision and inference now use matching pre-action
pairs and image ranges. Checkpoints include optimizer/scheduler/RNG/trust state.

These are new implementations. Original numerical results, residual networks,
full hyperparameter records and trained comparison checkpoints are not
recovered by editing code. The 100-stage infrastructure example also does not
specify the complete benchmark learning-curve budget: some reported S2T-H values
exceed 100. Original stage limits and checkpoint grids remain needed.

The publication scope is MERL and its component controls; external baseline
ports are intentionally excluded at the authors' request. Separate physical
WM tools are available, but the simulation entrypoint does not implement the
complete physical-robot fixed-data policy adaptation experiment.

### Verified runtime evidence

The 2026-10-08 checks used an isolated Linux copy and local pretrained assets;
existing experiments and checkpoints were preserved. Full logs and fixture
artifacts are retained privately under `tmp_files/release_validation/`.

| Check | Observed result | Scope |
| --- | --- | --- |
| Contract suite | 82 tests pass on Linux; Windows passes with three Linux Bash tests skipped | Includes actual Hydra composition for five modes, masks, gradients, trust, RNG restoration and episode-disjoint physical-data splits |
| Fresh grounded export | Six valid trajectories; 2,551 executed actions and `T+1` observations per trajectory | Actual single-GPU policy collection with the 512-action cap; training data, not held-out performance |
| Actual simulator update | Visual UNet and proxy parameters change; finite losses/gradients | One engineering update with pretrained weights, FP32 trainable storage and BF16 computation |
| No-oracle lifecycle | Depths 1–4, frozen residual predictor, partial 5-step prediction, zero calibration environment calls | Two diffusion steps and reduced residual fitting for software validation; not held-out residual accuracy |
| Simulator conditioning | Changing future GT does not alter the actual UNet conditioning channels | Verifies the past-only anchor boundary |
| Simulator checkpoint | Strict full-weight, optimizer, scheduler, residual and RNG roundtrip; reconstructed stage context | Repeated request seeds preserve predictions across request order and restoration |
| Actor stage | Real rollout, categorical old log-probabilities, 48-chunk update and sharded/runtime save complete | One H100, BF16 model storage, 32-action cap; about 14 minutes including cold startup |
| Actor resume/evaluation | Restore stage 1, continue stage 2, save again, evaluate two held-out states and export videos/PNGs/trajectories; exit 0 | Same engineering precision/cap; about 18 minutes. Both short trials fail with no invalid rollouts |
| Auxiliary HDF5 WM tool | Train, episode-disjoint validation and model/optimizer checkpoint save complete; finite losses | Simulation-derived HDF5 fixture and auxiliary synthetic targets; not physical-robot experimental evidence |

The short actor fixture has zero GRPO gradient because all candidates fail.
It establishes execution/checkpoint plumbing, not a learned policy improvement.
The simulator update peaks at approximately 34.2 GiB on the tested H100.
Default fused SDPA produced non-finite visual gradients despite a finite loss;
math SDPA passed the actual-model forward/backward update and is now used for
simulator optimization, including gradient-checkpoint recomputation.

Single-H100 validation cannot certify three-rank FSDP, full FP32 actor training,
actor-to-simulator RPCs across the complete four-GPU allocation, or a 100-stage
run. These remain ACP runtime gates. Start with the full-budget one-stage
command in [the runbook](h100_runbook.md), retaining the configured simulator
update and real rollout budgets. Passing it still does not reproduce benchmark
numbers or demonstrate method superiority.

## Historical audit of the pre-release implementation

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

## LIBERO-PRO protocol caveat

`EnvironmentReplacePerturbator.perturb` currently hardcodes the replacement to
`living_room_table`; its random candidate selection is commented out. Tasks already
using that environment may receive no environment shift. The startup cleanup keeps
this behavior to preserve the existing protocol. Do not describe these runs as
uniformly random environment perturbations. Before an OOD robustness claim, define
an explicit per-task perturbation panel, check that each intended shift changes the
BDDL semantics, and reuse frozen BDDL/init assets across all methods. The official
initial-state generator has no seed argument; record asset hashes as well as the
configuration and source revision.

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

Hardcoded W&B and Hugging Face credentials were found in tracked runtime,
launcher and vendored sources and removed from current files. They remain in
the pre-existing Git history and require revocation/rotation by the account
owner; no history rewrite has been performed. Runtime credentials must come
from the process environment.

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
