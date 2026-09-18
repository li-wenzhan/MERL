# Online MBRL pilot: 2026-09-18 02:06:51 UTC

## Evidence and interpretation

The four-H100 run completed distributed initialization, one actor optimizer step,
two WM optimizer steps, checkpoint saving and rollout-worker synchronization.
It exited with code 1 during final evaluation after 2,472 seconds (41m12s).

| Observation | Value | Interpretation |
| --- | --- | --- |
| Actor gradient norm | 0.2528 | A finite, nonzero update occurred |
| Imagined actor tokens | 2,688 | Imagined inputs reached actor optimization |
| Imagined actor weight | 1.0 | Consistent with the no-trust baseline |
| Real training samples | 2 valid out of 6 requested | Training coverage was incomplete |
| WM optimizer steps | 2 | Updated weights exist; no evidence of convergence |
| Final evaluation | 1 valid success, 2 invalid environments | No comparable success rate |

The configured six outer steps were an upper bound; the 900-second soft training
budget stops between iterations. It excludes initialization and final evaluation.
Several `timing/*` values are NaN because `Timer.last` is read before its context
exits; these are instrumentation defects, not evidence of NaN model gradients.
Use the outer wall time and explicit completed timer messages until repaired.

## Root cause and fix

The renderer resolver always returned EGL device 0. Ray restricted actor ranks
to `CUDA_VISIBLE_DEVICES=0`, `1`, and `2`. robosuite 1.4.1 checks that the explicit
`MUJOCO_EGL_DEVICE_ID` belongs to that visibility list at import time. Consequently,
ranks 1 and 2 failed both real training collection and final evaluation.

Auto selection now uses the first numeric identifier in the rendering child's
effective visibility list. Explicit conflicting selections fail clearly. The
actor's CUDA visibility is not widened. The former preflight assertion that every
rank must use local EGL device 0 was removed. The online baseline also rejects a
filtered real batch smaller than the requested sample budget before actor/WM
updates, instead of accepting the two surviving samples.

Validation: 65 Linux tests pass, including nonzero/reordered visible device IDs
and incomplete real-batch rejection. A real single-H100 `libero_10_env` reset,
RGB render and action step pass. Nonzero GPU rendering still requires the next
four-GPU ACP run; the CCI exposes only device 0.

## Experiment decision

Preserve the failed run's actor `global_step_0` and WM `global_step_1` checkpoints,
logs and videos as diagnostic artifacts. Do not resume them for the clean baseline
or report the invalid evaluation as either 1/3 or 100% success.

Start a fresh normal short training run from the common SFT and initial WM weights.
Require all six real samples, actual actor/WM updates, checkpoint synchronization,
and three valid held-out evaluations before accepting the run. Then prioritize
MERL to obtain a second updated WM checkpoint. Compare both against the existing
frozen WM using identical stored actions, history, diffusion steps and seeds.

For resource planning, the failed run is a measured 41-minute reference, not clean
throughput: successful rendering on all ranks adds work, while failing/retrying
environments also consumed time. Initially reserve 45--75 minutes per online mode;
two online modes may require 90--150 minutes. Record timings from the first clean
run before expanding task or seed coverage. A one-step/two-step pilot establishes
execution only, not a convincing method ranking or trust-mechanism contribution.

Follow-up engineering priority is accurate per-stage timers, then profiling real
collection, WM rollouts and full checkpoint I/O. Do not reduce final evaluation
horizons or cherry-pick successful videos to compensate for the short training
budget.
