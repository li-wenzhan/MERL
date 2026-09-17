# No-oracle trust: implementation and integration contract

## Status

`merl/trust.py` and `merl/imagination.py` implement the residual predictor and
Appendix A trust equations with CPU-testable boundaries. They are **not yet wired
into `fit_wm_v5`**. Existing training launchers still use the legacy mirror path.
Do not label their output as no-oracle MERL or as a reproduction of rebuttal Table II.
This implementation has no recovered training data/checkpoint corresponding to
the reported paper numbers. Unit tests establish contracts, not empirical accuracy.

## Data boundary

1. After updating **both** simulator components, replay already stored grounded
   actions from the corresponding grounded anchor. Do not query a live environment.
2. Encode anchor, predicted and grounded frames with one frozen encoder and
   normalization. Spatially pool each frame to `[D]`. Use exactly the same encoder
   for later imagined inputs. Latent MSE and proxy MAE use a masked temporal mean.
3. Build `CalibrationBatch` only from the calibration trajectory partition. Exact
   anchor identifiers and action tensors must agree. Matching task/trial alone is
   insufficient. Provenance identifiers are assertions supplied by the exporter;
   the API cannot verify that an upstream exporter used the claimed observations.
4. Call `ResidualPredictor.fit(..., stage=s)` after the simulator update. A successful
   fit atomically replaces the previous snapshot. It resets and deterministically
   fits a small two-output MLP on CPU; normalization and network weights are saved.
   Interrupted fits leave the old snapshot intact and must be rerun.
5. Freeze the snapshot for all imagination and policy updates at that stage.
   `predict()` accepts only inference features and requires matching stage and
   simulator revision. A revision identifies the video predictor, reward proxy,
   encoder and preprocessing contract together.
6. A new simulator revision invalidates old residual predictions/replay scores.
   Recompute scores or discard those chunks; do not silently reuse stale trust.

Predictor inputs are anchor latent, predicted latent mean/variance, the ordered
action chunk, valid-step mask, one-based depth and numeric stage context. Outputs
are nonnegative estimates of visual MSE and proxy MAE. Future GT, terminal success,
and actual recursive continuation never enter the inference feature object.

Calibration currently uses exact **depth-one grounded windows** only. Predictions
at depths 2--4 are consequently out-of-distribution extrapolation and require
held-out validation. Including depth as an input does not establish depth robustness.
The one-stage refit also does not learn variation across constant stage context.
The paper/rebuttal does not specify the residual network, optimizer, encoder pooling,
or their hyperparameters; the choices here are explicit engineering choices.

## API

```python
from merl.trust import ResidualPredictor, StageScheduler, trust_scores

predictor = ResidualPredictor(seed=0)
fit_metrics = predictor.fit(calibration_batch, stage=stage)
predictor.save("outputs/residual.pt")

# measured errors are available only for exact stored calibration pairs
measured = trust_scores(calibration_batch.residuals())
ratio, horizon = scheduler.update(float(measured.error.mean()))

# future-free features from a recursive imagined candidate
residuals = predictor.predict(features, simulator_revision=revision, stage=stage)
scores = trust_scores(residuals)
```

Persist `dataclasses.asdict(scheduler)` alongside the policy/simulator checkpoint.
Restore with `StageScheduler(**state)`. The frozen predictor checkpoint restores
normalizers, feature signature, stage, revision, training trajectory IDs, and weights.
Feature/context conventions must also be part of the run manifest.

`merl.imagination.imagine()` accepts a stored anchor, policy callback, simulator
callback, encoder and frozen predictor. It queries the policy on the last predicted
observation, masks a partial final chunk and makes no environment calls. Simulator
adapters must manage historical frames/actions using only stored/predicted data.
They must supply deterministic latent encoding for calibration versus inference.
The current API supports one candidate trajectory at a time; collect multiple
trajectories before normalizing replay probabilities over the full chunk population.

The current production adapters, FSDP lockstep execution, action-token/old-log-prob
retention, history-conditioned Ctrl-World calls and actor `DataProto` conversion
remain integration work. A callback must not hide live environment access.

### Stored-window exporter and history boundary

`merl.stored_calibration` implements offline pair construction and recursive
history management. `StoredTrajectory` requires **T+1 observations for T executed
actions**, explicit post-action proxy targets, a complete-trajectory split and an
episode ID. It does not guess alignment from padded rollout videos. Window
`start=k` uses `o[k]` as anchor, replays `a[k:k+C]` and labels against
`o[k+1:k+C+1]`. Its H historical observations are the outcomes of the H preceding
actions, ending at the anchor. Insufficient history is rejected, not synthesized.

```python
from merl.stored_calibration import (
    HistorySimulator, build_calibration_batch, context_at,
)

# step(past_context, executed_actions) -> (predicted_frames, proxy_scores)
# encode(frames[N,...]) -> frozen deterministic latents[N,D]
batch = build_calibration_batch(
    calibration_episodes, windows=[(0, 8), (0, 16)],
    history_size=8, chunk_size=8, stage_context=stage_context,
    simulator_revision=revision, step=step, encode=encode,
)
predictor.fit(batch, stage=stage)

# Start each recursive candidate with its own history object.
context = context_at(grounded_episode, start=8, history_size=8)
simulator = HistorySimulator(context, step, encode)
chunks = imagine(
    anchor=context.anchor, instruction=context.instruction,
    horizon=horizon, chunk_size=8, policy=policy,
    simulator=simulator, encode_anchor=simulator.encode_anchor,
    predictor=predictor, stage_context=stage_context,
    simulator_revision=revision, stage=stage,
)
```

Only past context and exact stored actions reach the simulator while constructing
calibration pairs; future observations/labels are accessed separately afterward.
Recursive history then advances exclusively through predicted frames/actions.
Predicted and grounded frames use the same supplied encoder, rather than mixing
diffusion latents with independently sampled VAE latents. The exporter rejects
non-calibration trajectories before any simulator call and masks partial windows.
Tests change future labels while holding the past fixed and verify identical
prediction features, as well as action/history alignment at recursive depths.

The callback adapters are still required: convert policy gripper coordinates to
executed coordinates **once**, preserve policy tokens separately, define the
deterministic visual encoding and adapt Ctrl-World tensor/image conventions.
This boundary implementation is not an activation switch for the legacy trainer.

## Equations and policy contract

- Combined error: `alpha_obs * E_obs + alpha_proxy * E_proxy`.
- Replay: `softmax(-alpha_p * log(error + zeta_p))`, equivalent to Eqs. 21--22.
- Weight: `clamp(exp(-eta_w * error), w_min, 1)`.
- Stage: EMA of **measured error**, then inverse-power confidence, affine ratio
  and floored horizon. Defaults follow Appendix A: ratio `[.05, .95]`, horizon
  `[8, 32]`; unreported scales remain configurable.
- Sample chunks with replacement using a dedicated generator. Do not apply
  importance correction toward uniform replay: Appendix A.5 intentionally biases
  admission toward low-error chunks.
- `mixed_policy_loss` accepts per-chunk clipped loss sums, averages each branch
  independently and multiplies imagined losses by detached trust weights. It divides
  by chunk count, **not sum of weights**. Preserve GRPO group membership upstream.

The legacy scheduler/replay/actor have different contracts; plugging a scalar
predictor into them alone does not implement the full method.

## Calibration file

Save a tensor/primitive-only dictionary with `torch.save`:

```text
features:
  anchor_latent: float [N,D]
  imagined_latents: float [N,C,D]
  actions: float [N,C,A]
  depth: integer [N] (all 1 for exact calibration)
  stage_context: float [N,S]
  valid_steps: bool [N,C]
grounded_latents: float [N,C,D]
predicted_proxy: float [N,C]
target_proxy: float [N,C]
grounded_actions: float [N,C,A]
trajectory_ids: list[str]
anchor_ids: list[str]
predicted_anchor_ids: list[str]
simulator_revision: str
split: calibration
```

```bash
python -m merl.calibrate_trust --input windows.pt --output outputs/residual.pt --stage 1
python -m unittest discover -s tests -v
```

The CLI writes training-only fit diagnostics and a frozen checkpoint. It does not
interpret training MAE as held-out trust validity.

## Required experimental evidence

Claim: future-free trust predicts local error and improves policy utility.

Split by complete trajectory **before** generating windows. Keep recursive matched
futures in a separate evaluator; score and freeze predictions before opening labels.
Report depths 2 and 3--4 separately. Record Pearson correlation of predicted/observed
error, low-error rank AUC, top-30% precision and trajectory-bootstrap intervals.
Predeclare the empirical-reliability event and threshold for ECE; continuous
`exp(-error)` is not automatically a calibrated probability. Do not infer a missing
metric definition from the table values.

Compare uniform, visual-only, proxy-only and combined trust with equal candidate
counts, action horizons, optimization budget and initial checkpoints. Policy utility
needs matched-count Top/Middle/Bottom-bin refinement with multiple seeds. A passing
unit test or attractive selected video cannot establish any of these claims.
