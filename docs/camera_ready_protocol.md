# Camera-ready implementation contract

The final paper is the research specification for `--protocol camera-ready`,
the default public entrypoint. The earlier `fit_wm_v5` loop is reachable only
through `--protocol legacy`. New implementation correctness and numerical
reproduction of the paper are separate verification tasks.

## Lifecycle and module boundaries

| Paper mechanism | Implementation | Contract |
| --- | --- | --- |
| Grounded budget | `merl/paper_trainer.py::fit`, real rollout exporter | Three prompt groups, two trajectories per group; exactly six valid episodes; at most 512 executed commands each |
| Executed action / observation alignment | `merl/paper.py::save_grounded_trajectory` | `o[t]` precedes `u[t]`; `o[t+1]` follows it. Gripper converted once for execution; categorical tokens stored separately |
| Evolving simulator | `merl/paper_simulator.py::update` | Grounded one-chunk visual supervision plus soft binary progress classification; no self-forcing auxiliary loss |
| Truncated target | `merl/trust.py::success_to_go`, `merl/proxy.py` | `z * gamma_q ** min(T-1-t, H_q-1)`; do not round soft targets to binary labels |
| Proxy conditioning | `training_window`, `step`, real-world adapter | Pre-action observation and corresponding executed command, RGB in `[-1,1]` |
| Stored calibration | `merl/stored_calibration.py::build_calibration_batch` | Stored anchor/history/actions only reach prediction; future GT/targets are opened separately for labels. Recursive replay supports depths 1–4 |
| Common visual residual | `PaperSimulator.encode` | Same frozen VAE posterior mean and fixed spatial pooling for GT and decoded predictions; no sampled/raw diffusion-latent mismatch |
| Frozen no-oracle estimator | `merl/trust.py::ResidualPredictor` | Refresh after simulator update; freeze through imagination/actor update; require matching stage and simulator revision |
| Stage-level trust | `StageScheduler`, `PaperSimulator.prepare_stage` | EMA measured visual error plus grounded pre-action proxy error; inverse-power confidence, affine ratio and floored horizon |
| Recursive policy input | `merl/paper_rollout.py::generate_imagination` | Re-query policy on the latest predicted RGB; no LIBERO environment in this branch |
| Chunk-level trust | `trust_scores`, driver replay | Normalize inverse-error priority over candidate chunks; exponential clipped weights; no uniform-replay importance correction |
| Mixed GRPO objective | `merl/paper.py`, `dp_rob.py::update_paper_policy` | Valid categorical-token sums per chunk; independent real/imagined chunk-count denominators; detached advantages/weights |
| GPU allocation | `fsdp_workers.py` paper RPCs | Three FSDP actor GPUs, one shared simulator GPU; no simulator copies on actor/reference GPUs |
| Stage resume | `merl/checkpoint.py`, worker runtime RPCs | Model, Adam, LR schedule, RNGs, sampling generator, trust EMA and predictor; publish completion only after all saves succeed |

Invalid or incomplete grounded collection aborts the stage rather than silently
reducing a control's interaction allowance. Final episode length can differ
through success/termination; actual transitions are recorded separately.
Partial imagined chunks mask both temporal supervision and action-token losses.

## Trust and optimization

For each candidate chunk, aggregate predicted residuals as
`e = alpha_obs * E_obs + alpha_proxy * E_proxy`.
Admission uses `(e + epsilon)^(-priority_exponent)` normalized over all candidates;
loss weights are `clamp(exp(-weight_eta * e), weight_min, 1)`.
These two mechanisms intentionally bias updates toward trusted experience.

The stage scheduler measures visual error on depth-one predictions and proxy
error on grounded pre-action pairs. Chunk proxy labels instead compare imagined
pre-action scores against stored success-to-go targets. These are distinct
quantities, as required by the paper.

GRPO group statistics are computed before replay sampling. Real groups compare
terminal environment success; imagined groups compare local mean proxy progress
at the same anchor and depth. The latter aggregation is an explicit engineering
instantiation of the paper's unspecified progress-to-advantage conversion.
Repeated anchor draws have distinct candidate IDs, RNG seeds and artifacts.

The camera-ready policy objective uses one symmetric clipping threshold
(default 0.2); the legacy asymmetric 0.2/0.28 bounds are not its default.
Actor learning rate (5e-6) and rollout temperature (1.2) are matched engineering
defaults across all component controls, not recovered original experiment records.

Actor parameters and Adam storage use FP32 by default with BF16 compute. The
online simulator likewise keeps trainable parameters/Adam moments in FP32;
frozen backbones and inference computation use BF16. In the distributed actor
objective, global branch coefficients are multiplied by world size once because
FSDP averages gradients. There is one optimizer step per completed stage.

Simulator optimization uses PyTorch math SDPA throughout forward and backward,
including checkpoint recomputation. The tested H100 runtime's default fused
attention produced non-finite visual gradients with a finite loss; math SDPA
passed the actual-model update. Non-finite losses or gradients abort before
optimizer steps and cannot become completed checkpoints.

## Explicit defaults and original records

`configs/camera_ready.json` is the release's inspectable configuration. The
paper specifies six trajectories, the 3,072-transition allowance, 8x7 chunks,
one RGB/no proprioception, the stage mapping bounds and trust equations.
It does not recover the original residual architecture, optimizer settings,
spatial pooling, all error scales, calibration window count, candidate/batch
counts, or exact proxy-progress aggregation.

The release therefore supplies a two-output CPU MLP, fixed 4x6 VAE pooling,
explicit fitting and replay defaults. These implement the mechanism but do not
establish that original paper numbers have been reproduced. The default
100-stage run follows the paper's infrastructure example, not a recovered final
benchmark budget. The final evaluation grid, seed list, asset panel and full
original configurations must accompany any exact reproduction claim.

The supplied LIBERO-10 SFT statistics cannot automatically support Spatial,
Object and Goal: each suite needs a valid matching normalization key and SFT
initialization. Asset checks fail instead of substituting a different suite's
action statistics.

## Evidence to retain

Each completed stage stores fresh grounded trajectories, all candidate
imagination videos, predicted residuals, admission probabilities, selected
indices, weights, proxy scores, measured calibration error, actual grounded
transitions, timings and actor gradient norm. Model-generated proxy progress
never becomes a measured environment success label.

To validate the no-oracle claim empirically, hold out whole trajectories and
score predictions before opening future labels. Report depth-specific residual
correlation/ranking and policy utility under equal candidate, interaction and
update budgets. Training fit MAE and passing tests are not held-out calibration
accuracy or evidence of MERL's performance advantage.
