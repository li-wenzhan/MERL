# No-oracle trust

MERL calibrates reliability using stored grounded data and applies that calibration to new imagined chunks. The simulator update, calibration, imagination and policy update use a fixed stage order.

## Stored calibration

`Simulator.prepare_stage` in `merl/simulator.py` loads the current stage's six grounded trajectories. `build_calibration_batch` in `merl/stored_calibration.py` anchors prediction at a stored observation and recursively replays recorded actions for depths 1–4. Recorded future observations supply residual labels; recursive inputs use past context and predicted RGB.

`Simulator.encode` uses the same frozen VAE posterior mean and spatial pooling for both grounded and predicted RGB. Masks cover the actual valid chunk length. `CalibrationBatch.residuals` computes visual mismatch and progress-proxy mismatch over that mask.

The progress proxy evaluates the observation preceding each executed action. Its grounded target is discounted, truncated success-to-go. Stage proxy error is measured on grounded observation/action pairs; chunk calibration evaluates imagined observation/action pairs.

## Frozen residual predictor

`ResidualPredictor.fit` refreshes a two-output MLP after simulator adaptation. Its inputs contain the anchor latent, imagined latent statistics, ordered executed actions, valid-step mask, rollout depth and stage context. Training uses paired stored residuals. Feature normalization and MLP parameters are fitted together and frozen for the subsequent imagination and actor update.

`ChunkFeatures` is the inference interface. `predict` takes these features and a matching simulator revision/stage. Grounded future observations are contained only in `CalibrationBatch`, which is used during fitting. Both the predictor snapshot and its feature normalization are saved for resume.

## Stage and chunk trust

`StageScheduler` smooths the weighted visual/proxy error with an EMA, converts it to inverse-power confidence, and maps confidence to the mixture ratio and floored imagination horizon. Bounds and coefficients are configured in `configs/merl.json`.

For chunk error `ε = α_obs ε_obs + α_proxy ε_proxy`, replay priority is `(ε + priority_epsilon)^(-priority_exponent)`. Normalized priorities choose imagined chunks; trust weights are `max(weight_min, exp(-weight_eta × ε))`. The two functions serve separate roles: priority controls sampling, while weight scales the policy loss.

Real and imagined branches use their own chunk-count denominators. `branch_coefficients` and `clipped_chunk_loss` in `merl/algorithm.py` detach advantages and trust weights, apply the valid-token mask before exponentiation, and sum categorical loss terms inside each chunk. `update_chunk_policy` in `verl/workers/actor/dp_rob.py` accumulates that objective across actor microbatches.

## Action and timestep conventions

One chunk contains at most eight executed 7D commands. Imagination horizons count environment steps, between 8 and 32. `imagined_rollout.py` decodes and denormalizes categorical policy tokens, converts the gripper once to the environment convention, and retains the categorical tokens separately for optimization. Padding in a final partial chunk contributes neither residual labels nor token loss.

The simulator, proxy and residual predictor remain frozen during recursive policy rollout and optimization. Calibration records `trust/calibration_environment_calls=0`; driver logs also record actual grounded transitions, the selected mixture ratio/horizon, replay weights and gradient norms.
