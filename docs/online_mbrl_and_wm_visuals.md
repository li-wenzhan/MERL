# Online-updated MBRL and simulator visual comparisons

## Mode contract

The default camera-ready `ONLINE_MBRL` enables simulator evolution with fixed
mixing/horizon, uniform imagined replay and unit imagined weights. It shares
the real/imagined objective and grounded allowance with MERL. Simulator update
precedes imagination at the same stage. `STATIC_TRUST` freezes the simulator and
retains MERL's calibration, scheduling and chunk trust. See the full
[camera-ready control table](camera_ready_protocol.md).

### Historical legacy behavior

The following mode descriptions concern `--protocol legacy` only. They are
retained to interpret earlier artifacts; the new default does not use the mirror.

`ONLINE_MBRL` is a new baseline implementation. Previously the entrypoint froze
MBRL and the trainer allowed simulator updates only in MERL; setting `fine_tune`
alone could not create this baseline.

| Mode | Actor data | Simulator updates | Trust behavior |
| --- | --- | --- | --- |
| MFRL | Real-environment rollouts | None | None |
| MBRL | WM-path rollouts | Frozen | Existing MBRL behavior |
| ONLINE_MBRL | WM-path rollouts | Real-data update every outer iteration | Uniform sampling, unit imagined weights, fixed horizon, no confidence gate/anchor requirement/adaptive mixing |
| MERL | Real/WM mixture | Online | Existing legacy confidence, anchor and scheduling mechanisms |

Online MBRL first collects a separate real batch, writes `train_real` shards,
then uses the MBRL actor path (imagined fraction 1). After the actor update it
updates the simulator, saves `world_model/global_step_N/world_model.pth`, and
synchronizes all rollout workers before the next iteration. Real training
episodes are not inserted into its actor objective. Invalid/padded samples
remain masked; disabling trust does not make invalid samples trainable.

The baseline fails if the real batch, actual WM update, checkpoint save or worker
synchronization is missing. It logs actual real-data sample counts, WM update
steps and imagined actor weights. `merl/modes.py` rejects contradictory trust,
anchor, proxy-reward or warmup overrides. `imag_horizon_min` sets its fixed
horizon; WM loss and residual errors cannot change its horizon or actor ratio.

This inherits the existing **grounded mirror** WM rollout implementation. It is
not a new no-oracle recursive policy rollout. The default camera-ready protocol
integrates no-oracle recursive imagination. This legacy baseline is not a clean
`MERL minus trust` ablation: actor real/imagined mixture and reward routing also
differ. Report those differences and the additional real interaction cost.

## Training

Using the prepared ACP environment and local assets:

```bash
bash examples/run_presentation.sh --mode ONLINE_MBRL \
  --sft-checkpoint /models/openvla-oft --wm-checkpoint /models/ctrl-world.pt

# Only the three WM-based modes, sequentially on one four-GPU node:
bash examples/run_presentation.sh --modes MBRL ONLINE_MBRL MERL \
  --sft-checkpoint /models/openvla-oft --wm-checkpoint /models/ctrl-world.pt
```

The old default three-mode selection is unchanged. Add all four explicitly with
`--modes MFRL MBRL ONLINE_MBRL MERL` if resources permit. Every pilot now persists
updated WM checkpoints each outer step so early budget termination still leaves
weights for visualization. Plan approximately 30--50 minutes for the additional
online baseline, with uncertainty from real collection, WM updates and full
checkpoint I/O. Four sequential modes cannot be guaranteed within two hours.

## Reference trajectories and temporal alignment

Real presentation evaluations now also save lossless `trajectory.npz` next to
each valid `episode.json`: observations `[T+1,H,W,3]` and the **actual executed**
environment actions `[T,7]`. Action `a[t]` produces observation `o[t+1]`.
The gripper has already been normalized/inverted in the environment worker;
the comparison must not convert it again. Export rejects length mismatches.
Older video-only episodes cannot be retroactively given trustworthy actions.

Use a single frozen evaluation episode for all WM checkpoints. Online training
uses states 0--2 while the default evaluation uses 10--12; upstream SFT overlap
remains unknown. No reference trajectory is inserted into WM training by the
comparison command. Collect without retraining using:

```bash
bash examples/run_presentation.sh --mode MFRL --job evaluate --label SFT \
  --actor-gpus 1 --trials 1 --eval-offset 10 --sft-checkpoint /models/openvla-oft
```

## GT / frozen WM / updated WM / MERL panels

```bash
bash scripts/run_logged.sh python -m merl.wm_visual_compare \
  --episode /outputs/reference/MFRL/episodes/step_000000/task_XX_trial_YY_ID/episode.json \
  --checkpoint MBRL=/models/ctrl-world.pt \
  --checkpoint ONLINE_MBRL=/outputs/online/world_model/global_step_N/world_model.pth \
  --checkpoint MERL=/outputs/merl/world_model/global_step_N/world_model.pth \
  --start 64 --horizon 32 --inference-steps 8 --rollout recursive \
  --output tmp_files/wm_visuals/comparison_001
```

This uses one GPU and loads models sequentially. Supply actual distinct trained
checkpoints; missing or duplicate weights fail rather than masquerading as
different methods. A single `--checkpoint MBRL=...` produces a GT/frozen-WM
reference while training is pending. Allow roughly 5--15 minutes for three
models and a 64-frame window initially; measure the saved per-model timings.

A single-H100 run of the frozen checkpoint completed a 64-frame recursive
comparison in 198 seconds including startup/loading (8 diffusion steps,
8-frame chunks). This validates the rendering/inference path only; the new
four-GPU online training/update/synchronization path still requires ACP runtime
validation. Training time estimates above are not measured throughput.

Each model gets identical initial GT history, executed future actions, diffusion
steps and per-chunk seeds. `recursive` feeds only predictions back after the
first anchor. `teacher_forced` explicitly refreshes GT history at every chunk;
it is a different, labeled diagnostic. Neither protocol asks a policy for new
actions, so both compare **fixed-action model fidelity**, not closed-loop task
success. The final partial chunk uses only available actions, without padded GT.

Outputs include `comparison.mp4`, per-frame PNG panels, `contact_sheet.png`,
lossless prediction/proxy arrays, and a manifest with source/checkpoint/reference
hashes, seeds, timing, normalized pixel MSE and PSNR. Raw reference frames are
retained separately; display/metric frames use the same 320x192 resize for all
columns. Pixel fidelity alone is not a perceptual or downstream success claim.

Trust acts on data admission, weights and scheduling; it does not directly
sharpen simulator output. Identical WM weights and inputs should yield identical
predictions. A visual advantage for MERL requires trained checkpoint evidence
under a matched evaluation protocol, plus budget/data accounting and a separate
trust ablation to establish the cause.
