# Short-budget presentation pilot

Run the three existing modes sequentially on one four-H100 node. This workflow
produces actual checkpoints and real-environment videos; it does not synthesize
successes or select trials based on outcomes. The protocol is a small exploratory
comparison, not a paper reproduction or proof that MERL outperforms a baseline.
The production WM path still has the discrepancies in
[the implementation audit](implementation_audit.md), including grounded mirror
rollouts and the pending no-oracle integration.

## Run

Activate the prepared Linux environment, synchronize code outside the ACP job,
then submit this command through ACP's web UI:

```bash
bash examples/run_presentation.sh \
  --sft-checkpoint /models/openvla-oft \
  --wm-checkpoint /models/ctrl-world.pt
```

The job contains no Git invocation, installs or network downloads. Offline asset
checks remain enabled to catch wrong paths before allocating Ray workers.
Defaults are task 0, three training initial states (0--2), three evaluation
initial states (10--12), six maximum outer updates per mode, and a 15-minute
training soft limit per mode. A completed outer update includes rollout, actor
update, WM update where applicable, logging and checkpoint saving. The time limit
is checked between updates; an update already running is allowed to finish.
Initialization and final evaluation are outside this limit.

Planning estimate: roughly 60--120 minutes for all modes, with substantial
uncertainty until the first ACP update is measured. Three initialization phases,
final evaluations, shared-storage checkpoint writes and the final update can
exceed that estimate. Do not impose a hard 15-minute process kill: it would lose
the final evaluation. Use `--mode MERL` etc. to run only a missing/failed mode in
a fresh output directory. `--steps 12 --training-minutes 25` provides a larger
budget if resources permit; it does not guarantee improvement.

Three actor GPUs are used for every training mode. WM modes reserve the fourth
GPU for the WM trainer; MFRL leaves it unused. CCI single-GPU evaluation is
supported via `--mode MFRL --job evaluate --label SFT --actor-gpus 1`.
Evaluation uses greedy real-environment actions and the full 512-step limit.
Training uses 384 steps and two samples per prompt. The pilot shares actor LR,
clip ranges, temperature and minibatch sizes across modes; legacy mode-specific
WM/anchor/KL guards remain in effect and are recorded in resolved configurations.
WM diffusion is shortened to eight steps and MERL warmup to one outer update.
These are pilot settings, not tuned or reproduced paper settings.
The pilot explicitly exports full Hugging Face actor checkpoints so they can be
evaluated independently later. Each FP32 actor export is approximately 30 GB;
the existing retention policy keeps two per mode. Allow roughly 200 GB for actor
exports across the three modes, plus rollout/replay/WM artifacts. Shared-storage
write speed affects the timing estimate.

## Saved evidence

Every invocation creates a new `tmp_files/ppt_runs/<UTC timestamp>/`:

- `protocol.json`: task/state selection and hashes of the evaluation YAML,
  BDDL files and initial-state files. The frozen panel is checked again after
  each mode. Evaluation states are disjoint from this online training; overlap
  with the upstream SFT training data is unknown.
- `MODE/run_info.json`: exact command, input checkpoint, status and elapsed time.
- `MODE/episodes/step_*/task_*/`: complete episode MP4, up to five unaltered PNG
  keyframes and JSON outcome/provenance. Invalid episodes are recorded too.
- `runs/MODE/experiment/`: resolved config, metrics, checkpoints and run log.
- `runs/MODE/experiment/ray_logs/`: periodically saved Ray worker, scheduler and
  runtime-environment text log tails (up to 256 KiB per file). `index.json`
  records original sizes and capture time. These survive loss of node-local
  `/tmp`; an abrupt platform kill can lose the last capture interval.
- `logs/`: per-mode complete stdout/stderr and exit/timing records. The outer
  orchestration log also goes to `tmp_files/acp_logs/`.
- `report/task_XX_trial_YY.mp4` and `.png`: side-by-side robot videos and contact
  sheets for every requested trial from completed valid panels.
- `report/scores.png`, `summary.json`, `trials.csv`, `README.md`: descriptive
  success counts, actual outer updates, gradient/imagined-weight evidence and
  limitations. No expected winner is encoded. Incomplete or invalid panels do
  not receive a comparable success rate.

Video panels align by stored frame index at 30 playback FPS, not physical wall
time. Shorter episodes hold the last frame with an explicit ended label. Raw
episode frames and action-step counts are preserved separately. Rank-padding
episodes do not enter the exported evaluation panel.

The report refreshes after each mode. A failed mode stops the sequence by default
to avoid repeating a shared infrastructure failure; `--continue-on-error`
explicitly attempts the remaining modes. Rebuild from saved artifacts without loading models:

```bash
python -m merl.presentation_report tmp_files/ppt_runs/RUN_ID
```

Look for nonzero finite gradients and actual checkpoints before claiming that a
policy learned. For MERL/MBRL also inspect positive imagined actor weights and
tokens; for MERL, require completed WM update steps before claiming simulator
evolution. Sparse success rewards can yield zero advantages for all samples;
additional runtime cannot be assumed to produce a convincing ranking. Retain
unsuccessful episodes and report actual completed updates rather than describing
all three modes as having matched compute or interaction budgets.
