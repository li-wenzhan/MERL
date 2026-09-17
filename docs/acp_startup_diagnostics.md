# Diagnosing ACP startup stalls

The 2026-09-17 16:14:20 UTC presentation run passed the OpenVLA, LIBERO and WM
asset checks. Ray reported four GPUs and eight CPUs, selected three actor ranks
and one reserved WM slot, and built the three task datasets. The last supplied
console line is the replay-pool message at 16:18:03 UTC. The shared experiment
contains an empty rollout directory, but no completed training checkpoint.

Neither supplied log contains a traceback, a command exit summary or a platform
termination reason. This locates the last observed progress near worker startup;
it does not establish whether initialization was slow, blocked, or interrupted
externally. Missing optional LIBERO demonstration files and robosuite private
macros did not prevent the successful asset checks. Do not call them the cause.

## Changes

- Log resource allocation, worker submission, rank-zero registration, distributed
  rendezvous, actor model initialization and WM trainer initialization.
- Placement groups time out after 180 seconds. Rank-zero discovery has a
  360-second limit and propagates worker constructor errors; registration metadata
  has a 180-second limit. Actor and WM model initialization each have a 900-second
  limit. Distributed process-group operations have a 600-second timeout.
- Pending Ray gets print progress every 30 seconds. Worker initialization emits
  Python stack snapshots every 120 seconds; these snapshots are diagnostic and
  do not themselves indicate a failure.
- The launcher copies Ray text-log tails from local `/tmp` into the experiment's
  `ray_logs/` every 30 seconds and at process exit. Up to 256 KiB per file is kept,
  with original sizes and capture time in `index.json`. A platform hard kill can
  lose the last interval; these are log tails, not complete Ray archives.
- A failed presentation mode stops the sequence unless `--continue-on-error` is
  explicitly supplied. No Git or Internet checks are added.

The 900-second training budget is a soft limit evaluated between updates, after
initialization. It excludes model loading, checkpoint export and final evaluation;
it is not a 15-minute ACP wall-clock guarantee. Startup limits above apply per
stage, not to the entire job.

## Next run

Run normal short training with the updated launcher. Put `ONLINE_MBRL` and `MERL`
first when updated WM weights for presentation figures are the priority; the
frozen checkpoint is already available for the MBRL WM-image column. This does
not replace training the frozen-WM actor for a policy-success comparison.

If initialization stalls again, preserve the outer ACP log and the corresponding
experiment's `run.log`, `launch_manifest.json` and `ray_logs/`, plus the ACP
platform exit reason. Use the last `[startup]` stage and the worker stderr to
distinguish scheduling, module imports, rendezvous and checkpoint loading. Do not
change NCCL transport settings or model hyperparameters without that evidence.

The original three-worker registration and a Gloo all-reduce completed on the CCI
CPU runtime. This does not validate three-GPU NCCL/FSDP or resolve the original
ACP root cause. The single-GPU CCI cannot reproduce the full ACP topology.
