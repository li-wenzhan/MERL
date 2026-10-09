"""Short-budget training followed by complete real-environment video evaluation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from merl.launch import ROOT, digest
from merl.modes import MODES


def training_overrides(steps, seconds):
    """Use the same training budget for every selected mode."""
    return [f"trainer.max_training_seconds={seconds}", "trainer.save_freq=1",
            "trainer.checkpoint_keep=1", "trainer.test_freq=1000000",
            "trainer.final_val_after_train=true", "merl.simulator_steps=2",
            "actor_rollout_ref.world_model.num_inference_steps=8"]


def protocol_for(config, task_ids, trials, horizon, eval_offset=10):
    from verl.utils.libero_path import load_libero_pro_config
    settings = load_libero_pro_config(str(config))
    active = [tag for key, tag in settings["perturbation_mapping"].items() if settings.get(key)]
    if active not in ([], ["env"]):
        raise ValueError("The quick comparison supports the existing original/environment-shift panel only")
    suite = "libero_10" + ("_env" if active else "")
    benchmark = Path(settings["libero_pro_root"]) / "libero/libero"
    assets = {}
    for kind, extension in (("bddl_files", "*.bddl"), ("init_files", "*.pruned_init")):
        files = sorted((benchmark / kind / suite).glob(extension))
        if len(files) != 10:
            raise ValueError(f"Freeze the complete ten-task {suite} asset panel before running: {kind}")
        assets.update({str(p.relative_to(benchmark)): digest(p) for p in files})
    protocol = dict(suite=suite, task_ids=task_ids, trials=trials, horizon=horizon,
                    trial_ids=list(range(eval_offset, eval_offset + trials)),
                    greedy_evaluation=True, evaluation_config_sha256=digest(config), asset_sha256=assets)
    protocol["id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("ALL", *MODES), default="ALL")
    p.add_argument("--modes", nargs="+", choices=MODES, help="Explicit sequential subset; use instead of --mode")
    p.add_argument("--continue-on-error", action="store_true", help="Attempt remaining modes after a failure")
    p.add_argument("--job", choices=("train", "evaluate"), default="train")
    p.add_argument("--config", type=Path, default=ROOT / "configs/merl.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--actor-checkpoint", type=Path, help="Sharded MERL actor to evaluate")
    p.add_argument("--label", help="Evaluation label, e.g. SFT; never changes the policy")
    p.add_argument("--vla-init", "--sft-checkpoint", dest="sft_checkpoint", type=Path, required=True)
    p.add_argument("--unnorm-key", help="Action statistics key used by the initialization")
    p.add_argument("--wm-checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--tasks", type=int, nargs="+", default=[0])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--eval-offset", type=int, default=10)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--training-minutes", type=float, default=15)
    p.add_argument("--actor-gpus", type=int, default=3)
    p.add_argument("--eval-config", type=Path, default=ROOT / "configs/evaluation_config.yaml")
    args = p.parse_args()
    modes = args.modes or (["MFRL", "MBRL", "MERL"] if args.mode == "ALL" else [args.mode])
    if args.modes and (args.mode != "ALL" or args.job != "train" or len(set(args.modes)) != len(args.modes)):
        p.error("--modes requires training, unique modes, and no --mode override")
    if args.job == "evaluate" and args.mode == "ALL":
        p.error("Evaluate one supplied checkpoint at a time; use --mode MFRL --label SFT for the initial policy")
    if args.job == "train" and args.label:
        p.error("Training labels are fixed to the selected modes")
    if args.actor_checkpoint and args.job != "evaluate":
        p.error("--actor-checkpoint is for evaluation")
    if (not args.tasks or len(set(args.tasks)) != len(args.tasks) or any(t < 0 or t > 9 for t in args.tasks)
            or min(args.trials, args.steps, args.training_minutes, args.actor_gpus) <= 0):
        p.error("Provide unique task IDs in 0..9 and positive budgets")
    if args.job == "train" and args.actor_gpus != 3:
        p.error("The short training profile requires three actor GPUs (plus a fourth for WM modes)")
    if args.eval_offset < args.trials or args.eval_offset + args.trials > 50:
        p.error("Evaluation states must be disjoint from training states and within the frozen 50-state panel")
    if args.job == "train" and any(mode != "MFRL" for mode in modes) and not args.wm_checkpoint:
        p.error("An explicit WM checkpoint is required")
    if args.job == "train" and len(args.tasks) * args.trials < args.actor_gpus:
        p.error("The training panel must contain at least one prompt per actor GPU")
    root = (args.output or ROOT / "outputs/comparisons" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    root.mkdir(parents=True, exist_ok=False)
    protocol = protocol_for(args.eval_config.resolve(), args.tasks, args.trials, 512, args.eval_offset)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    failed = False
    for mode in modes:
        label = args.label or mode
        folder = root / mode
        folder.mkdir()
        experiment = f"{mode.lower()}_ppt_{time.time_ns()}"
        overrides = [f"data.num_trials_per_task={args.trials}",
                     f"data.eval_trial_offset={args.eval_offset}",
                     f"actor_rollout_ref.rollout.allowed_task_ids={json.dumps(args.tasks)}",
                     "actor_rollout_ref.rollout.eval_max_steps=512", "trainer.strict_validate_rollout=true",
                     "actor_rollout_ref.rollout.video_save_interval=1",
                     f"actor_rollout_ref.rollout.presentation_dir={json.dumps(str(folder / 'episodes'))}",
                     f"actor_rollout_ref.rollout.presentation_protocol={json.dumps(protocol['id'])}",
                     f"actor_rollout_ref.rollout.presentation_label={json.dumps(label)}"]
        if args.job == "train":
            overrides += training_overrides(args.steps, args.training_minutes * 60)
        command = [sys.executable, "-u", "-m", "merl.launch", "--mode", mode, "--job", args.job,
                   "--experiment", experiment, "--sft-checkpoint", str(args.sft_checkpoint.resolve()),
                   "--eval-config", str(args.eval_config.resolve()), "--output-root", str(root / "runs"),
                   "--actor-gpus", str(args.actor_gpus), "--trials", str(args.trials),
                   "--stages", str(args.steps), "--seed", str(args.seed),
                   "--config", str(args.config.resolve())]
        if args.unnorm_key:
            command += ["--unnorm-key", args.unnorm_key]
        if args.actor_checkpoint:
            command += ["--actor-checkpoint", str(args.actor_checkpoint.resolve())]
        if args.wm_checkpoint and mode != "MFRL" and args.job == "train":
            command += ["--wm-checkpoint", str(args.wm_checkpoint.resolve())]
        command += ["--", *overrides]
        info = dict(label=label, mode=mode, job=args.job, status="running", command=command,
                    protocol_id=protocol["id"], input_checkpoint=str(args.sft_checkpoint.resolve()),
                    experiment_dir=str(root / "runs" / mode / experiment),
                    source_hashes={str(path.relative_to(ROOT)): digest(path) for path in
                                   (ROOT / "merl/presentation_run.py", ROOT / "merl/presentation_report.py")},
                    training_stages=args.steps, training_minutes=args.training_minutes)
        info_path = folder / "run_info.json"
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        started = time.monotonic()
        env = dict(os.environ, ACP_LOG_DIR=str(root / "logs"))
        print(f"[presentation] starting {label}; artifacts={folder}", flush=True)
        result = subprocess.run(["bash", str(ROOT / "scripts/run_logged.sh"), *command], cwd=ROOT, env=env)
        info.update(status="completed" if result.returncode == 0 else "failed", exit_code=result.returncode,
                    elapsed_seconds=time.monotonic() - started)
        info["assets_unchanged"] = protocol_for(args.eval_config.resolve(), args.tasks, args.trials, 512, args.eval_offset) == protocol
        failed |= result.returncode != 0 or not info["assets_unchanged"]
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        from merl.presentation_report import build_report
        build_report(root)
        if not info["assets_unchanged"]:
            raise RuntimeError("Evaluation assets changed; do not compare these runs")
        if result.returncode != 0 and not args.continue_on_error:
            print(f"[presentation] stopping after {label} failed; remaining modes were not started. "
                  f"Inspect {root / 'logs'} and {info['experiment_dir']}/ray_logs", flush=True)
            break
    print(f"[presentation] report={root / 'report'}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
