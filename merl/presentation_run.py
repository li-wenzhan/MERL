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


def training_overrides(steps, seconds):
    # Explicit pilot settings shared across modes; not paper-reproduction settings.
    return [f"trainer.total_training_steps={steps}", f"trainer.total_epochs={steps}",
            f"trainer.max_training_seconds={seconds}", "trainer.save_freq=1",
            "trainer.test_freq=1000000", "trainer.final_val_after_train=true",
            "data.n_samples=2", "data.filter_accuracy=false",
            "actor_rollout_ref.model.checkpoint_format=hf_full_state_dict",
            "actor_rollout_ref.actor.optim.lr=0.000005",
            "actor_rollout_ref.actor.ppo_mini_batch_size=6",
            "actor_rollout_ref.actor.traj_mini_batch_size=6",
            "actor_rollout_ref.actor.clip_ratio_high=0.2",
            "actor_rollout_ref.actor.clip_ratio_low=0.2",
            "actor_rollout_ref.rollout.temperature=1.0",
            "actor_rollout_ref.rollout.train_max_steps=384",
            "actor_rollout_ref.world_model.wm_inner_steps=2",
            "actor_rollout_ref.world_model.wm_warmup_steps=1",
            "actor_rollout_ref.world_model.num_inference_steps=8",
            "actor_rollout_ref.world_model.eval_num_inference_steps=8",
            "actor_rollout_ref.world_model.imag_horizon_min=64",
            "actor_rollout_ref.world_model.imag_horizon_max=128"]


def protocol_for(config, task_ids, trials, horizon, eval_offset=10):
    import yaml
    settings = yaml.safe_load(config.read_text())
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
    p.add_argument("--mode", choices=("ALL", "MFRL", "MBRL", "MERL"), default="ALL")
    p.add_argument("--job", choices=("train", "evaluate"), default="train")
    p.add_argument("--label", help="Evaluation label, e.g. SFT; never changes the policy")
    p.add_argument("--sft-checkpoint", type=Path, required=True)
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
    if args.job == "evaluate" and args.mode == "ALL":
        p.error("Evaluate one supplied checkpoint at a time; use --mode MFRL --label SFT for the initial policy")
    if args.job == "train" and args.label:
        p.error("Training labels are fixed to the selected modes")
    if (not args.tasks or len(set(args.tasks)) != len(args.tasks) or any(t < 0 or t > 9 for t in args.tasks)
            or min(args.trials, args.steps, args.training_minutes, args.actor_gpus) <= 0):
        p.error("Provide unique task IDs in 0..9 and positive budgets")
    if args.job == "train" and args.actor_gpus != 3:
        p.error("The short training profile requires three actor GPUs (plus a fourth for WM modes)")
    if args.eval_offset < args.trials or args.eval_offset + args.trials > 50:
        p.error("Evaluation states must be disjoint from training states and within the frozen 50-state panel")
    if args.job == "train" and args.mode != "MFRL" and not args.wm_checkpoint:
        p.error("An explicit WM checkpoint is required")
    if args.job == "train" and len(args.tasks) * args.trials < args.actor_gpus:
        p.error("The training panel must contain at least one prompt per actor GPU")
    root = (args.output or ROOT / "tmp_files/ppt_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    root.mkdir(parents=True, exist_ok=False)
    protocol = protocol_for(args.eval_config.resolve(), args.tasks, args.trials, 512, args.eval_offset)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    modes = ["MFRL", "MBRL", "MERL"] if args.mode == "ALL" else [args.mode]
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
                   "--actor-gpus", str(args.actor_gpus), "--trials", str(args.trials)]
        if args.wm_checkpoint and mode != "MFRL" and args.job == "train":
            command += ["--wm-checkpoint", str(args.wm_checkpoint.resolve())]
        command += ["--", *overrides]
        info = dict(label=label, mode=mode, job=args.job, status="running", command=command,
                    protocol_id=protocol["id"], input_checkpoint=str(args.sft_checkpoint.resolve()),
                    experiment_dir=str(root / "runs" / mode / experiment),
                    source_hashes={str(path.relative_to(ROOT)): digest(path) for path in
                                   (ROOT / "merl/presentation_run.py", ROOT / "merl/presentation_report.py")},
                    caveat="Short-budget legacy implementation pilot; no guarantee of method ranking or paper reproduction.")
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
    print(f"[presentation] report={root / 'report'}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
