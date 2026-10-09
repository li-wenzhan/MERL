"""LIBERO policy training, evaluation and trajectory collection."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from merl.modes import MODES, ONLINE_WM_MODES, validate_online_mbrl
from merl.ray_diagnostics import RayLogCapture

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/launch_profiles.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--config", type=Path, default=ROOT / "configs/merl.json")
    p.add_argument("--stages", type=int, default=100, help="Total outer refinement stages, including resumed stages")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-keep", type=int, default=2, help="Completed checkpoints retained in this fresh run; 0 keeps all")
    p.add_argument("--resume-from", type=Path, help="A completed MERL stage .pt, restored into a fresh run")
    p.add_argument("--actor-checkpoint", type=Path, help="FSDP actor checkpoint to evaluate, using SFT assets as the base")
    p.add_argument("--job", choices=("train", "evaluate", "collect"), default="train")
    p.add_argument("--vla-init", "--sft-checkpoint", dest="sft_checkpoint", type=Path, required=True,
                   help="Categorical OpenVLA-OFT initialization directory")
    p.add_argument("--wm-checkpoint", type=Path)
    p.add_argument("--eval-config", type=Path, default=ROOT / "configs/evaluation_config.yaml")
    p.add_argument("--wm-config", type=Path, default=ROOT / "configs/wm_online_config.py")
    p.add_argument("--output-root", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--experiment", required=True)
    p.add_argument("--suite", default="libero_10", choices=("libero_10", "libero_spatial", "libero_object", "libero_goal"))
    p.add_argument("--unnorm-key", help="Action statistics key; defaults to the selected suite")
    p.add_argument("--actor-gpus", type=int, default=3)
    p.add_argument("--collection-dir", "--shared-wm-eval", dest="shared_wm_eval", type=Path)
    p.add_argument("--split", default="wm_eval_fixed_mini", choices=("wm_train", "wm_eval_fixed_mini", "wm_eval_fixed_full"))
    p.add_argument("--collection-trial-offset", type=int, help="First collected state ID; defaults to 0 for training, 10 for evaluation")
    p.add_argument("--trials", type=int, default=2, help="Trials per task for collection/evaluation")
    p.add_argument("--smoke", action="store_true", help="Run one short update")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Print command without accessing assets or GPUs")
    group.add_argument("--check", action="store_true", help="Check assets and compose config; no Ray or training")
    p.add_argument("--render-check", action="store_true", help="Also reset and step a real environment")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides after --")
    return p


def build_settings(args):
    if args.actor_gpus < 1 or args.trials < 1:
        raise ValueError("actor-gpus and trials must be positive")
    if (not args.experiment or args.experiment in (".", "..")
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in args.experiment)):
        raise ValueError("experiment must be a single ASCII name")
    if args.job == "collect" and args.mode != "MFRL":
        raise ValueError("trajectory collection requires real-only MFRL mode")
    if args.job == "collect" and not args.shared_wm_eval:
        raise ValueError("--collection-dir is required for trajectory collection")
    if args.collection_trial_offset is not None and args.job != "collect":
        raise ValueError("--collection-trial-offset applies to collection")
    raw = json.loads(PROFILE.read_text())
    cfg = dict(raw["common"])
    root = args.output_root.expanduser().resolve() / args.mode / args.experiment
    model = root / "actor_assets"
    wm_enabled = args.mode != "MFRL" and args.job == "train"
    if wm_enabled and not args.wm_checkpoint:
        raise ValueError("MERL/MBRL training requires an explicit --wm-checkpoint")
    cfg.update({
        "data.task_suite_name": args.suite,
        "actor_rollout_ref.model.path": str(model),
        "actor_rollout_ref.rollout.pretrained_checkpoint": str(model),
        "actor_rollout_ref.rollout.experiment_name": args.experiment,
        "actor_rollout_ref.rollout.unnorm_key": args.unnorm_key or args.suite,
        "actor_rollout_ref.rollout.task_suite_name": args.suite,
        "actor_rollout_ref.rollout.libero_pro_eval_config_path": str(args.eval_config.expanduser().resolve()),
        "actor_rollout_ref.world_model.config_path": str(args.wm_config.expanduser().resolve()),
        "actor_rollout_ref.world_model.enable": wm_enabled,
        "actor_rollout_ref.world_model.fine_tune": wm_enabled and args.mode in ONLINE_WM_MODES,
        "actor_rollout_ref.world_model.load_from_ckpt": wm_enabled,
        "actor_rollout_ref.world_model.fixed_eval_enabled": False,
        "actor_rollout_ref.wm_gpu_idx": args.actor_gpus if wm_enabled else 0,
        "actor_rollout_ref.rollout_base_dir": str(root / "rollouts"),
        "trainer.default_local_dir": str(root), "trainer.project_name": args.mode,
        "trainer.experiment_name": args.experiment, "trainer.train_mode": args.mode,
        "trainer.resume.enable": False, "trainer.resume.resume_dir": str(root),
        "trainer.n_gpus_per_node": args.actor_gpus,
        # Inherit the launcher/ACP environment; the old JSON forces GLX paths.
        "trainer.runtime_env": "",
        "trainer.ray_num_cpus": max(4, 2 * args.actor_gpus + 2),
        "trainer.ray_tmpdir": f"/tmp/merl_ray_{hashlib.sha256(str(root).encode()).hexdigest()[:12]}",
        "trainer.ray_num_gpus": args.actor_gpus + int(wm_enabled), "trainer.ray_address": "",
    })
    # Preserve per-rank batch sizes when selecting a different actor count.
    for key in ("data.train_batch_size", "data.val_batch_size", "data.rollout_batch_size",
                "actor_rollout_ref.actor.ppo_micro_batch_size",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size", "actor_rollout_ref.ref.log_prob_micro_batch_size"):
        cfg[key] = args.actor_gpus
    for key in ("actor_rollout_ref.actor.ppo_mini_batch_size", "actor_rollout_ref.actor.traj_mini_batch_size"):
        cfg[key] = cfg[key] // 3 * args.actor_gpus
    if wm_enabled:
        cfg["actor_rollout_ref.world_model.ckpt_path"] = str(args.wm_checkpoint.expanduser().resolve())
    if args.shared_wm_eval:
        cfg["actor_rollout_ref.world_model.fixed_eval_root"] = str(args.shared_wm_eval.expanduser().resolve())
    if args.job != "train":
        cfg.update({"data.num_trials_per_task": args.trials, "data.n_samples": 1,
                    "trainer.val_only": args.job == "evaluate", "trainer.val_before_train": args.job == "evaluate",
                    "trainer.strict_validate_rollout": args.job == "evaluate",
                    "trainer.strict_mode_assert": False})
    if args.job == "collect":
        trial_offset = args.collection_trial_offset
        if trial_offset is None:
            trial_offset = 0 if args.split == "wm_train" else 10
        if trial_offset < 0 or trial_offset + args.trials > 50:
            raise ValueError("collection trial IDs must fit within the 50-state panel")
        cfg.update({"trainer.rollout_before_train": True, "trainer.sim_rollout_epoch": 1,
                    "data.rollout_trial_offset": trial_offset,
                    "trainer.preserve_rollout_base_dir": True, "trainer.rollout_train_split": args.split,
                    "trainer.rollout_save_eval": False, "trainer.rollout_save_to_hdfs": True,
                    "trainer.rollout_do_sample": False, "actor_rollout_ref.rollout.temperature": 0.0,
                    "actor_rollout_ref.rollout_base_dir": str(args.shared_wm_eval.expanduser().resolve())})
    if args.job == "collect" and args.split == "wm_train":
        cfg["actor_rollout_ref.rollout.grounded_export_dir"] = str(
            args.shared_wm_eval.expanduser().resolve() / "wm_train" / "trajectories")
    if args.smoke:
        cfg.update({"trainer.total_training_steps": 1, "trainer.total_epochs": 1,
                    "data.filter_accuracy": False,
                    "actor_rollout_ref.rollout.train_max_steps": 16,
                    "actor_rollout_ref.rollout.eval_max_steps": 16,
                    "actor_rollout_ref.rollout.allowed_task_ids": [0],
                    "actor_rollout_ref.rollout.max_success_attempts": 1,
                    "actor_rollout_ref.world_model.wm_inner_steps": 1,
                    "actor_rollout_ref.world_model.imag_horizon_min": 8,
                    "actor_rollout_ref.world_model.imag_horizon_max": 16,
                    "actor_rollout_ref.world_model.num_inference_steps": 2,
                    "actor_rollout_ref.world_model.eval_num_inference_steps": 2})
    from dataclasses import asdict, replace
    from merl.algorithm import MERLConfig
    merl = MERLConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    merl = replace(merl, seed=args.seed).for_mode(args.mode)
    if args.smoke:
        merl = replace(merl, simulator_steps=1, residual_fit_steps=20, grounded_step_cap=16,
                        calibration_depth=1, real_chunks_per_update=6, imagined_chunks_per_update=6)
    if args.stages < 1 or args.checkpoint_keep < 0 or args.actor_gpus not in (1, 3):
        raise ValueError("MERL supports positive stage counts and 1 or 3 actor GPUs")
    cfg.update({
        "merl": asdict(merl), "trainer.engine": "merl",
        "trainer.total_training_steps": 1 if args.smoke else args.stages,
        "trainer.total_epochs": args.stages, "trainer.val_only": args.job == "evaluate",
        "trainer.val_before_train": args.job == "evaluate",
        "trainer.final_val_after_train": True, "trainer.test_freq": 10, "trainer.save_freq": 5,
        "trainer.checkpoint_keep": args.checkpoint_keep,
        "trainer.strict_validate_rollout": True, "trainer.preserve_rollout_base_dir": True,
        "trainer.resume_from": str(args.resume_from.resolve()) if args.resume_from else "",
        "trainer.actor_checkpoint": str(args.actor_checkpoint.resolve()) if args.actor_checkpoint else "",
        "actor_rollout_ref.world_model.shared_simulator": True,
        "actor_rollout_ref.world_model.merl_config": asdict(merl),
        "actor_rollout_ref.world_model.train_mode": args.mode,
        "actor_rollout_ref.world_model.mixed_precision": "bf16",
        "actor_rollout_ref.world_model.dtype": "bf16",
        "actor_rollout_ref.world_model.self_forcing_weight": 0.0,
        "actor_rollout_ref.world_model.use_wm_reward_proxy": True,
        "actor_rollout_ref.world_model.wm_real_anchor_reward": False,
        "actor_rollout_ref.world_model.require_wm_anchor_reward": False,
        "actor_rollout_ref.world_model.zero_unanchored_wm_weight": False,
        "actor_rollout_ref.world_model.weak_update_enable": False,
        "actor_rollout_ref.world_model.merl_imagined_reward_hard_constraint": False,
        "actor_rollout_ref.world_model.imag_advantage_abs_clip": 0,
        "actor_rollout_ref.world_model.wm_warmup_steps": 0,
        "actor_rollout_ref.world_model.wm_grpo_uid_mode": "source",
        "actor_rollout_ref.model.checkpoint_format": "fsdp_sharded_state_dict",
        "actor_rollout_ref.actor.optim.lr": 5e-6,
        "actor_rollout_ref.actor.clip_ratio_high": 0.2,
        "actor_rollout_ref.actor.clip_ratio_low": 0.2,
        "actor_rollout_ref.rollout.temperature": 1.2,
        "actor_rollout_ref.rollout.train_max_steps": merl.grounded_step_cap,
        "actor_rollout_ref.rollout.eval_max_steps": 512,
        "actor_rollout_ref.rollout.merl_config": asdict(merl),
        "actor_rollout_ref.rollout.presentation_dir": str(root / "evaluation"),
        "actor_rollout_ref.rollout.presentation_label": args.mode,
        "actor_rollout_ref.rollout.save_training_videos": True,
        "actor_rollout_ref.actor.use_kl_loss": False, "algorithm.kl_ctrl.kl_coef": 0.,
        "algorithm.adv_estimator": "grpo", "data.n_samples": 2 if args.job == "train" else 1,
        "data.filter_accuracy": False, "data.eval_trial_offset": 10,
    })
    if args.resume_from and args.job != "train":
        raise ValueError("--resume-from is a training operation")
    if args.actor_checkpoint and args.job == "train":
        raise ValueError("--actor-checkpoint is for evaluation/collection; use --resume-from for training")
    return cfg, root


def hydra_args(settings, extra):
    extra = extra[1:] if extra[:1] == ["--"] else extra
    protected = {"trainer.train_mode", "trainer.default_local_dir", "trainer.project_name",
                 "trainer.experiment_name", "trainer.n_gpus_per_node", "trainer.nnodes",
                 "actor_rollout_ref.wm_gpu_idx", "actor_rollout_ref.model.path",
                 "actor_rollout_ref.world_model.enable", "actor_rollout_ref.world_model.load_from_ckpt",
                 "actor_rollout_ref.world_model.ckpt_path", "trainer.resume.enable",
                 "trainer.engine", "actor_rollout_ref.world_model.shared_simulator"}
    protected.update({"actor_rollout_ref.world_model.fine_tune", "actor_rollout_ref.world_model.config_path",
                      "actor_rollout_ref.world_model.fixed_eval_enabled", "actor_rollout_ref.world_model.fixed_eval_root",
                      "actor_rollout_ref.rollout.pretrained_checkpoint", "actor_rollout_ref.rollout.libero_pro_eval_config_path",
                      "actor_rollout_ref.rollout_base_dir", "trainer.ray_num_gpus", "trainer.ray_address",
                      "trainer.rollout_before_train", "trainer.val_only", "trainer.val_before_train",
                      "trainer.rollout_train_split", "trainer.preserve_rollout_base_dir",
                      "trainer.runtime_env", "data.task_suite_name", "actor_rollout_ref.rollout.task_suite_name",
                      "data.rollout_trial_offset",
                      "actor_rollout_ref.rollout.unnorm_key"})
    def value(item):
        if isinstance(item, dict):
            return "{" + ",".join(f"{key}:{value(v)}" for key, v in item.items()) + "}"
        if isinstance(item, (list, tuple)):
            return "[" + ",".join(value(v) for v in item) + "]"
        return json.dumps(item, separators=(",", ":"))
    rendered = {k: f"++{k}={value(v)}" for k, v in settings.items()}
    for item in extra:
        if "=" not in item or item.startswith(("~", "--")):
            raise ValueError("additional arguments must be Hydra key=value overrides")
        key = item.lstrip("+").split("=", 1)[0]
        if key in protected or any(item.startswith(key + ".") for item in protected):
            raise ValueError("use launcher options for mode, assets, layout and output paths")
        rendered[key] = "++" + item.lstrip("+")
    return list(rendered.values())


def compose_config(overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(cfg)
    if cfg.trainer.train_mode == "ONLINE_MBRL" and not cfg.trainer.val_only:
        validate_online_mbrl(cfg.actor_rollout_ref.world_model)
    if cfg.trainer.get("engine") == "merl":
        from merl.algorithm import MERLConfig
        merl = MERLConfig.from_dict(OmegaConf.to_container(cfg.merl, resolve=True))
        if merl.grounded_trajectories != 6:
            raise ValueError("MERL collection requires exactly six grounded trajectories")
        if cfg.actor_rollout_ref.world_model.enable and merl.grounded_step_cap <= merl.history_size:
            raise ValueError("simulator training needs grounded trajectories longer than its stored history")
        if cfg.trainer.total_training_steps < 1:
            raise ValueError("MERL requires a positive outer-stage limit")
        model, rollout = cfg.actor_rollout_ref.model, cfg.actor_rollout_ref.rollout
        if (model.vla != "openvla-oft" or model.action_token_len != 7 or model.action_chunks_len != 8
                or merl.chunk_size != 8 or merl.history_size != 8 or rollout.use_proprio
                or rollout.num_images_in_input != 1):
            raise ValueError("MERL requires tokenized OpenVLA-OFT, 8x7 commands, 8 history frames, one RGB and no proprioception")
        if cfg.actor_rollout_ref.actor.clip_ratio_low != cfg.actor_rollout_ref.actor.clip_ratio_high:
            raise ValueError("MERL uses symmetric policy clipping")
        if (not cfg.trainer.val_only and not cfg.trainer.get("rollout_before_train", False)
                and cfg.trainer.n_gpus_per_node == 1
                and str(cfg.actor_rollout_ref.actor.fsdp_config.get("model_dtype", "fp32")).lower() in ("fp32", "float32", "none")):
            raise ValueError("full FP32 Adam training requires three actor GPUs; single-GPU evaluation/collection remains available")
    if cfg.trainer.nnodes != 1:
        raise ValueError("the launcher supports one node")
    for value in (cfg.data.train_batch_size, cfg.data.val_batch_size,
                  cfg.actor_rollout_ref.actor.ppo_mini_batch_size,
                  cfg.actor_rollout_ref.ref.log_prob_micro_batch_size):
        if value < 1 or value % cfg.trainer.n_gpus_per_node:
            raise ValueError("global batch sizes must be divisible by actor-gpus")
    return OmegaConf.to_container(cfg, resolve=True)


def runtime_env():
    env = dict(os.environ)
    defaults = {"ROBOT_PLATFORM": "LIBERO", "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "NCCL_DEBUG": "WARN", "TORCH_DISTRIBUTED_DEBUG": "OFF", "TOKENIZERS_PARALLELISM": "false",
                "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "MERL_LIBERO_ENV_BACKEND": "egl",
                "MERL_LIBERO_EGL_DEVICE_ID": "auto", "MERL_ENV_MP_START_METHOD": "spawn",
                "MERL_LIBERO_ENV_SERVICE_ENABLE": "true", "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                "MERL_RAY_FORCE_LOCAL": "true", "RAY_USAGE_STATS_ENABLED": "0", "HYDRA_FULL_ERROR": "1"}
    for key, value in defaults.items():
        env.setdefault(key, value)
    if "CUDA_VISIBLE_DEVICES" in env:
        env.setdefault("MERL_GLOBAL_CUDA_VISIBLE_DEVICES", env["CUDA_VISIBLE_DEVICES"])
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def main():
    p = parser()
    args = p.parse_args()
    try:
        settings, run_dir = build_settings(args)
        overrides = hydra_args(settings, args.overrides)
    except ValueError as exc:
        p.error(str(exc))
    command = [sys.executable, "-u", "-m", "verl.trainer.main_ppo", *overrides]
    if args.dry_run:
        print(json.dumps({"engine": "merl", "command": command}, indent=2))
        return
    resolved = compose_config(overrides)
    env = runtime_env()
    wm = resolved["actor_rollout_ref"]["world_model"]
    if platform.system() != "Linux":
        p.error("asset checks and execution require Linux")

    def check(script, *arguments):
        started = time.monotonic()
        print(f"[launch] {datetime.now(timezone.utc).isoformat()} preflight begin: {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
                       cwd=ROOT, env=env, check=True)
        print(f"[launch] preflight passed: {script}; elapsed_seconds={time.monotonic() - started:.1f}", flush=True)

    checkpoint = args.sft_checkpoint.expanduser().resolve()
    stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
    unnorm_key = args.unnorm_key or (args.suite if args.suite in stats else args.suite + "_no_noops")
    check("preflight_openvla_oft.py", "--checkpoint", checkpoint, "--unnorm-key", unnorm_key)
    check("preflight_libero_pro_compat.py", "--config", args.eval_config.expanduser().resolve(), "--actor-import")
    if wm["enable"]:
        check("preflight_world_model_backbone.py", "--config", args.wm_config.expanduser().resolve(),
              "--checkpoint", args.wm_checkpoint.expanduser().resolve())
    if wm["fixed_eval_enabled"]:
        for split in ("wm_eval_fixed_mini", "wm_eval_fixed_full"):
            if not any((Path(wm["fixed_eval_root"]) / split).rglob("*.tar")):
                p.error(f"missing fixed WM evaluation shards: {split}; run --job collect first")
    if args.render_check:
        check("preflight_libero_env_service.py", "--config", args.eval_config.expanduser().resolve(),
              "--task-suite", args.suite, "--timeout-s", "240")
    if args.check:
        print("[launch] assets and resolved Hydra configuration OK; no training started")
        return
    if args.job == "collect":
        split_dir = args.shared_wm_eval.expanduser().resolve() / args.split
        if split_dir.exists() and any(split_dir.iterdir()):
            p.error(f"collection split already contains data: {split_dir}; use a fresh root")
    import torch
    devices = [dict(name=torch.cuda.get_device_name(i), memory=torch.cuda.get_device_properties(i).total_memory)
               for i in range(torch.cuda.device_count())]
    required = args.actor_gpus + int(wm["enable"])
    if len(devices) < required:
        p.error(f"requires {required} visible GPUs; found {len(devices)}. Use --check for asset validation")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"engine": "merl", "job": args.job,
                "smoke": args.smoke,
                "input_assets": {"sft_checkpoint": str(checkpoint),
                                 "sft_index_sha256": digest(checkpoint / "model.safetensors.index.json"),
                                 "sft_statistics_sha256": digest(checkpoint / "dataset_statistics.json"),
                                 "wm_checkpoint": str(args.wm_checkpoint) if wm["enable"] else None},
                "started_utc": datetime.now(timezone.utc).isoformat(), "command": command,
                "resolved_config": resolved, "devices": devices, "python": sys.version,
                # ACP may run a copied tree without Git or network access.
                "code_revision_label": env.get("MERL_CODE_REVISION"),
                "console_log": env.get("MERL_CONSOLE_LOG"),
                "source_hashes": {str(path): digest(path) for path in (
                    PROFILE, args.eval_config.expanduser().resolve(), args.wm_config.expanduser().resolve(),
                    ROOT / "merl/launch.py", ROOT / "verl/trainer/main_ppo.py",
                    args.config.expanduser().resolve(), ROOT / "merl/algorithm.py", ROOT / "merl/checkpoint.py",
                    ROOT / "merl/trainer.py", ROOT / "merl/imagined_rollout.py", ROOT / "merl/simulator.py",
                    ROOT / "merl/trust.py", ROOT / "merl/stored_calibration.py", ROOT / "merl/proxy.py",
                    ROOT / "merl/episode_artifacts.py", ROOT / "verl/utils/dataset/rob_dataset.py",
                    ROOT / "merl/modes.py",
                    ROOT / "merl/ray_diagnostics.py", ROOT / "verl/single_controller/ray/base.py",
                    ROOT / "verl/utils/libero_runtime.py",
                    ROOT / "verl/trainer/ppo/ray_trainer.py", ROOT / "verl/workers/fsdp_workers.py",
                    ROOT / "verl/workers/actor/dp_rob.py", ROOT / "verl/workers/rollout/rob_rollout_wm_pro.py")},
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "ray", "numpy")},
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"), "status": "preparing"}
    manifest_path = run_dir / "launch_manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save()
    code = 1
    try:
        subprocess.run(["bash", str(ROOT / "scripts/prepare_vla_assets.sh"),
                        str(args.sft_checkpoint.expanduser().resolve()), str(run_dir / "actor_assets")],
                       cwd=ROOT, env=env, check=True)
        manifest["status"] = "running"
        save()
        ray_log_root = run_dir / "ray_logs"
        manifest["ray_log_tails"] = str(ray_log_root)
        save()
        print(f"[launch] preserving Ray runtime log tails: {ray_log_root}", flush=True)
        with RayLogCapture(resolved.get("trainer", {}).get("ray_tmpdir", settings["trainer.ray_tmpdir"]), ray_log_root), \
                (run_dir / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        manifest["status"] = "completed" if code == 0 else "failed"
    except BaseException:
        manifest["status"] = "interrupted_or_launch_error"
        raise
    finally:
        manifest.update(exit_code=code, ended_utc=datetime.now(timezone.utc).isoformat())
        save()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
