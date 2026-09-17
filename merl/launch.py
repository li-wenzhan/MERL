"""Unified LIBERO training, evaluation and collection on CCI/ACP."""

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

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/launch_profiles.json"
MODES = ("MERL", "MFRL", "MBRL")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--job", choices=("train", "evaluate", "collect"), default="train")
    p.add_argument("--sft-checkpoint", type=Path, required=True)
    p.add_argument("--wm-checkpoint", type=Path)
    p.add_argument("--eval-config", type=Path, default=ROOT / "configs/evaluation_config.yaml")
    p.add_argument("--wm-config", type=Path, default=ROOT / "configs/wm_online_config.py")
    p.add_argument("--output-root", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--experiment", required=True)
    p.add_argument("--suite", default="libero_10", choices=("libero_10", "libero_spatial", "libero_object", "libero_goal"))
    p.add_argument("--actor-gpus", type=int, default=3)
    p.add_argument("--shared-wm-eval", type=Path)
    p.add_argument("--wm-eval", choices=("off", "fixed"), default="off")
    p.add_argument("--split", default="wm_eval_fixed_mini", choices=("wm_eval_fixed_mini", "wm_eval_fixed_full"))
    p.add_argument("--trials", type=int, default=2, help="Trials per task for collection/evaluation")
    p.add_argument("--smoke", action="store_true", help="One short update; not benchmark evidence")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Print command without accessing assets or GPUs")
    group.add_argument("--check", action="store_true", help="Check assets and compose config on CCI; no Ray or training")
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
        raise ValueError("fixed evaluation collection requires real-only MFRL mode")
    if args.wm_eval == "fixed" and (args.mode == "MFRL" or args.job != "train"):
        raise ValueError("fixed WM evaluation requires MERL/MBRL training")
    if (args.wm_eval == "fixed" or args.job == "collect") and not args.shared_wm_eval:
        raise ValueError("--shared-wm-eval is required for fixed evaluation or collection")
    raw = json.loads(PROFILE.read_text())
    cfg = {**raw["common"], **raw["modes"][args.mode]}
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
        "actor_rollout_ref.rollout.unnorm_key": args.suite,
        "actor_rollout_ref.rollout.task_suite_name": args.suite,
        "actor_rollout_ref.rollout.libero_pro_eval_config_path": str(args.eval_config.expanduser().resolve()),
        "actor_rollout_ref.world_model.config_path": str(args.wm_config.expanduser().resolve()),
        "actor_rollout_ref.world_model.enable": wm_enabled,
        "actor_rollout_ref.world_model.fine_tune": wm_enabled and args.mode == "MERL",
        "actor_rollout_ref.world_model.load_from_ckpt": wm_enabled,
        "actor_rollout_ref.world_model.fixed_eval_enabled": args.wm_eval == "fixed",
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
        cfg.update({"trainer.rollout_before_train": True, "trainer.sim_rollout_epoch": 1,
                    "trainer.preserve_rollout_base_dir": True, "trainer.rollout_train_split": args.split,
                    "trainer.rollout_save_eval": False, "trainer.rollout_save_to_hdfs": True,
                    "trainer.rollout_do_sample": False, "actor_rollout_ref.rollout.temperature": 0.0,
                    "actor_rollout_ref.rollout_base_dir": str(args.shared_wm_eval.expanduser().resolve())})
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
    return cfg, root


def hydra_args(settings, extra):
    extra = extra[1:] if extra[:1] == ["--"] else extra
    protected = {"trainer.train_mode", "trainer.default_local_dir", "trainer.project_name",
                 "trainer.experiment_name", "trainer.n_gpus_per_node", "trainer.nnodes",
                 "actor_rollout_ref.wm_gpu_idx", "actor_rollout_ref.model.path",
                 "actor_rollout_ref.world_model.enable", "actor_rollout_ref.world_model.load_from_ckpt",
                 "actor_rollout_ref.world_model.ckpt_path", "trainer.resume.enable"}
    protected.update({"actor_rollout_ref.world_model.fine_tune", "actor_rollout_ref.world_model.config_path",
                      "actor_rollout_ref.world_model.fixed_eval_enabled", "actor_rollout_ref.world_model.fixed_eval_root",
                      "actor_rollout_ref.rollout.pretrained_checkpoint", "actor_rollout_ref.rollout.libero_pro_eval_config_path",
                      "actor_rollout_ref.rollout_base_dir", "trainer.ray_num_gpus", "trainer.ray_address",
                      "trainer.rollout_before_train", "trainer.val_only", "trainer.val_before_train",
                      "trainer.rollout_train_split", "trainer.preserve_rollout_base_dir",
                      "trainer.runtime_env", "data.task_suite_name", "actor_rollout_ref.rollout.task_suite_name",
                      "actor_rollout_ref.rollout.unnorm_key"})
    rendered = {k: f"++{k}={json.dumps(v, separators=(',', ':'))}" for k, v in settings.items()}
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
    if cfg.trainer.nnodes != 1:
        raise ValueError("the maintained launcher supports one ACP node")
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
                "MERL_LIBERO_EGL_DEVICE_ID": "0", "MERL_ENV_MP_START_METHOD": "spawn",
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
        print(json.dumps({"profile": "legacy_debug_not_paper_reproduction", "command": command}, indent=2))
        return
    resolved = compose_config(overrides)
    env = runtime_env()
    wm = resolved["actor_rollout_ref"]["world_model"]
    if platform.system() != "Linux":
        p.error("asset checks and execution require the Linux CCI/ACP environment")

    def check(script, *arguments):
        started = time.monotonic()
        print(f"[launch] {datetime.now(timezone.utc).isoformat()} preflight begin: {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
                       cwd=ROOT, env=env, check=True)
        print(f"[launch] preflight passed: {script}; elapsed_seconds={time.monotonic() - started:.1f}", flush=True)

    checkpoint = args.sft_checkpoint.expanduser().resolve()
    stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
    unnorm_key = args.suite if args.suite in stats else args.suite + "_no_noops"
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
        p.error(f"requires {required} visible GPUs on ACP; found {len(devices)}. Use --check on CCI")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"profile": "legacy_debug_not_paper_reproduction", "job": args.job,
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
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
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
