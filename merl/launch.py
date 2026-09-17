"""Portable entrypoint for existing four-GPU profiles; see docs/h100_runbook.md."""

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


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "MERL": "examples/train_merl_debug_4gpu_fix_pro_.sh",
    "MFRL": "examples/MFRL/train_mfrl_debug_4gpu_fix_pro_.sh",
    "MBRL": "examples/MBRL/train_mbrl_debug_4gpu_fix_pro_.sh",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=tuple(SCRIPTS))
    parser.add_argument("--sft-checkpoint", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, default=ROOT / "configs/evaluation_config.yaml")
    parser.add_argument("--wm-config", type=Path, default=ROOT / "configs/wm_online_config.py")
    parser.add_argument("--shared-wm-eval", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--suite", default="libero_10", choices=("libero_10", "libero_spatial", "libero_object", "libero_goal"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides after --")
    args = parser.parse_args()
    if not args.experiment or args.experiment in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in args.experiment):
        parser.error("experiment must be a single ASCII name (letters, digits, '-', '_', '.')")
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    # These settings determine paths/mode already checked by this wrapper.
    protected = ("trainer.train_mode", "actor_rollout_ref.world_model.enable",
                 "actor_rollout_ref.model.path", "trainer.default_local_dir",
                 "trainer.experiment_name", "trainer.project_name", "trainer.n_gpus_per_node",
                 "trainer.nnodes", "actor_rollout_ref.wm_gpu_idx")
    if any(item.lstrip("+").split("=", 1)[0] in protected for item in overrides):
        parser.error("use wrapper options for mode/assets/output; four-GPU layout is fixed")
    selected = {
        "SFT_MODEL_PATH": str(args.sft_checkpoint.expanduser().resolve()),
        "CKPT_PATH": str(args.output_root.expanduser().resolve()),
        "DATASET_NAME": args.suite,
        "EXPERIMENT_NAME": args.experiment,
        "libero_pro_eval_config_path": str(args.eval_config.expanduser().resolve()),
        "world_model_config_path": str(args.wm_config.expanduser().resolve()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
        "NUM_GPUS": "3", "WM_GPU_IDX": "3", "AUTO_ADJUST_GPU_LAYOUT": "false",
        "RESUME_ENABLE": "false",
    }
    if args.shared_wm_eval:
        selected["shared_wm_eval_root"] = str(args.shared_wm_eval.expanduser().resolve())
    command = ["bash", str(ROOT / SCRIPTS[args.mode]), *overrides]
    print(json.dumps({"command": command, "environment_overrides": selected,
                      "profile": "legacy_debug_not_paper_reproduction"}, indent=2), flush=True)
    if args.dry_run:
        return
    if platform.system() != "Linux":
        parser.error("training requires a Linux CUDA host; use --dry-run locally")
    for path in (Path(selected["SFT_MODEL_PATH"]) / "dataset_statistics.json", args.eval_config,
                 *([args.wm_config] if args.mode != "MFRL" else [])):
        if not path.is_file():
            parser.error(f"missing required asset: {path}")
    env = dict(os.environ, **selected)
    # Use exactly the Python environment that invoked this module in the shell.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    probe = subprocess.run([sys.executable, "-c",
        "import json,torch; print(json.dumps([dict(name=torch.cuda.get_device_name(i),memory=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]))"],
        env=env, check=True, capture_output=True, text=True)
    devices = json.loads(probe.stdout)
    if len(devices) != 4 or any(d["memory"] < 79_000_000_000 for d in devices):
        parser.error(f"expected four visible 80GB GPUs; observed {devices}")
    run_dir = Path(selected["CKPT_PATH"]) / args.mode / args.experiment
    if run_dir.exists():
        parser.error(f"run directory already exists; use a new experiment name: {run_dir}")
    versions = {}
    for name in ("torch", "transformers", "diffusers", "ray", "accelerate", "numpy", "tensordict"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    run_dir.mkdir(parents=True)
    manifest = dict(schema_version=1, profile="legacy_debug_not_paper_reproduction",
                    started_utc=datetime.now(timezone.utc).isoformat(), git_commit=commit,
                    git_dirty=bool(dirty), command=command, environment_overrides=selected,
                    python=sys.version, packages=versions, devices=devices,
                    source_hashes={str(path): digest(path) for path in
                                   (ROOT / SCRIPTS[args.mode], args.eval_config,
                                    *([args.wm_config] if args.mode != "MFRL" else []))})
    manifest_path = run_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        code = subprocess.call(command, env=env, cwd=ROOT)
    except BaseException:
        manifest["status"] = "interrupted_or_launch_error"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise
    manifest.update(exit_code=code, ended_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
