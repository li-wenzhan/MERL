"""Train a visual simulator and progress proxy from stored LIBERO trajectories."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time

from .algorithm import MERLConfig, load_grounded_trajectory
from .checkpoint import atomic_save, restore_rng, rng_state

ROOT = Path(__file__).resolve().parents[1]


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def training_data(directory, config):
    """Accept only aligned training trajectories; never consume evaluation NPZs."""
    paths = sorted(Path(directory).expanduser().resolve().rglob("*.npz"))
    if not paths:
        raise ValueError("no trajectory NPZs found; collect --split wm_train first")
    items = [load_grounded_trajectory(path, config) for path in paths]
    if not any(len(item.actions) > config.history_size for item in items):
        raise ValueError("training trajectories need more actions than the history length")
    ids = [item.trajectory_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate training trajectory IDs")
    inventory = {str(path): file_digest(path) for path in paths}
    return items, inventory


def check_resume(payload, contract):
    if payload.get("format_version") != 1 or payload.get("contract") != contract:
        raise ValueError("resume requires identical data, model settings and training configuration")
    if type(payload.get("step")) is not int or payload["step"] < 1:
        raise ValueError("resume snapshot must contain a completed positive training step")


def save_training(model, optimizer, output, step, contract):
    """Publish the resume sidecar last, after its model checkpoint is complete."""
    weights = output / f"checkpoint-{step}.pt"
    state = output / f"checkpoint-{step}.train_state.pt"
    atomic_save(model.state_dict(), weights)
    atomic_save(dict(format_version=1, step=step, contract=contract,
                     checkpoint=weights.name, optimizer=optimizer.state_dict(), rng=rng_state()), state)
    temporary = output / "latest.json.tmp"
    temporary.write_text(json.dumps(dict(step=step, checkpoint=weights.name,
                                         resume_from=state.name), indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "latest.json")
    return weights, state


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="NPZ trajectories from --split wm_train")
    p.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    p.add_argument("--config", type=Path, default=ROOT / "configs/merl.json")
    p.add_argument("--wm-config", type=Path, default=ROOT / "configs/wm_online_config.py")
    p.add_argument("--steps", type=int, default=5000, help="Absolute target optimizer step, including resumed steps")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--checkpoint-keep", type=int, default=2, help="Number of snapshots retained in this run; 0 keeps all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--from-checkpoint", type=Path, help="Initialize model weights without optimizer state")
    initialization.add_argument("--resume-from", type=Path, help="Restore a .train_state.pt snapshot")
    return p


def main():
    p = parser()
    args = p.parse_args()
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    if min(args.steps, args.save_every) < 1 or args.checkpoint_keep < 0 or args.seed < 0:
        p.error("steps and save-every must be positive; seed and checkpoint-keep must be nonnegative")
    if not 0 < args.learning_rate < float("inf"):
        p.error("learning-rate must be positive and finite")
    output = args.output.expanduser().resolve()
    if output.exists():
        p.error("--output must be a fresh directory; resume into a new directory")
    config = replace(MERLConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8"))),
                     seed=args.seed, simulator_steps=1).for_mode("ONLINE_MBRL")
    items, inventory = training_data(args.data, config)
    spec = importlib.util.spec_from_file_location("merl_simulator_config", args.wm_config)
    if spec is None or spec.loader is None:
        p.error("unable to import --wm-config")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wm_args = module.wm_args()
    wm_args.mixed_precision, wm_args.dtype = "bf16", "torch.bfloat16"
    wm_args.__post_init__()
    wm_args.is_img_pregenerated, wm_args.self_forcing_weight = False, 0.0
    wm_args.learning_rate = args.learning_rate
    wm_args.load_from_ckpt = bool(args.from_checkpoint or args.resume_from)
    contract = dict(config=asdict(config), data_sha256=inventory,
                    wm_config_sha256=file_digest(args.wm_config), learning_rate=args.learning_rate,
                    svd_model_path=str(Path(wm_args.svd_model_path).resolve()),
                    clip_model_path=str(Path(wm_args.clip_model_path).resolve()))
    payload, weights, step = None, args.from_checkpoint, 0
    if args.resume_from:
        payload = torch.load(args.resume_from.resolve(), map_location="cpu", weights_only=True)
        check_resume(payload, contract)
        weights = args.resume_from.resolve().parent / payload["checkpoint"]
        if not weights.resolve().is_relative_to(args.resume_from.resolve().parent):
            p.error("resume checkpoint must belong to the snapshot directory")
        step = payload["step"]
        if step >= args.steps:
            p.error("--steps must exceed the completed resume step")
    if weights is not None and not weights.is_file():
        p.error(f"missing simulator checkpoint: {weights}")
    if weights is None and os.environ.get("HF_HUB_OFFLINE") == "1":
        from torchvision.models import ResNet18_Weights
        cache = Path(torch.hub.get_dir()) / "checkpoints" / ResNet18_Weights.DEFAULT.url.rsplit("/", 1)[-1]
        if not cache.is_file():
            p.error("cache ResNet18_Weights.DEFAULT under TORCH_HOME before offline simulator initialization")
    accelerator = Accelerator(mixed_precision="bf16")
    if accelerator.device.type != "cuda" or accelerator.num_processes != 1:
        p.error("simulator initialization uses one CUDA GPU; launch with python, not torchrun")
    from modules.ctrl_world.model_loading import load_trusted_state_dict
    from modules.ctrl_world.models.ctrl_world_new import CtrlWorld
    from .simulator import Simulator

    set_seed(config.seed)
    model = CtrlWorld(wm_args)
    if weights is not None:
        model.load_state_dict(load_trusted_state_dict(str(weights), map_location="cpu"), strict=True)
    model.to(dtype=torch.bfloat16)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                 lr=args.learning_rate)
    model, optimizer = accelerator.prepare(model, optimizer)
    simulator = Simulator(accelerator.unwrap_model(model), wm_args, accelerator.device,
                          asdict(config), "ONLINE_MBRL")
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
        del payload
    output.mkdir(parents=True)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), contract=contract,
                    target_steps=args.steps, completed_steps=step, trajectories=len(items),
                    initialization=str(weights) if weights else "svd_clip_imagenet_backbones",
                    source_hashes={str(path.relative_to(ROOT)): file_digest(path) for path in (
                        Path(__file__), ROOT / "merl/algorithm.py", ROOT / "merl/simulator.py",
                        ROOT / "merl/checkpoint.py", ROOT / "modules/ctrl_world/models/ctrl_world_new.py")},
                    packages={name: importlib.metadata.version(name) for name in (
                        "torch", "accelerate", "diffusers", "transformers", "numpy")},
                    device=torch.cuda.get_device_name(accelerator.device),
                    wm_args=wm_args.to_dict(), status="running")
    owned = []
    manifest_path = output / "manifest.json"

    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    write_manifest()
    started = time.monotonic()
    try:
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
            while step < args.steps:
                simulator.stage = step + 1
                metrics = simulator.update(items, optimizer, accelerator)
                step += 1
                record = dict(step=step, elapsed_seconds=time.monotonic() - started, **metrics)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if step % args.save_every == 0 or step == args.steps:
                    owned.append(save_training(accelerator.unwrap_model(model), optimizer, output, step, contract))
                    if args.checkpoint_keep:
                        while len(owned) > args.checkpoint_keep:
                            for path in owned.pop(0):
                                path.unlink()
                    manifest["completed_steps"] = step
                    write_manifest()
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "interrupted_or_error"
        raise
    finally:
        manifest["executed_steps"] = step
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["peak_gpu_bytes"] = torch.cuda.max_memory_allocated(accelerator.device)
        write_manifest()


if __name__ == "__main__":
    main()
