from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
CTRL_WORLD_ROOT = REPO_ROOT / "modules" / "ctrl_world"
for path in (REPO_ROOT, CTRL_WORLD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.wm_offline_config import wm_args
from modules.ctrl_world.models.ctrl_world_new import CtrlWorld
from real_world.world_model.hdf5_finetune_dataset import RealWorldHDF5WindowDataset

TRAIN_STATE_FORMAT = "real_world_wm_hdf5_train_state_v1"
CHECKPOINT_STEP_RE = re.compile(r"checkpoint-(\d+)")


def _str_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss_scalar(
    loss_dict: dict[str, torch.Tensor], total_loss: torch.Tensor
) -> dict[str, float]:
    payload = {"loss_total": float(total_loss.detach().float().cpu())}
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            payload[key] = float(value.detach().float().cpu())
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_STEP_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _sidecar_train_state_path(model_ckpt_path: Path) -> Path:
    return model_ckpt_path.with_name(f"{model_ckpt_path.stem}.train_state.pt")


def _model_ckpt_from_train_state(
    train_state_path: Path, payload: dict[str, Any]
) -> Path:
    model_checkpoint = payload.get("model_checkpoint")
    if model_checkpoint:
        model_ckpt_path = Path(str(model_checkpoint))
        if not model_ckpt_path.is_absolute():
            model_ckpt_path = train_state_path.parent / model_ckpt_path
        return model_ckpt_path
    if train_state_path.name.endswith(".train_state.pt"):
        return train_state_path.with_name(
            train_state_path.name.replace(".train_state.pt", ".pt")
        )
    raise ValueError(
        f"cannot infer model checkpoint from train-state file: {train_state_path}"
    )


def _is_train_state_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return payload.get("format") == TRAIN_STATE_FORMAT or (
        "optimizer_state" in payload and "global_step" in payload
    )


def _coerce_model_state_dict(
    payload: Any, *, source_path: Path
) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            state_dict = payload.get(key)
            if isinstance(state_dict, dict):
                return state_dict
        if all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise ValueError(
        f"unsupported model checkpoint payload in {source_path}; expected a pure state_dict "
        "or a dict containing state_dict/model_state."
    )


def _load_model_checkpoint(
    model: torch.nn.Module, ckpt_path: Path, *, label: str
) -> None:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing {label}: {ckpt_path}")
    payload = _torch_load(ckpt_path)
    if _is_train_state_payload(payload):
        model_ckpt_path = _model_ckpt_from_train_state(ckpt_path, payload)
        print(
            f"[real-wm-finetune] {label} is train state; loading model from {model_ckpt_path}"
        )
        payload = _torch_load(model_ckpt_path)
        ckpt_path = model_ckpt_path
    state_dict = _coerce_model_state_dict(payload, source_path=ckpt_path)
    model.load_state_dict(state_dict, strict=True)


def _load_resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_path: Path,
) -> int:
    if not resume_path.is_file():
        raise FileNotFoundError(f"missing --resume-from checkpoint: {resume_path}")

    payload = _torch_load(resume_path)
    train_state_payload: dict[str, Any] | None = None
    model_ckpt_path = resume_path

    if _is_train_state_payload(payload):
        train_state_payload = payload
        model_ckpt_path = _model_ckpt_from_train_state(resume_path, payload)
    else:
        sidecar_path = _sidecar_train_state_path(resume_path)
        if sidecar_path.is_file():
            sidecar_payload = _torch_load(sidecar_path)
            if _is_train_state_payload(sidecar_payload):
                train_state_payload = sidecar_payload

    print(f"[real-wm-finetune] Resuming model from {model_ckpt_path}")
    _load_model_checkpoint(model, model_ckpt_path, label="resume model checkpoint")

    global_step = _parse_checkpoint_step(model_ckpt_path) or 0
    if train_state_payload is not None:
        global_step = int(train_state_payload.get("global_step", global_step))
        optimizer_state = train_state_payload.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print(
                f"[real-wm-finetune] Restored optimizer state from step {global_step}"
            )
        else:
            print(
                "[real-wm-finetune] Train state has no optimizer_state; resuming with a fresh optimizer."
            )
    else:
        print(
            "[real-wm-finetune] No sidecar .train_state.pt found; "
            f"resuming from model weights only at inferred step {global_step}."
        )
    return global_step


def _checkpoint_model_paths(output_dir: Path) -> list[tuple[int, Path]]:
    entries: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*.pt"):
        if path.name.endswith(".train_state.pt") or path.name == "checkpoint-final.pt":
            continue
        step = _parse_checkpoint_step(path)
        if step is not None:
            entries.append((step, path))
    return sorted(entries, key=lambda item: item[0], reverse=True)


def _rotate_checkpoints(output_dir: Path, max_keep: int) -> None:
    if max_keep <= 0:
        return
    for _, model_ckpt_path in _checkpoint_model_paths(output_dir)[max_keep:]:
        sidecar_path = _sidecar_train_state_path(model_ckpt_path)
        for path in (model_ckpt_path, sidecar_path):
            if path.exists():
                path.unlink()
                print(f"[real-wm-finetune] removed old checkpoint {path}")


def _save_training_checkpoint(
    output_dir: Path,
    args: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    *,
    global_step: int,
    max_keep_checkpoints: int,
) -> None:
    model_ckpt_path = output_dir / f"checkpoint-{global_step}.pt"
    train_state_path = _sidecar_train_state_path(model_ckpt_path)
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), model_ckpt_path)
    torch.save(
        {
            "format": TRAIN_STATE_FORMAT,
            "global_step": int(global_step),
            "model_checkpoint": model_ckpt_path.name,
            "optimizer_state": optimizer.state_dict(),
        },
        train_state_path,
    )
    _write_infer_config(
        output_dir / "real_wm_infer_finetuned.yaml", args, model_ckpt_path
    )
    _write_json(
        output_dir / "latest_checkpoint.json",
        {
            "global_step": int(global_step),
            "model_checkpoint": str(model_ckpt_path),
            "train_state": str(train_state_path),
            "max_keep_checkpoints": int(max_keep_checkpoints),
        },
    )
    print(f"[real-wm-finetune] saved {model_ckpt_path}")
    print(f"[real-wm-finetune] saved {train_state_path}")
    _rotate_checkpoints(output_dir, int(max_keep_checkpoints))


def _write_infer_config(path: Path, args: Any, ckpt_path: Path) -> None:
    payload = {
        "wm_config_path": "configs/wm_online_config.py",
        "svd_model_path": args.svd_model_path,
        "clip_model_path": args.clip_model_path,
        "ckpt_path": str(ckpt_path),
        "load_from_ckpt": True,
        "device": "cuda:0",
        "dtype": args.dtype,
        "width": int(args.width),
        "height": int(args.height),
        "num_history": int(args.num_history),
        "num_future_frames": int(args.num_frames),
        "num_inference_steps": int(args.num_inference_steps),
        "decode_chunk_size": int(args.decode_chunk_size),
        "guidance_scale": float(args.guidance_scale),
        "fps": int(args.fps),
        "output_fps": int(args.fps),
        "motion_bucket_id": int(args.motion_bucket_id),
        "frame_level_cond": bool(args.frame_level_cond),
        "his_cond_zero": bool(args.his_cond_zero),
        "reward_threshold": float(getattr(args, "reward_threshold", 0.5)),
        "action_dim": int(args.action_dim),
        "action_input_range": "model",
        "binarize_gripper": True,
        "invert_gripper": False,
        "zero_history_actions_if_missing": True,
        "save_input_history": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False, allow_unicode=True)


def _build_total_loss(
    loss_dict: dict[str, torch.Tensor],
    *,
    self_forcing_weight: float,
    reward_loss_weight: float,
) -> torch.Tensor:
    device = next(iter(loss_dict.values())).device
    total = loss_dict["loss_noise"]
    total = total + float(reward_loss_weight) * loss_dict.get(
        "loss_reward", torch.zeros((), device=device)
    )
    total = total + float(self_forcing_weight) * loss_dict.get(
        "loss_self_forcing", torch.zeros((), device=device)
    )
    return total


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    args: Any,
    *,
    max_batches: int,
    reward_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        with accelerator.autocast():
            loss_dict, _ = model(batch)
            total = _build_total_loss(
                loss_dict,
                self_forcing_weight=args.self_forcing_weight,
                reward_loss_weight=reward_loss_weight,
            )
        metrics = _loss_scalar(loss_dict, total)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
        count += 1
    model.train()
    if count == 0:
        return {}
    return {f"val_{key}": value / count for key, value in sums.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Fine-tune Ctrl-World with real-world HDF5/H5 data."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Directory of .h5/.hdf5 files or one merged .h5 file.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--svd-model-path", default=None)
    parser.add_argument("--clip-model-path", default=None)
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Warm-start Ctrl-World checkpoint with reward model.",
    )
    parser.add_argument("--instruction", default="Put the red block into the box.")
    parser.add_argument("--image-key", default="observations/images/cam_high")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--action-slice", default="right")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--action-offset", type=float, default=0.0)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--num-future", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--action-stride", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-train-steps", type=int, default=5000)
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--max-keep-checkpoints", type=int, default=2)
    parser.add_argument("--validation-steps", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--mixed-precision", default="fp16", choices=["no", "fp16", "bf16"]
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--self-forcing-weight", type=float, default=1.0)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--decode-chunk-size", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--motion-bucket-id", type=int, default=127)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Resume training from checkpoint-N.pt or checkpoint-N.train_state.pt. "
            "The model-only .pt remains inference-compatible; the sidecar train-state restores optimizer/global_step."
        ),
    )
    parser.add_argument("--save-final", action="store_true")
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    _set_seed(int(cli.seed))
    output_dir = Path(cli.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    args = wm_args()
    args.output_dir = str(output_dir)
    args.tag = output_dir.name
    args.wandb_run_name = output_dir.name
    args.svd_model_path = _str_or_none(cli.svd_model_path) or args.svd_model_path
    args.clip_model_path = _str_or_none(cli.clip_model_path) or args.clip_model_path
    args.ckpt_path = _str_or_none(cli.ckpt_path) or _str_or_none(args.ckpt_path)
    args.is_img_pregenerated = False
    args.num_views = 1
    args.img_resizes = (int(args.height), int(args.width))
    args.num_history = int(cli.num_history)
    args.num_frames = int(cli.num_future)
    args.action_dim = 7
    args.fps = int(cli.fps)
    args.learning_rate = float(cli.learning_rate)
    args.gradient_accumulation_steps = int(cli.gradient_accumulation_steps)
    args.mixed_precision = cli.mixed_precision
    args.train_batch_size = int(cli.batch_size)
    args.num_workers = int(cli.num_workers)
    args.max_train_steps = int(cli.max_train_steps)
    args.checkpointing_steps = int(cli.checkpointing_steps)
    args.max_keep_checkpoints = int(cli.max_keep_checkpoints)
    args.validation_steps = int(cli.validation_steps)
    args.max_grad_norm = float(cli.max_grad_norm)
    args.self_forcing_weight = float(cli.self_forcing_weight)
    args.num_inference_steps = int(cli.num_inference_steps)
    args.decode_chunk_size = int(cli.decode_chunk_size)
    args.guidance_scale = float(cli.guidance_scale)
    args.motion_bucket_id = int(cli.motion_bucket_id)
    args.seed = int(cli.seed)

    _write_json(output_dir / "train_hdf5_finetune_args.json", vars(cli))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
    )
    logger = get_logger(__name__, log_level="INFO")

    train_dataset = RealWorldHDF5WindowDataset(
        cli.data_root,
        mode="train",
        image_key=cli.image_key,
        action_key=cli.action_key,
        action_slice=cli.action_slice,
        instruction=cli.instruction,
        num_history=args.num_history,
        num_future=args.num_frames,
        frame_stride=cli.frame_stride,
        action_stride=cli.action_stride,
        sample_stride=cli.sample_stride,
        image_size=args.img_resizes,
        val_ratio=cli.val_ratio,
        seed=cli.seed,
        action_scale=cli.action_scale,
        action_offset=cli.action_offset,
        max_windows=cli.max_train_windows,
    )
    val_dataset = RealWorldHDF5WindowDataset(
        cli.data_root,
        mode="val",
        image_key=cli.image_key,
        action_key=cli.action_key,
        action_slice=cli.action_slice,
        instruction=cli.instruction,
        num_history=args.num_history,
        num_future=args.num_frames,
        frame_stride=cli.frame_stride,
        action_stride=cli.action_stride,
        sample_stride=cli.sample_stride,
        image_size=args.img_resizes,
        val_ratio=cli.val_ratio,
        seed=cli.seed,
        action_scale=cli.action_scale,
        action_offset=cli.action_offset,
        max_windows=cli.max_val_windows,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) >= args.train_batch_size,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = CtrlWorld(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    resume_from = _str_or_none(cli.resume_from)
    global_step = 0
    if resume_from is not None:
        global_step = _load_resume_checkpoint(
            model, optimizer, Path(resume_from).expanduser()
        )
    elif args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path).expanduser()
        print(
            f"[real-wm-finetune] Loading warm-start Ctrl-World checkpoint: {ckpt_path}"
        )
        _load_model_checkpoint(model, ckpt_path, label="--ckpt-path")
    model.train()

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    if accelerator.is_main_process:
        _write_json(
            output_dir / "dataset_summary.json",
            {
                "data_root": cli.data_root,
                "train_windows": len(train_dataset),
                "val_windows": len(val_dataset),
                "image_key": cli.image_key,
                "action_key": cli.action_key,
                "action_slice": cli.action_slice,
                "instruction": cli.instruction,
                "reward_definition": "shape [T], last frame = 1, all previous frames = 0",
            },
        )

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Real-world HDF5 Ctrl-World fine-tuning *****")
    logger.info(f"  Train windows = {len(train_dataset)}")
    logger.info(f"  Val windows = {len(val_dataset)}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Max train steps = {args.max_train_steps}")
    logger.info(f"  Resume global step = {global_step}")

    progress = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Real-WM Steps",
    )

    while global_step < args.max_train_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_dict, _ = model(batch)
                    total_loss = _build_total_loss(
                        loss_dict,
                        self_forcing_weight=args.self_forcing_weight,
                        reward_loss_weight=float(cli.reward_loss_weight),
                    )
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if accelerator.is_local_main_process:
                    progress.set_postfix(_loss_scalar(loss_dict, total_loss))

                if global_step % 50 == 0 and accelerator.is_main_process:
                    metrics = _loss_scalar(loss_dict, total_loss)
                    print(f"[real-wm-finetune] step={global_step} metrics={metrics}")

                if (
                    args.validation_steps > 0
                    and global_step % args.validation_steps == 0
                ):
                    metrics = _validate(
                        model,
                        val_loader,
                        accelerator,
                        args,
                        max_batches=int(cli.validation_batches),
                        reward_loss_weight=float(cli.reward_loss_weight),
                    )
                    if metrics and accelerator.is_main_process:
                        print(
                            f"[real-wm-finetune] validation step={global_step} metrics={metrics}"
                        )

                if (
                    args.checkpointing_steps > 0
                    and global_step % args.checkpointing_steps == 0
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        _save_training_checkpoint(
                            output_dir,
                            args,
                            model,
                            optimizer,
                            accelerator,
                            global_step=global_step,
                            max_keep_checkpoints=int(cli.max_keep_checkpoints),
                        )

                if global_step >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and cli.save_final:
        final_ckpt = output_dir / "checkpoint-final.pt"
        torch.save(accelerator.unwrap_model(model).state_dict(), final_ckpt)
        _write_infer_config(
            output_dir / "real_wm_infer_finetuned.yaml", args, final_ckpt
        )
        print(f"[real-wm-finetune] saved final checkpoint {final_ckpt}")


if __name__ == "__main__":
    main()
