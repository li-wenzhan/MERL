"""Compare simulator checkpoints on identical stored observations and executed actions."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_reference(path):
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if (not meta.get("valid") or meta.get("split") != "evaluation"
            or meta.get("action_convention") != "libero_env_executed"
            or meta.get("alignment") != "observations[t+1] follows actions[t]"
            or meta.get("observation_source") != "real_environment"):
        raise ValueError("Use a valid held-out real-environment episode with explicit executed-action alignment")
    trajectory = path.parent / meta["trajectory"]
    with np.load(trajectory, allow_pickle=False) as archive:
        observations, actions = archive["observations"], archive["actions"]
    if (observations.dtype != np.uint8 or observations.ndim != 4 or observations.shape[-1] != 3
            or actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all()
            or len(observations) != len(actions) + 1 or len(actions) != meta["environment_steps"]):
        raise ValueError("Expected uint8 RGB observations[T+1] and finite executed actions[T,7]")
    return observations, actions, meta, trajectory


def predict_reference(observations, actions, *, start, horizon, history_size, chunk_size,
                      rollout, seed, predict):
    """Future GT is used only for targets, never as recursive prediction input."""
    if rollout not in ("recursive", "teacher_forced"):
        raise ValueError("Unknown rollout protocol")
    if min(history_size, chunk_size, horizon) < 1 or start < history_size or start + horizon > len(actions):
        raise ValueError("Requested window exceeds the stored trajectory or lacks real history")
    history = observations[start - history_size + 1:start + 1].copy()
    predicted, scores = [], []
    for offset in range(0, horizon, chunk_size):
        cursor = start + offset
        count = min(chunk_size, horizon - offset)
        if rollout == "teacher_forced":
            history = observations[cursor - history_size + 1:cursor + 1].copy()
        frames, proxy = predict(history.copy(), actions[cursor - history_size:cursor].copy(),
                                actions[cursor:cursor + count].copy(), seed + offset // chunk_size)
        frames, proxy = np.asarray(frames), np.asarray(proxy).reshape(-1)
        if (frames.shape != (count, *observations.shape[1:]) or frames.dtype != np.uint8
                or len(proxy) != count or not np.isfinite(proxy).all()):
            raise ValueError("Simulator returned invalid frames or proxy scores")
        predicted.append(frames)
        scores.append(proxy)
        history = np.concatenate([history, frames], axis=0)[-history_size:]
    return np.concatenate(predicted), np.concatenate(scores)


def visual_metrics(predicted, target):
    error = ((predicted.astype(np.float64) - target.astype(np.float64)) / 255.0) ** 2
    mse_per_frame = error.mean(axis=(1, 2, 3))
    mse = float(mse_per_frame.mean())
    return dict(pixel_mse=mse, psnr_db=(-10 * float(np.log10(mse))) if mse > 0 else None,
                identical_pixels=mse == 0, frame_mse=mse_per_frame.tolist())


def save_comparison(root, labels, videos, start, rollout):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from merl.presentation_report import font
    root = Path(root)
    frames = len(videos[0])
    selected = set(int(i) for i in np.linspace(0, frames - 1, min(5, frames)))
    panels = []
    with imageio.get_writer(str(root / "comparison.mp4"), fps=8) as writer:
        for index in range(frames):
            canvas = Image.new("RGB", (320 * len(labels), 240), "#101827")
            draw = ImageDraw.Draw(canvas)
            for column, (label, video) in enumerate(zip(labels, videos)):
                x = column * 320
                draw.text((x + 8, 3), label, font=font(19), fill="white")
                draw.text((x + 8, 27), f"t={start + index + 1} | {rollout} | fixed actions", font=font(12), fill="#bdd8ef")
                canvas.paste(Image.fromarray(video[index]), (x, 48))
            writer.append_data(np.asarray(canvas))
            if index in selected:
                canvas.save(root / f"frame_{start + index + 1:05d}.png")
                panels.append(canvas)
    sheet = Image.new("RGB", (panels[0].width, 240 * len(panels)))
    for index, frame in enumerate(panels):
        sheet.paste(frame, (0, index * 240))
    sheet.save(root / "contact_sheet.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True, help="Saved episode.json with trajectory.npz")
    p.add_argument("--checkpoint", action="append", required=True, help="LABEL=/path/to/WM.pt; repeat for each model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wm-config", default="configs/wm_online_config.py")
    p.add_argument("--start", type=int, default=64)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--rollout", choices=("recursive", "teacher_forced"), default="recursive")
    p.add_argument("--inference-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    observations, actions, meta, trajectory = load_reference(args.episode)
    checkpoints = {}
    for entry in args.checkpoint:
        label, separator, path = entry.partition("=")
        if not separator or label not in ("MBRL", "ONLINE_MBRL", "MERL") or label in checkpoints:
            p.error("Use unique MBRL, ONLINE_MBRL or MERL checkpoint labels")
        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.is_file():
            p.error(f"Missing actual trained checkpoint: {checkpoint}")
        checkpoints[label] = checkpoint
    if args.inference_steps <= 0:
        p.error("--inference-steps must be positive")

    # Importing the adapter does not allocate a model; reject invalid windows first.
    from real_world.world_model.ctrl_world_adapter import CtrlWorldRealAdapter, _load_wm_args
    from real_world.world_model.schemas import RealWorldWMConfig
    from real_world.world_model.preprocess import rgb_frames_to_tensor
    from PIL import Image
    import torch
    from merl.launch import runtime_env
    import os
    os.environ.update(runtime_env())
    wm = _load_wm_args(args.wm_config)
    history_size, chunk_size = int(wm.num_history), int(wm.num_frames)
    if args.start < history_size or args.horizon <= 0 or args.start + args.horizon > len(actions):
        p.error("Select a window with sufficient stored history and no padded future")
    if not torch.cuda.is_available():
        p.error("A CUDA GPU is required for checkpoint inference")
    if (int(wm.width), int(wm.height)) != (320, 192):
        p.error("The current comparison renderer expects the configured 320x192 Ctrl-World backbone")
    observations = np.stack([np.asarray(Image.fromarray(frame).resize((320, 192), Image.Resampling.BILINEAR))
                             for frame in observations])
    target = observations[args.start + 1:args.start + args.horizon + 1]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", episode=str(args.episode.resolve()), episode_sha256=file_digest(args.episode),
                    trajectory_sha256=file_digest(trajectory), protocol_id=meta["protocol_id"],
                    task_id=meta["task_id"], trial_id=meta["trial_id"], rollout=args.rollout,
                    start=args.start, horizon=args.horizon, history_size=history_size, chunk_size=chunk_size,
                    seed=args.seed, inference_steps=args.inference_steps,
                    action_convention="libero_env_executed; no further gripper conversion",
                    preprocessing="RGB uint8 PIL bilinear resize to 320x192",
                    wm_config_sha256=file_digest(args.wm_config), source_sha256=file_digest(__file__),
                    caveat="Held-out fixed-action simulator fidelity; this does not measure closed-loop policy success or trust admission.",
                    results={})
    def save():
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    save()
    labels, videos = ["GT"], [target]
    np.save(root / "gt.npy", target, allow_pickle=False)
    try:
        for label, checkpoint in checkpoints.items():
            started = time.monotonic()
            sha = file_digest(checkpoint)
            if any(row["checkpoint_sha256"] == sha for row in manifest["results"].values()):
                raise ValueError("Identical checkpoints cannot stand in for different trained methods")
            config = RealWorldWMConfig(wm_config_path=args.wm_config, ckpt_path=str(checkpoint), device=args.device,
                                      num_history=history_size, num_future_frames=chunk_size,
                                      num_inference_steps=args.inference_steps, guidance_scale=wm.guidance_scale,
                                      fps=wm.fps, motion_bucket_id=wm.motion_bucket_id,
                                      frame_level_cond=wm.frame_level_cond, his_cond_zero=wm.his_cond_zero)
            adapter = CtrlWorldRealAdapter(config)
            def predict(history, past, future, seed):
                torch.manual_seed(seed)
                images = rgb_frames_to_tensor(list(history), height=192, width=320)
                frames, scores, _ = adapter.predict(current_image=images[-1], history_images=images,
                                                    history_actions=past, future_actions=future,
                                                    task_instruction=meta["instruction"])
                return frames, scores
            frames, scores = predict_reference(observations, actions, start=args.start, horizon=args.horizon,
                                                history_size=history_size, chunk_size=chunk_size,
                                                rollout=args.rollout, seed=args.seed, predict=predict)
            np.savez_compressed(root / f"{label}.npz", frames=frames, proxy_scores=scores)
            manifest["results"][label] = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha,
                                               elapsed_seconds=time.monotonic() - started,
                                               **visual_metrics(frames, target))
            labels.append(label)
            videos.append(frames)
            save()
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
        save_comparison(root, labels, videos, args.start, args.rollout)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        save()
    print(f"[wm comparison] {root / 'comparison.mp4'}", flush=True)


if __name__ == "__main__":
    main()
