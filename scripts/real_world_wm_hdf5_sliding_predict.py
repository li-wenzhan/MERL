from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_world.world_model.io import save_video, write_json
from real_world.world_model.predictor import RealWorldWMPredictor
from real_world.world_model.schemas import RealWorldWMConfig, RealWorldWMRequest
from scripts.extract_real_world_wm_hdf5 import (
    _find_action_dataset,
    _find_image_dataset,
    _load_actions,
    _load_frames,
    _save_image_sequence,
)


def _load_config(path: str) -> RealWorldWMConfig:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as file_obj:
        payload: dict[str, Any] = yaml.safe_load(file_obj) or {}
    return RealWorldWMConfig.from_dict(payload)


def _window_indices(
    *,
    current_frame: int,
    num_history: int,
    num_future: int,
    frame_stride: int,
    action_stride: int,
) -> tuple[list[int], list[int], list[int]]:
    input_frame_indices = list(
        range(current_frame - num_history * frame_stride, current_frame + 1, frame_stride)
    )
    history_action_indices = list(
        range(current_frame - num_history * action_stride, current_frame, action_stride)
    )
    future_indices = list(
        range(current_frame, current_frame + num_future * action_stride, action_stride)
    )
    return input_frame_indices, history_action_indices + future_indices, future_indices


def _validate_indices(
    *,
    indices: list[int],
    count: int,
    label: str,
) -> None:
    if not indices:
        raise ValueError(f"{label} indices are empty")
    if indices[0] < 0 or indices[-1] >= count:
        raise ValueError(
            f"{label} indices out of range: first={indices[0]}, last={indices[-1]}, count={count}"
        )


def _stats(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Sliding HDF5 real-world world model prediction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="Put the red block into the box.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-key", default="observations/images/cam_high")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--action-slice", default="right")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True, help="Exclusive raw HDF5 frame end.")
    parser.add_argument("--chunk-step", type=int, default=None)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--num-future", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--action-stride", type=int, default=None)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--save-chunks", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--export-frames", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hdf5_path = Path(args.hdf5).expanduser()
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"missing HDF5 file: {hdf5_path}")

    start_frame = int(args.start_frame)
    end_frame = int(args.end_frame)
    if end_frame <= start_frame:
        raise ValueError(f"--end-frame must be greater than --start-frame, got {start_frame}..{end_frame}")

    action_stride = int(args.action_stride or args.frame_stride)
    chunk_step = int(args.chunk_step or (args.num_future * action_stride))
    if chunk_step <= 0:
        raise ValueError(f"chunk_step must be positive, got {chunk_step}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_dir / "sliding_chunks"
    if args.save_chunks:
        chunks_dir.mkdir(parents=True, exist_ok=True)
    save_videos = bool(args.save_videos) and not bool(args.skip_videos)

    config: Optional[RealWorldWMConfig] = None
    predictor: Optional[RealWorldWMPredictor] = None
    if not args.extract_only:
        config = _load_config(args.config)
        config.num_future_frames = int(args.num_future)
        config.output_fps = int(args.fps)
        if args.num_inference_steps is not None:
            config.num_inference_steps = int(args.num_inference_steps)
        if args.device is not None:
            config.device = args.device
        predictor = RealWorldWMPredictor.from_config(config)

    combined_pred_frames: list[np.ndarray] = []
    combined_gt_frames: list[np.ndarray] = []
    combined_scores: list[float] = []
    chunks: list[dict[str, Any]] = []

    with h5py.File(hdf5_path, "r") as h5_file:
        image_key, image_dataset = _find_image_dataset(h5_file, args.image_key)
        action_key, action_dataset = _find_action_dataset(h5_file, args.action_key)
        frame_count = int(image_dataset.shape[0])
        action_count = int(action_dataset.shape[0])

        for chunk_idx, current_frame in enumerate(range(start_frame, end_frame, chunk_step)):
            input_indices, action_indices, future_indices = _window_indices(
                current_frame=current_frame,
                num_history=int(args.num_history),
                num_future=int(args.num_future),
                frame_stride=int(args.frame_stride),
                action_stride=action_stride,
            )
            _validate_indices(indices=input_indices, count=frame_count, label="input_frame")
            _validate_indices(indices=future_indices, count=frame_count, label="future_frame")
            _validate_indices(indices=action_indices, count=action_count, label="action")

            keep_count = sum(1 for index in future_indices if index < end_frame)
            if keep_count <= 0:
                continue

            input_frames = _load_frames(image_dataset, input_indices)
            gt_future_frames = _load_frames(image_dataset, future_indices)
            actions = _load_actions(action_dataset, action_indices, args.action_slice)
            chunk_dir = chunks_dir / f"chunk_{chunk_idx:04d}"

            chunk_payload: dict[str, Any] = {
                "chunk_idx": int(chunk_idx),
                "task_instruction": args.task,
                "current_frame": int(current_frame),
                "input_frame_indices": [int(index) for index in input_indices],
                "history_frame_indices": [int(index) for index in input_indices[:-1]],
                "future_frame_indices": [int(index) for index in future_indices],
                "kept_future_frame_indices": [
                    int(index) for index in future_indices[:keep_count]
                ],
                "action_indices": [int(index) for index in action_indices],
                "action_stats": _stats(actions),
            }

            if args.save_chunks:
                chunk_dir.mkdir(parents=True, exist_ok=True)
                _save_image_sequence(str(chunk_dir / "history_frames"), input_frames[:-1])
                _save_image_sequence(str(chunk_dir / "gt_future_frames"), gt_future_frames)
                chunk_payload["history_frames_dir"] = str(chunk_dir / "history_frames")
                chunk_payload["history_frames_include_current"] = False
                chunk_payload["current_frame_image"] = "gt_future_frames/frame_000000.png"
                chunk_payload["gt_future_frames_dir"] = str(chunk_dir / "gt_future_frames")
                if save_videos:
                    save_video(str(chunk_dir / "gt_future.mp4"), gt_future_frames, fps=int(args.fps))

            combined_gt_frames.extend(gt_future_frames[:keep_count])

            if not args.extract_only:
                assert predictor is not None
                request = RealWorldWMRequest(
                    frames=input_frames,
                    future_actions=actions,
                    task_instruction=args.task,
                    fps=int(args.fps),
                    metadata={
                        "hdf5": str(hdf5_path),
                        "task_instruction": args.task,
                        "image_key": image_key,
                        "action_key": action_key,
                        "action_slice": args.action_slice,
                        "chunk_idx": int(chunk_idx),
                        "current_frame": int(current_frame),
                    },
                )
                result = predictor.predict(request, output_dir=None)
                pred_frames = list(result.pred_frames)
                combined_pred_frames.extend(pred_frames[:keep_count])
                combined_scores.extend(
                    [float(score) for score in np.asarray(result.pred_scores).reshape(-1)[:keep_count]]
                )
                if args.save_chunks:
                    _save_image_sequence(str(chunk_dir / "pred_future_frames"), pred_frames)
                    chunk_payload["pred_future_frames_dir"] = str(chunk_dir / "pred_future_frames")
                    write_json(
                        str(chunk_dir / "chunk_result.json"),
                        {
                            "task_instruction": args.task,
                            "reward_threshold": float(result.reward_threshold),
                            "success_any": result.success_any,
                            "pred_scores": [
                                float(score) for score in np.asarray(result.pred_scores).reshape(-1)
                            ],
                            "current_frame": int(current_frame),
                            "input_frame_indices": [int(index) for index in input_indices],
                            "history_frame_indices": [
                                int(index) for index in input_indices[:-1]
                            ],
                            "future_frame_indices": [int(index) for index in future_indices],
                            "kept_future_frame_indices": [
                                int(index) for index in future_indices[:keep_count]
                            ],
                            "action_indices": [int(index) for index in action_indices],
                        },
                    )
                    if save_videos:
                        save_video(
                            str(chunk_dir / "pred_future.mp4"),
                            pred_frames,
                            fps=int(args.fps),
                        )
                chunk_payload["success_any"] = result.success_any

            if args.save_chunks:
                write_json(str(chunk_dir / "chunk_manifest.json"), chunk_payload)

            chunks.append(chunk_payload)
            print(
                f"[hdf5-wm-slide] chunk={chunk_idx:04d} current={current_frame} "
                f"future={future_indices} keep={keep_count}"
            )

    if save_videos:
        save_video(
            str(output_dir / "gt_future_sliding.mp4"),
            combined_gt_frames,
            fps=int(args.fps),
        )
        if combined_pred_frames:
            save_video(
                str(output_dir / "pred_future_sliding.mp4"),
                combined_pred_frames,
                fps=int(args.fps),
            )

    if args.export_frames:
        _save_image_sequence(str(output_dir / "gt_future_sliding_frames"), combined_gt_frames)
        if combined_pred_frames:
            _save_image_sequence(str(output_dir / "pred_future_sliding_frames"), combined_pred_frames)

    manifest = {
        "hdf5": str(hdf5_path),
        "config": str(args.config),
        "task_instruction": args.task,
        "image_key": image_key,
        "action_key": action_key,
        "action_slice": args.action_slice,
        "start_frame": int(start_frame),
        "end_frame_exclusive": int(end_frame),
        "chunk_step": int(chunk_step),
        "num_history": int(args.num_history),
        "num_future": int(args.num_future),
        "frame_stride": int(args.frame_stride),
        "action_stride": int(action_stride),
        "fps": int(args.fps),
        "num_inference_steps": args.num_inference_steps,
        "extract_only": bool(args.extract_only),
        "save_videos": bool(save_videos),
        "pred_frame_count": len(combined_pred_frames),
        "gt_frame_count": len(combined_gt_frames),
        "chunks": chunks,
    }
    write_json(str(output_dir / "sliding_manifest.json"), manifest)

    if combined_scores:
        reward_payload = {
            "scores": combined_scores,
            "success_any": bool(
                config is not None and np.any(np.asarray(combined_scores) >= config.reward_threshold)
            ),
            "reward_threshold": None if config is None else float(config.reward_threshold),
        }
        write_json(str(output_dir / "rewards_sliding.json"), reward_payload)

    print(f"[hdf5-wm-slide] output_dir={output_dir}")
    print(f"[hdf5-wm-slide] chunks={len(chunks)}")
    print(f"[hdf5-wm-slide] gt_frames={len(combined_gt_frames)} pred_frames={len(combined_pred_frames)}")


if __name__ == "__main__":
    main()
