from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import h5py
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_world.world_model.io import save_video


DEFAULT_IMAGE_CANDIDATES = (
    "observations/images/cam_high",
    "observations/images/right_front",
    "observations/images/front",
    "observations/images/camera",
    "images/cam_high",
    "images/front",
)

DEFAULT_ACTION_CANDIDATES = (
    "action",
    "actions",
    "master_action",
    "observations/action",
    "observations/actions",
)


def _iter_datasets(group: h5py.Group) -> Iterable[tuple[str, h5py.Dataset]]:
    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))

    datasets: list[tuple[str, h5py.Dataset]] = []
    group.visititems(visit)
    return datasets


def _dataset_exists(h5_file: h5py.File, key: str) -> bool:
    normalized = key.strip("/")
    return normalized in h5_file and isinstance(h5_file[normalized], h5py.Dataset)


def _find_image_dataset(h5_file: h5py.File, image_key: Optional[str]) -> tuple[str, h5py.Dataset]:
    if image_key:
        if not _dataset_exists(h5_file, image_key):
            raise KeyError(f"image dataset not found: {image_key}")
        normalized = image_key.strip("/")
        return normalized, h5_file[normalized]

    for key in DEFAULT_IMAGE_CANDIDATES:
        if _dataset_exists(h5_file, key):
            return key, h5_file[key]

    candidates: list[tuple[int, str, h5py.Dataset]] = []
    for name, dataset in _iter_datasets(h5_file):
        lower = name.lower()
        shape = dataset.shape
        if not shape:
            continue
        score = 0
        if any(token in lower for token in ("image", "camera", "cam", "rgb")):
            score += 10
        if "depth" in lower:
            score -= 8
        if len(shape) == 4 and (shape[-1] in (1, 3, 4) or shape[1] in (1, 3, 4)):
            score += 20
        elif len(shape) in (1, 2):
            score += 3
        if score > 0:
            candidates.append((score, name, dataset))

    if not candidates:
        raise KeyError("could not auto-detect an image dataset; pass --image-key")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, name, dataset = candidates[0]
    return name, dataset


def _find_action_dataset(h5_file: h5py.File, action_key: Optional[str]) -> tuple[str, h5py.Dataset]:
    if action_key:
        if not _dataset_exists(h5_file, action_key):
            raise KeyError(f"action dataset not found: {action_key}")
        normalized = action_key.strip("/")
        return normalized, h5_file[normalized]

    for key in DEFAULT_ACTION_CANDIDATES:
        if _dataset_exists(h5_file, key):
            dataset = h5_file[key]
            if dataset.ndim >= 2 and int(dataset.shape[-1]) >= 7:
                return key, dataset

    candidates: list[tuple[int, str, h5py.Dataset]] = []
    for name, dataset in _iter_datasets(h5_file):
        lower = name.lower()
        if "action" not in lower:
            continue
        if dataset.ndim < 2 or int(dataset.shape[-1]) < 7:
            continue
        score = 10
        if name.strip("/") == "action":
            score += 20
        if "master" in lower:
            score += 4
        candidates.append((score, name, dataset))

    if not candidates:
        raise KeyError("could not auto-detect an action dataset; pass --action-key")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, name, dataset = candidates[0]
    return name, dataset


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required to decode compressed HDF5 image frames.") from exc
    return cv2


def _decode_frame(value: Any) -> np.ndarray:
    if isinstance(value, (bytes, bytearray)):
        raw = np.frombuffer(value, dtype=np.uint8)
        return _decode_compressed(raw)

    array = np.asarray(value)
    if array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            frame = array[..., :3]
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
        elif array.shape[0] in (1, 3, 4):
            frame = np.moveaxis(array[:3], 0, -1)
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
        else:
            raise ValueError(f"unsupported image frame shape: {array.shape}")
        return np.clip(frame, 0, 255).astype(np.uint8)

    if array.ndim == 2 and array.dtype == np.uint8 and min(array.shape) > 8:
        frame = np.repeat(array[..., None], 3, axis=-1)
        return frame.astype(np.uint8, copy=False)

    if array.ndim in (1, 2) and array.dtype == np.uint8:
        return _decode_compressed(array.reshape(-1))

    raise ValueError(f"unsupported encoded image frame shape={array.shape} dtype={array.dtype}")


def _decode_compressed(raw: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    frame_bgr = cv2.imdecode(raw.astype(np.uint8, copy=False), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("failed to decode compressed image bytes from HDF5")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _select_action_columns(actions: np.ndarray, action_slice: str) -> np.ndarray:
    spec = action_slice.strip().lower()
    if spec == "right":
        if actions.shape[1] < 14:
            if actions.shape[1] == 7:
                return actions
            raise ValueError(f"right arm slice needs 14D or 7D actions, got {actions.shape}")
        return actions[:, 7:14]
    if spec == "left":
        if actions.shape[1] < 7:
            raise ValueError(f"left arm slice needs at least 7D actions, got {actions.shape}")
        return actions[:, :7]
    if spec == "all":
        return actions
    if ":" in spec:
        start_text, end_text = spec.split(":", 1)
        start = int(start_text) if start_text else None
        end = int(end_text) if end_text else None
        selected = actions[:, slice(start, end)]
        if selected.shape[1] <= 0:
            raise ValueError(f"empty action slice {action_slice!r} for action shape {actions.shape}")
        return selected
    raise ValueError("--action-slice must be right, left, all, or start:end")


def _as_indices(start: int, end: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if end <= start:
        raise ValueError(f"empty frame/action range: start={start}, end={end}")
    return list(range(start, end, stride))


def _auto_current_frame(
    *,
    frame_count: int,
    action_count: int,
    num_history: int,
    num_future: int,
    frame_stride: int,
    action_stride: int,
    include_history_actions: bool,
    current_frame_ratio: Optional[float],
) -> int:
    min_current, max_current = _valid_current_frame_range(
        frame_count=frame_count,
        action_count=action_count,
        num_history=num_history,
        num_future=num_future,
        frame_stride=frame_stride,
        action_stride=action_stride,
        include_history_actions=include_history_actions,
    )
    if current_frame_ratio is None:
        current_frame_ratio = 0.5
    ratio = float(current_frame_ratio)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"current_frame_ratio must be in [0, 1], got {ratio}")
    return int(round(min_current + ratio * (max_current - min_current)))


def _valid_current_frame_range(
    *,
    frame_count: int,
    action_count: int,
    num_history: int,
    num_future: int,
    frame_stride: int,
    action_stride: int,
    include_history_actions: bool,
) -> tuple[int, int]:
    min_current = num_history * frame_stride
    if include_history_actions:
        min_current = max(min_current, num_history * action_stride)

    max_current = min(frame_count - 1, action_count - 1 - (num_future - 1) * action_stride)
    if max_current < min_current:
        raise ValueError(
            "episode is too short for the requested prediction window: "
            f"min_current={min_current}, max_current={max_current}, "
            f"frame_count={frame_count}, action_count={action_count}"
        )
    return min_current, max_current


def _build_predict_indices(
    *,
    frame_count: int,
    action_count: int,
    current_frame: Optional[int],
    current_frame_ratio: Optional[float],
    num_history: int,
    num_future: int,
    frame_stride: int,
    action_stride: int,
    include_history_actions: bool,
) -> tuple[int, list[int], list[int], list[int]]:
    if num_history <= 0:
        raise ValueError(f"num_history must be positive, got {num_history}")
    if num_future <= 0:
        raise ValueError(f"num_future must be positive, got {num_future}")

    if current_frame is None:
        current_frame = _auto_current_frame(
            frame_count=frame_count,
            action_count=action_count,
            num_history=num_history,
            num_future=num_future,
            frame_stride=frame_stride,
            action_stride=action_stride,
            include_history_actions=include_history_actions,
            current_frame_ratio=current_frame_ratio,
        )

    current_frame = int(current_frame)
    frame_start = current_frame - num_history * frame_stride
    frame_indices = list(range(frame_start, current_frame + 1, frame_stride))
    if frame_start < 0 or current_frame >= frame_count:
        raise ValueError(
            f"current_frame={current_frame} is outside the valid frame history range for "
            f"frame_count={frame_count}, num_history={num_history}, frame_stride={frame_stride}"
        )

    future_indices = list(
        range(current_frame, current_frame + num_future * action_stride, action_stride)
    )
    if future_indices[-1] >= action_count:
        raise ValueError(
            f"future action window ends at {future_indices[-1]}, but action_count={action_count}"
        )

    if include_history_actions:
        history_start = current_frame - num_history * action_stride
        history_indices = list(range(history_start, current_frame, action_stride))
        if history_start < 0:
            raise ValueError(
                f"current_frame={current_frame} is outside the valid action history range"
            )
        action_indices = history_indices + future_indices
    else:
        action_indices = future_indices

    return current_frame, frame_indices, action_indices, future_indices


def _load_frames(dataset: h5py.Dataset, indices: list[int]) -> list[np.ndarray]:
    return [_decode_frame(dataset[int(index)]) for index in indices]


def _load_actions(dataset: h5py.Dataset, indices: list[int], action_slice: str) -> np.ndarray:
    raw = np.asarray([dataset[int(index)] for index in indices], dtype=np.float32)
    if raw.ndim != 2:
        raw = raw.reshape(raw.shape[0], -1)
    selected = _select_action_columns(raw, action_slice).astype(np.float32, copy=False)
    if not np.all(np.isfinite(selected)):
        raise ValueError("selected actions contain NaN or Inf")
    return selected


def _save_image_sequence(path: str, frames: list[np.ndarray]) -> str:
    root = Path(path).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        Image.fromarray(frame).save(root / f"frame_{idx:06d}.png")
    return str(root)


def _save_image(path: str, frame: np.ndarray) -> str:
    path_obj = Path(path).expanduser()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    Image.fromarray(frame).save(path_obj)
    return str(path_obj)


def _stats(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Extract real-world WM mp4/npy inputs from an HDF5 episode.")
    parser.add_argument("--hdf5", required=True, help="Input HDF5 episode.")
    parser.add_argument("--output-video", default=None, help="Optional output RGB mp4 path.")
    parser.add_argument("--output-image-dir", default=None, help="Optional output RGB image sequence directory.")
    parser.add_argument("--output-current-image", default=None, help="Optional current top-view RGB image path.")
    parser.add_argument("--output-gt-future-video", default=None, help="Optional HDF5 GT future RGB mp4 path.")
    parser.add_argument("--output-gt-future-image-dir", default=None, help="Optional HDF5 GT future image directory.")
    parser.add_argument("--output-actions", required=True, help="Output right-arm action .npy path.")
    parser.add_argument("--metadata-json", default=None, help="Optional extraction metadata path.")
    parser.add_argument("--image-key", default=None, help="HDF5 image dataset key.")
    parser.add_argument("--action-key", default=None, help="HDF5 action dataset key.")
    parser.add_argument("--action-slice", default="right", help="right, left, all, or start:end.")
    parser.add_argument("--window-mode", choices=("predict", "full"), default="predict")
    parser.add_argument("--current-frame", type=int, default=None)
    parser.add_argument("--current-frame-ratio", type=float, default=None)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--num-future", type=int, default=8)
    parser.add_argument("--include-history-actions", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--action-stride", type=int, default=None)
    parser.add_argument("--fps", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hdf5_path = Path(args.hdf5).expanduser()
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"missing HDF5 file: {hdf5_path}")

    action_stride = int(args.action_stride or args.frame_stride)
    if args.output_video is None and args.output_image_dir is None:
        raise ValueError("provide --output-video and/or --output-image-dir")
    output_video = Path(args.output_video).expanduser() if args.output_video is not None else None
    output_actions = Path(args.output_actions).expanduser()
    if args.output_video is not None:
        assert output_video is not None
        output_video.parent.mkdir(parents=True, exist_ok=True)
    output_actions.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as h5_file:
        image_key, image_dataset = _find_image_dataset(h5_file, args.image_key)
        action_key, action_dataset = _find_action_dataset(h5_file, args.action_key)
        frame_count = int(image_dataset.shape[0])
        action_count = int(action_dataset.shape[0])

        if args.window_mode == "predict":
            valid_current_frame_min, valid_current_frame_max = _valid_current_frame_range(
                frame_count=frame_count,
                action_count=action_count,
                num_history=args.num_history,
                num_future=args.num_future,
                frame_stride=int(args.frame_stride),
                action_stride=action_stride,
                include_history_actions=bool(args.include_history_actions),
            )
            current_frame, frame_indices, action_indices, gt_future_frame_indices = _build_predict_indices(
                frame_count=frame_count,
                action_count=action_count,
                current_frame=args.current_frame,
                current_frame_ratio=args.current_frame_ratio,
                num_history=args.num_history,
                num_future=args.num_future,
                frame_stride=int(args.frame_stride),
                action_stride=action_stride,
                include_history_actions=bool(args.include_history_actions),
            )
        else:
            start = max(0, int(args.start_frame))
            end = int(args.end_frame) if args.end_frame is not None else min(frame_count, action_count)
            if args.max_frames is not None:
                end = min(end, start + int(args.max_frames) * int(args.frame_stride))
            frame_indices = _as_indices(start, min(end, frame_count), int(args.frame_stride))
            action_indices = _as_indices(start, min(end, action_count), action_stride)
            current_frame = frame_indices[-1]
            gt_future_frame_indices = []
            valid_current_frame_min = None
            valid_current_frame_max = None

        frames = _load_frames(image_dataset, frame_indices)
        gt_future_frames = _load_frames(image_dataset, gt_future_frame_indices) if gt_future_frame_indices else []
        actions = _load_actions(action_dataset, action_indices, args.action_slice)

    output_video_text = None
    output_image_dir_text = None
    output_current_image_text = None
    output_gt_future_video_text = None
    output_gt_future_image_dir_text = None
    if args.output_video is not None:
        assert output_video is not None
        save_video(str(output_video), frames, fps=int(args.fps))
        output_video_text = str(output_video)
    if args.output_image_dir is not None:
        output_image_dir_text = _save_image_sequence(args.output_image_dir, frames)
    if args.output_current_image is not None:
        output_current_image_text = _save_image(args.output_current_image, frames[-1])
    if args.output_gt_future_video is not None and gt_future_frames:
        output_gt_future_video_text = save_video(
            args.output_gt_future_video,
            gt_future_frames,
            fps=int(args.fps),
        )
    if args.output_gt_future_image_dir is not None and gt_future_frames:
        output_gt_future_image_dir_text = _save_image_sequence(
            args.output_gt_future_image_dir,
            gt_future_frames,
        )
    np.save(output_actions, actions)

    warnings: list[str] = []
    if actions.shape[1] != 7:
        warnings.append(
            f"selected action_dim={actions.shape[1]}; real_world/configs/real_wm_infer.yaml defaults to action_dim=7"
        )
    if float(np.max(np.abs(actions))) > 1.05:
        warnings.append(
            "selected actions exceed [-1, 1]; verify real_wm_infer.yaml action_input_range/action normalization"
        )

    metadata = {
        "hdf5": str(hdf5_path),
        "image_key": image_key,
        "action_key": action_key,
        "action_slice": args.action_slice,
        "window_mode": args.window_mode,
        "current_frame": int(current_frame),
        "current_frame_ratio": args.current_frame_ratio,
        "valid_current_frame_min": valid_current_frame_min,
        "valid_current_frame_max": valid_current_frame_max,
        "frame_indices": [int(index) for index in frame_indices],
        "gt_future_frame_indices": [int(index) for index in gt_future_frame_indices],
        "action_indices": [int(index) for index in action_indices],
        "frame_count": len(frame_indices),
        "action_count": len(action_indices),
        "fps": int(args.fps),
        "frame_stride": int(args.frame_stride),
        "action_stride": int(action_stride),
        "include_history_actions": bool(args.include_history_actions),
        "output_video": output_video_text,
        "output_image_dir": output_image_dir_text,
        "output_current_image": output_current_image_text,
        "output_gt_future_video": output_gt_future_video_text,
        "output_gt_future_image_dir": output_gt_future_image_dir_text,
        "output_actions": str(output_actions),
        "action_stats": _stats(actions),
        "warnings": warnings,
    }

    metadata_path = Path(args.metadata_json).expanduser() if args.metadata_json else output_actions.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    print(f"[hdf5-wm] image_key={image_key} action_key={action_key}")
    print(
        f"[hdf5-wm] mode={args.window_mode} current_frame={current_frame} "
        f"frames={len(frame_indices)} actions={actions.shape}"
    )
    if valid_current_frame_min is not None and valid_current_frame_max is not None:
        print(
            f"[hdf5-wm] valid_current_frame_range="
            f"[{valid_current_frame_min}, {valid_current_frame_max}]"
        )
    print(f"[hdf5-wm] input_frame_indices={frame_indices}")
    if gt_future_frame_indices:
        print(f"[hdf5-wm] gt_future_frame_indices={gt_future_frame_indices}")
    print(f"[hdf5-wm] action_indices={action_indices}")
    if output_video_text is not None:
        print(f"[hdf5-wm] video={output_video_text}")
    if output_image_dir_text is not None:
        print(f"[hdf5-wm] image_dir={output_image_dir_text}")
    if output_current_image_text is not None:
        print(f"[hdf5-wm] current_image={output_current_image_text}")
    if output_gt_future_video_text is not None:
        print(f"[hdf5-wm] gt_future_video={output_gt_future_video_text}")
    if output_gt_future_image_dir_text is not None:
        print(f"[hdf5-wm] gt_future_image_dir={output_gt_future_image_dir_text}")
    print(f"[hdf5-wm] actions={output_actions}")
    print(f"[hdf5-wm] metadata={metadata_path}")
    for warning in warnings:
        print(f"[hdf5-wm][warning] {warning}")


if __name__ == "__main__":
    main()
