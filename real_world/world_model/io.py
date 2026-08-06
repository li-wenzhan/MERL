from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV (cv2) is required for video IO. Install opencv-python or "
            "use --input-dir with an image sequence for input."
        ) from exc
    return cv2


def read_video_frames(
    video_path: str,
    *,
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> List[np.ndarray]:
    path = Path(video_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"missing input video: {path}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    frames: List[np.ndarray] = []
    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            frame_idx += 1
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"no frames were decoded from {path}")
    return frames


def read_image_sequence(image_dir: str, *, max_frames: Optional[int] = None) -> List[np.ndarray]:
    root = Path(image_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"missing input image directory: {root}")
    image_paths = sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise RuntimeError(f"no image files found in {root}")

    frames: List[np.ndarray] = []
    for path in image_paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
    return frames


def load_actions(path: str) -> np.ndarray:
    action_path = Path(path).expanduser()
    if not action_path.is_file():
        raise FileNotFoundError(f"missing action file: {action_path}")

    suffix = action_path.suffix.lower()
    if suffix == ".npy":
        actions = np.load(action_path)
    elif suffix == ".npz":
        payload = np.load(action_path)
        key = "actions" if "actions" in payload else payload.files[0]
        actions = payload[key]
    elif suffix in {".json", ".jsonl"}:
        with open(action_path, "r", encoding="utf-8") as file_obj:
            if suffix == ".jsonl":
                actions = [json.loads(line)["action"] for line in file_obj if line.strip()]
            else:
                payload = json.load(file_obj)
                actions = payload["actions"] if isinstance(payload, dict) else payload
        actions = np.asarray(actions, dtype=np.float32)
    else:
        raise ValueError(f"unsupported action file suffix: {suffix}")

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [T, action_dim], got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contain NaN or Inf")
    return actions


def save_video(path: str, frames: np.ndarray | Iterable[np.ndarray], *, fps: int) -> str:
    path_obj = Path(path).expanduser()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    frame_list = list(frames) if not isinstance(frames, np.ndarray) else list(frames)
    if not frame_list:
        raise ValueError("cannot save empty video")

    cv2 = _import_cv2()
    first = np.asarray(frame_list[0])
    height, width = int(first.shape[0]), int(first.shape[1])
    writer = cv2.VideoWriter(
        str(path_obj),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to create video writer: {path_obj}")
    try:
        for frame in frame_list:
            frame = np.asarray(frame)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return str(path_obj)


def write_json(path: str, payload: Dict) -> str:
    path_obj = Path(path).expanduser()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
    return str(path_obj)
