from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EpisodeRef:
    path: str
    group: str
    frame_count: int
    action_count: int


@dataclass(frozen=True)
class WindowRef:
    episode_idx: int
    current_frame: int


def _iter_hdf5_files(root: str | Path) -> list[Path]:
    path = Path(root).expanduser()
    if path.is_file():
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"expected .h5/.hdf5 file, got {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"missing HDF5 root: {path}")
    files = sorted(
        item
        for suffix in ("*.h5", "*.hdf5")
        for item in path.rglob(suffix)
        if item.is_file()
    )
    if not files:
        raise FileNotFoundError(f"no .h5/.hdf5 files found under {path}")
    return files


def _dataset_exists(group: h5py.Group, key: str) -> bool:
    normalized = key.strip("/")
    return normalized in group and isinstance(group[normalized], h5py.Dataset)


def _group_has_episode(group: h5py.Group, image_key: str, action_key: str) -> bool:
    return _dataset_exists(group, image_key) and _dataset_exists(group, action_key)


def discover_episodes(
    root: str | Path,
    *,
    image_key: str,
    action_key: str,
) -> list[EpisodeRef]:
    episodes: list[EpisodeRef] = []
    for path in _iter_hdf5_files(root):
        with h5py.File(path, "r") as h5_file:
            if _group_has_episode(h5_file, image_key, action_key):
                image_ds = h5_file[image_key.strip("/")]
                action_ds = h5_file[action_key.strip("/")]
                episodes.append(
                    EpisodeRef(
                        path=str(path),
                        group="",
                        frame_count=int(image_ds.shape[0]),
                        action_count=int(action_ds.shape[0]),
                    )
                )

            def visit(name: str, obj: Any) -> None:
                if not isinstance(obj, h5py.Group):
                    return
                if _group_has_episode(obj, image_key, action_key):
                    image_ds = obj[image_key.strip("/")]
                    action_ds = obj[action_key.strip("/")]
                    episodes.append(
                        EpisodeRef(
                            path=str(path),
                            group=name,
                            frame_count=int(image_ds.shape[0]),
                            action_count=int(action_ds.shape[0]),
                        )
                    )

            h5_file.visititems(visit)

    unique: dict[tuple[str, str], EpisodeRef] = {}
    for episode in episodes:
        unique[(episode.path, episode.group)] = episode
    result = sorted(unique.values(), key=lambda item: (item.path, item.group))
    if not result:
        raise RuntimeError(
            "no valid real-world episodes found. Check --image-key and --action-key."
        )
    return result


def _decode_frame(value: Any) -> np.ndarray:
    if isinstance(value, (bytes, bytearray)):
        with Image.open(io.BytesIO(value)) as image:
            return np.asarray(image.convert("RGB"))

    array = np.asarray(value)
    if array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            frame = array[..., :3]
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            return np.clip(frame, 0, 255).astype(np.uint8)
        if array.shape[0] in (1, 3, 4):
            frame = np.moveaxis(array[:3], 0, -1)
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            return np.clip(frame, 0, 255).astype(np.uint8)
    if array.ndim == 2 and array.dtype == np.uint8:
        if min(array.shape) > 8:
            return np.repeat(array[..., None], 3, axis=-1)
        with Image.open(io.BytesIO(array.reshape(-1).tobytes())) as image:
            return np.asarray(image.convert("RGB"))
    if array.ndim == 1 and array.dtype == np.uint8:
        with Image.open(io.BytesIO(array.tobytes())) as image:
            return np.asarray(image.convert("RGB"))
    raise ValueError(f"unsupported image frame shape={array.shape} dtype={array.dtype}")


def _select_action_columns(actions: np.ndarray, action_slice: str) -> np.ndarray:
    spec = str(action_slice).strip().lower()
    if spec == "right":
        if actions.shape[1] == 7:
            return actions
        if actions.shape[1] < 14:
            raise ValueError(f"right arm action slice requires 14D or 7D actions, got {actions.shape}")
        return actions[:, 7:14]
    if spec == "left":
        if actions.shape[1] < 7:
            raise ValueError(f"left arm action slice requires at least 7D actions, got {actions.shape}")
        return actions[:, :7]
    if spec == "all":
        return actions
    if ":" in spec:
        start_text, end_text = spec.split(":", 1)
        start = int(start_text) if start_text else None
        end = int(end_text) if end_text else None
        selected = actions[:, slice(start, end)]
        if selected.shape[1] <= 0:
            raise ValueError(f"empty action slice {action_slice!r} for shape {actions.shape}")
        return selected
    raise ValueError("--action-slice must be right, left, all, or start:end")


class RealWorldHDF5WindowDataset(Dataset):
    """Ctrl-World fine-tuning windows from real-robot HDF5/H5 episodes."""

    def __init__(
        self,
        root: str | Path,
        *,
        mode: str,
        image_key: str = "observations/images/cam_high",
        action_key: str = "action",
        action_slice: str = "right",
        instruction: str = "Put the red block into the box.",
        num_history: int = 8,
        num_future: int = 8,
        frame_stride: int = 1,
        action_stride: Optional[int] = None,
        sample_stride: int = 1,
        image_size: tuple[int, int] = (192, 320),
        val_ratio: float = 0.1,
        seed: int = 1024,
        action_scale: float = 1.0,
        action_offset: float = 0.0,
        max_windows: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mode not in {"train", "val"}:
            raise ValueError("mode must be train or val")
        self.root = str(root)
        self.mode = mode
        self.image_key = image_key.strip("/")
        self.action_key = action_key.strip("/")
        self.action_slice = action_slice
        self.instruction = instruction
        self.num_history = int(num_history)
        self.num_future = int(num_future)
        self.frame_stride = int(frame_stride)
        self.action_stride = int(action_stride or frame_stride)
        self.sample_stride = int(sample_stride)
        self.image_size = tuple(int(v) for v in image_size)
        self.action_scale = float(action_scale)
        self.action_offset = float(action_offset)

        if self.num_history <= 0 or self.num_future <= 0:
            raise ValueError("num_history and num_future must be positive")
        if self.frame_stride <= 0 or self.action_stride <= 0 or self.sample_stride <= 0:
            raise ValueError("frame_stride, action_stride, and sample_stride must be positive")
        if not (0.0 <= float(val_ratio) < 1.0):
            raise ValueError("val_ratio must be in [0, 1)")

        self.episodes = discover_episodes(
            self.root,
            image_key=self.image_key,
            action_key=self.action_key,
        )
        windows = self._build_windows()
        eligible = sorted({window.episode_idx for window in windows})
        if not eligible:
            raise RuntimeError("no trainable windows built from real-world HDF5 data")
        rng = random.Random(int(seed))
        rng.shuffle(eligible)
        if val_ratio > 0:
            if len(eligible) < 2:
                raise ValueError("episode-disjoint validation requires at least two eligible episodes; use --val-ratio 0 to disable it")
            val_count = min(len(eligible) - 1, max(1, int(round(len(eligible) * float(val_ratio)))))
        else:
            if mode == "val":
                raise ValueError("validation is disabled when val_ratio is zero")
            val_count = 0
        val_episodes = set(eligible[:val_count])
        # Overlapping windows must never cross the train/validation boundary.
        selected = [window for window in windows
                    if (window.episode_idx in val_episodes) == (mode == "val")]
        rng.shuffle(selected)
        if max_windows is not None and int(max_windows) > 0:
            selected = selected[: int(max_windows)]
        self.windows = selected
        if not self.windows:
            raise RuntimeError("no trainable windows built from real-world HDF5 data")

        print(
            f"[real-hdf5-dataset] mode={mode} episodes={len(self.episodes)} "
            f"split_episodes={len({w.episode_idx for w in self.windows})} "
            f"windows={len(self.windows)} total_windows={len(windows)}"
        )

    def _build_windows(self) -> list[WindowRef]:
        windows: list[WindowRef] = []
        min_current = max(
            self.num_history * self.frame_stride,
            self.num_history * self.action_stride,
        )
        future_span = (self.num_future - 1) * max(self.frame_stride, self.action_stride)
        for episode_idx, episode in enumerate(self.episodes):
            length = min(episode.frame_count, episode.action_count)
            max_current = length - 1 - future_span
            if max_current < min_current:
                continue
            for current in range(min_current, max_current + 1, self.sample_stride):
                windows.append(WindowRef(episode_idx=episode_idx, current_frame=current))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def _open_episode(self, episode: EpisodeRef) -> tuple[h5py.File, h5py.Group]:
        h5_file = h5py.File(episode.path, "r")
        group = h5_file[episode.group] if episode.group else h5_file
        return h5_file, group

    def _frame_indices(self, current_frame: int) -> list[int]:
        history = list(
            range(
                current_frame - self.num_history * self.frame_stride,
                current_frame,
                self.frame_stride,
            )
        )
        future = list(
            range(
                current_frame,
                current_frame + self.num_future * self.frame_stride,
                self.frame_stride,
            )
        )
        return history + future

    def _action_indices(self, current_frame: int) -> list[int]:
        history = list(
            range(
                current_frame - self.num_history * self.action_stride,
                current_frame,
                self.action_stride,
            )
        )
        future = list(
            range(
                current_frame,
                current_frame + self.num_future * self.action_stride,
                self.action_stride,
            )
        )
        return history + future

    def _preprocess_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        array = np.stack(frames, axis=0)
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
        tensor = tensor / 255.0 * 2.0 - 1.0
        if tuple(tensor.shape[-2:]) != self.image_size:
            tensor = F.interpolate(
                tensor,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        return tensor.contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        episode = self.episodes[window.episode_idx]
        frame_indices = self._frame_indices(window.current_frame)
        action_indices = self._action_indices(window.current_frame)

        h5_file, group = self._open_episode(episode)
        try:
            image_ds = group[self.image_key]
            action_ds = group[self.action_key]
            frames = [_decode_frame(image_ds[int(frame_idx)]) for frame_idx in frame_indices]
            raw_actions = np.asarray(
                [action_ds[int(action_idx)] for action_idx in action_indices],
                dtype=np.float32,
            )
        finally:
            h5_file.close()

        if raw_actions.ndim != 2:
            raw_actions = raw_actions.reshape(raw_actions.shape[0], -1)
        actions = _select_action_columns(raw_actions, self.action_slice)
        actions = (actions + self.action_offset) * self.action_scale
        if not np.all(np.isfinite(actions)):
            raise ValueError(f"actions contain NaN/Inf in {episode.path}:{episode.group}")

        rewards = np.zeros((self.num_history + self.num_future,), dtype=np.int64)
        rewards[-1] = 1

        return {
            "text": self.instruction,
            "img": self._preprocess_frames(frames),
            "action": torch.from_numpy(actions.astype(np.float32, copy=False)),
            "reward": torch.from_numpy(rewards),
        }
