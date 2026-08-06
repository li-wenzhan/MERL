from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def rgb_frames_to_tensor(
    frames: list[np.ndarray],
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    if not frames:
        raise ValueError("frames must not be empty")
    resized = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"each frame must be RGB HWC, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[:2] != (height, width):
            frame = np.asarray(
                Image.fromarray(frame).resize((width, height), Image.BILINEAR)
            )
        resized.append(frame)

    array = np.stack(resized, axis=0)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
    tensor = tensor / 255.0 * 2.0 - 1.0
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.contiguous()


def split_history_current(
    frames: list[np.ndarray],
    *,
    num_history: int,
) -> Tuple[list[np.ndarray], np.ndarray]:
    if num_history <= 0:
        raise ValueError(f"num_history must be positive, got {num_history}")
    if not frames:
        raise ValueError("frames must not be empty")

    current = frames[-1]
    history = list(frames[:-1])
    if not history:
        history = [current]

    if len(history) >= num_history:
        history = history[-num_history:]
    else:
        pad_count = num_history - len(history)
        history = [history[0]] * pad_count + history
    return history, current


def normalize_actions(
    actions: np.ndarray,
    *,
    action_dim: int,
    action_input_range: str,
    binarize_gripper: bool,
    invert_gripper: bool,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [T, action_dim], got {actions.shape}")
    if actions.shape[1] != action_dim:
        raise ValueError(f"actions action_dim mismatch: expected {action_dim}, got {actions.shape[1]}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contain NaN or Inf")

    mode = str(action_input_range or "model").lower()
    normalized = actions.copy()
    if mode in {"model", "minus_one_one", "-1_1"}:
        pass
    elif mode in {"zero_one", "0_1"}:
        normalized = np.clip(2.0 * normalized - 1.0, -1.0, 1.0)
    else:
        raise ValueError(
            "action_input_range must be 'model' for [-1, 1] actions or 'zero_one' for [0, 1] actions"
        )

    if binarize_gripper:
        normalized[:, -1] = np.where(normalized[:, -1] >= 0.0, 1.0, -1.0)
    if invert_gripper:
        normalized[:, -1] *= -1.0
    return normalized.astype(np.float32, copy=False)


def split_actions(
    actions: np.ndarray,
    *,
    num_history: int,
    num_future: int,
    action_dim: int,
    history_action_count: Optional[int],
    zero_history_actions_if_missing: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
    if actions.ndim != 2 or actions.shape[1] != action_dim:
        raise ValueError(f"actions must have shape [T, {action_dim}], got {actions.shape}")
    if num_future <= 0:
        raise ValueError(f"num_future must be positive, got {num_future}")

    if history_action_count is not None:
        history_action_count = max(0, int(history_action_count))
        required = history_action_count + num_future
        if actions.shape[0] < required:
            raise ValueError(
                f"actions length {actions.shape[0]} < history_action_count + num_future ({required})"
            )
        history = actions[:history_action_count]
        future = actions[history_action_count : history_action_count + num_future]
        source = "explicit_split"
    elif actions.shape[0] >= num_history + num_future:
        history = actions[-(num_history + num_future) : -num_future]
        future = actions[-num_future:]
        source = "auto_history_and_future"
    elif actions.shape[0] >= num_future:
        future = actions[-num_future:]
        if not zero_history_actions_if_missing:
            raise ValueError("history actions are missing and zero_history_actions_if_missing=False")
        history = np.zeros((num_history, action_dim), dtype=np.float32)
        source = "zero_history_actions"
    else:
        raise ValueError(f"actions length {actions.shape[0]} is shorter than num_future={num_future}")

    if history.shape[0] < num_history:
        pad = np.zeros((num_history - history.shape[0], action_dim), dtype=np.float32)
        history = np.concatenate([pad, history], axis=0)
    elif history.shape[0] > num_history:
        history = history[-num_history:]
    return history.astype(np.float32), future.astype(np.float32), source


def uint8_from_float01(frames: np.ndarray) -> np.ndarray:
    return (np.clip(frames, 0.0, 1.0) * 255.0).astype(np.uint8)
