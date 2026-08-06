from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RealWorldWMConfig:
    wm_config_path: str = "configs/wm_online_config.py"
    svd_model_path: Optional[str] = None
    clip_model_path: Optional[str] = None
    ckpt_path: Optional[str] = None
    load_from_ckpt: bool = True
    device: str = "cuda:0"
    dtype: str = "torch.bfloat16"

    width: int = 320
    height: int = 192
    num_history: int = 8
    num_future_frames: int = 8
    num_inference_steps: int = 30
    decode_chunk_size: int = 8
    guidance_scale: float = 2.0
    fps: int = 4
    motion_bucket_id: int = 127
    frame_level_cond: bool = True
    his_cond_zero: bool = False
    reward_threshold: float = 0.5

    action_dim: int = 7
    action_input_range: str = "model"
    binarize_gripper: bool = True
    invert_gripper: bool = False
    zero_history_actions_if_missing: bool = True

    output_fps: int = 4
    save_input_history: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RealWorldWMConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in payload.items() if key in known}
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RealWorldWMRequest:
    frames: List[np.ndarray]
    future_actions: np.ndarray
    task_instruction: str
    history_actions: Optional[np.ndarray] = None
    camera_name: str = "main"
    fps: int = 4
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealWorldWMResult:
    pred_frames: np.ndarray
    pred_scores: np.ndarray
    task_instruction: str
    reward_threshold: float
    pred_video_path: Optional[str] = None
    input_history_video_path: Optional[str] = None
    rewards_path: Optional[str] = None
    result_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_any(self) -> bool:
        return bool(np.any(self.pred_scores >= self.reward_threshold))

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "task_instruction": self.task_instruction,
            "reward_threshold": float(self.reward_threshold),
            "success_any": self.success_any,
            "pred_scores": [float(x) for x in self.pred_scores.reshape(-1)],
            "pred_video_path": self.pred_video_path,
            "input_history_video_path": self.input_history_video_path,
            "rewards_path": self.rewards_path,
            "result_path": self.result_path,
            "metadata": self.metadata,
        }


def resolve_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())

