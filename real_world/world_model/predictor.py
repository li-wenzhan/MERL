from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .ctrl_world_adapter import CtrlWorldRealAdapter
from .io import save_video, write_json
from .preprocess import (
    normalize_actions,
    rgb_frames_to_tensor,
    split_actions,
    split_history_current,
)
from .schemas import RealWorldWMConfig, RealWorldWMRequest, RealWorldWMResult


class RealWorldWMPredictor:
    def __init__(self, config: RealWorldWMConfig):
        self.config = config
        self.adapter = CtrlWorldRealAdapter(config)

    @classmethod
    def from_config(cls, config: RealWorldWMConfig) -> "RealWorldWMPredictor":
        return cls(config)

    def predict(
        self,
        request: RealWorldWMRequest,
        *,
        output_dir: Optional[str] = None,
    ) -> RealWorldWMResult:
        history_frames, current_frame = split_history_current(
            request.frames,
            num_history=self.config.num_history,
        )
        history_tensor = rgb_frames_to_tensor(
            history_frames,
            height=self.config.height,
            width=self.config.width,
        )
        current_tensor = rgb_frames_to_tensor(
            [current_frame],
            height=self.config.height,
            width=self.config.width,
        )[0]

        future_actions = normalize_actions(
            request.future_actions,
            action_dim=self.config.action_dim,
            action_input_range=self.config.action_input_range,
            binarize_gripper=self.config.binarize_gripper,
            invert_gripper=self.config.invert_gripper,
        )
        if request.history_actions is not None:
            history_actions = normalize_actions(
                request.history_actions,
                action_dim=self.config.action_dim,
                action_input_range=self.config.action_input_range,
                binarize_gripper=self.config.binarize_gripper,
                invert_gripper=self.config.invert_gripper,
            )
            action_source = "request_history_actions"
            if history_actions.shape[0] < self.config.num_history:
                pad = np.zeros(
                    (self.config.num_history - history_actions.shape[0], self.config.action_dim),
                    dtype=np.float32,
                )
                history_actions = np.concatenate([pad, history_actions], axis=0)
            elif history_actions.shape[0] > self.config.num_history:
                history_actions = history_actions[-self.config.num_history :]
        else:
            history_actions, future_actions, action_source = split_actions(
                future_actions,
                num_history=self.config.num_history,
                num_future=self.config.num_future_frames,
                action_dim=self.config.action_dim,
                history_action_count=None,
                zero_history_actions_if_missing=self.config.zero_history_actions_if_missing,
            )

        if future_actions.shape[0] != self.config.num_future_frames:
            raise ValueError(
                f"future_actions length {future_actions.shape[0]} != num_future_frames={self.config.num_future_frames}"
            )

        pred_frames, pred_scores, adapter_metadata = self.adapter.predict(
            current_image=current_tensor,
            history_images=history_tensor,
            history_actions=history_actions,
            future_actions=future_actions,
            task_instruction=request.task_instruction,
        )

        result = RealWorldWMResult(
            pred_frames=pred_frames,
            pred_scores=pred_scores,
            task_instruction=request.task_instruction,
            reward_threshold=self.adapter.reward_threshold,
            metadata={
                **request.metadata,
                **adapter_metadata,
                "camera_name": request.camera_name,
                "input_fps": request.fps,
                "action_source": action_source,
                "history_frame_count": len(history_frames),
                "future_action_count": int(future_actions.shape[0]),
            },
        )

        if output_dir is not None:
            self.save_result(result, output_dir=output_dir, history_frames=history_frames)
        return result

    def save_result(
        self,
        result: RealWorldWMResult,
        *,
        output_dir: str,
        history_frames: Optional[list[np.ndarray]] = None,
    ) -> RealWorldWMResult:
        root = Path(output_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)

        result.pred_video_path = save_video(
            str(root / "pred_future.mp4"),
            result.pred_frames,
            fps=self.config.output_fps,
        )
        if history_frames is not None and self.config.save_input_history:
            result.input_history_video_path = save_video(
                str(root / "input_history.mp4"),
                history_frames,
                fps=self.config.output_fps,
            )

        reward_payload = {
            "reward_threshold": float(result.reward_threshold),
            "scores": [float(score) for score in result.pred_scores.reshape(-1)],
            "success_any": result.success_any,
        }
        result.rewards_path = write_json(str(root / "rewards.json"), reward_payload)
        result.result_path = write_json(str(root / "result.json"), result.to_json_dict())
        return result

