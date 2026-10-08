from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .preprocess import uint8_from_float01
from .schemas import RealWorldWMConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
CTRL_WORLD_ROOT = REPO_ROOT / "modules" / "ctrl_world"


def _ensure_ctrl_world_path() -> None:
    for path in (REPO_ROOT, CTRL_WORLD_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_wm_args(config_path: str):
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"missing world-model config: {path}")

    spec = importlib.util.spec_from_file_location("real_world_wm_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import world-model config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "wm_args"):
        raise AttributeError(f"{path} does not define wm_args")
    return module.wm_args()


def _apply_config_overrides(wm_args: Any, config: RealWorldWMConfig) -> Any:
    overrides: Dict[str, Any] = {
        "width": config.width,
        "height": config.height,
        "num_history": config.num_history,
        "num_frames": config.num_future_frames,
        "num_inference_steps": config.num_inference_steps,
        "decode_chunk_size": config.decode_chunk_size,
        "guidance_scale": config.guidance_scale,
        "fps": config.fps,
        "motion_bucket_id": config.motion_bucket_id,
        "frame_level_cond": config.frame_level_cond,
        "his_cond_zero": config.his_cond_zero,
        "reward_threshold": config.reward_threshold,
        "action_dim": config.action_dim,
        "load_from_ckpt": config.load_from_ckpt,
        "dtype": config.dtype,
    }
    if config.svd_model_path:
        overrides["svd_model_path"] = config.svd_model_path
    if config.clip_model_path:
        overrides["clip_model_path"] = config.clip_model_path
    if config.ckpt_path:
        overrides["ckpt_path"] = config.ckpt_path

    for key, value in overrides.items():
        if hasattr(wm_args, key):
            setattr(wm_args, key, value)
    if hasattr(wm_args, "__post_init__"):
        wm_args.__post_init__()
    return wm_args


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name or "").lower()
    if "bfloat16" in normalized or "bf16" in normalized:
        return torch.bfloat16
    if "float16" in normalized or "fp16" in normalized:
        return torch.float16
    return torch.float32


def _get_inner_module(module):
    return module.module if hasattr(module, "module") else module


def _encode_img_to_latent(model, img: torch.Tensor, device: torch.device) -> torch.Tensor:
    model = _get_inner_module(model)
    vae = _get_inner_module(model.vae)
    with torch.no_grad():
        latent = vae.encode(img.to(device)).latent_dist.mean
        latent = latent.mul_(vae.config.scaling_factor)
    return latent


class CtrlWorldRealAdapter:
    def __init__(self, config: RealWorldWMConfig):
        _ensure_ctrl_world_path()
        from modules.ctrl_world.model_loading import (
            load_trusted_state_dict,
            resolve_ctrl_world_ckpt_path,
        )
        from modules.ctrl_world.models.ctrl_world_new import CtrlWorld

        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_dtype(config.dtype)
        self.wm_args = _apply_config_overrides(_load_wm_args(config.wm_config_path), config)

        self.model = CtrlWorld(self.wm_args)
        ckpt_path = resolve_ctrl_world_ckpt_path(self.wm_args)
        if ckpt_path is not None:
            state_dict = load_trusted_state_dict(ckpt_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    @property
    def reward_threshold(self) -> float:
        return float(getattr(self.wm_args, "reward_threshold", self.config.reward_threshold))

    def _action_latent(
        self,
        actions: np.ndarray,
        task_instruction: str,
    ) -> torch.Tensor:
        wm = _get_inner_module(self.model)
        action_tensor = torch.from_numpy(actions).unsqueeze(0).to(
            device=self.device,
            dtype=self.dtype,
        )
        text = [task_instruction]
        try:
            return wm.action_encoder(
                action_tensor,
                text,
                wm.tokenizer,
                wm.text_encoder,
                self.wm_args.frame_level_cond,
            )
        except Exception:
            return wm.action_encoder(
                action_tensor,
                task_instruction,
                wm.tokenizer,
                wm.text_encoder,
                self.wm_args.frame_level_cond,
            )

    def predict(
        self,
        *,
        current_image: torch.Tensor,
        history_images: torch.Tensor,
        history_actions: np.ndarray,
        future_actions: np.ndarray,
        task_instruction: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        from modules.ctrl_world.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

        if current_image.dim() == 3:
            current_image = current_image.unsqueeze(0)
        if history_images.dim() == 4:
            history_images = history_images.unsqueeze(0)

        all_actions = np.concatenate([history_actions, future_actions], axis=0).astype(np.float32)
        num_future = int(future_actions.shape[0])

        with torch.no_grad():
            current_latent = _encode_img_to_latent(
                self.model,
                current_image.to(dtype=self.dtype),
                self.device,
            )
            history_flat = history_images.squeeze(0).to(dtype=self.dtype)
            history_latent = _encode_img_to_latent(self.model, history_flat, self.device).unsqueeze(0)
            action_latent = self._action_latent(all_actions, task_instruction)

            pred_frames, pred_latents = CtrlWorldDiffusionPipeline.__call__(
                _get_inner_module(self.model).pipeline,
                image=current_latent,
                text=action_latent,
                width=self.wm_args.width,
                height=self.wm_args.height,
                num_frames=num_future,
                history=history_latent,
                num_inference_steps=self.wm_args.num_inference_steps,
                decode_chunk_size=min(self.wm_args.decode_chunk_size, num_future),
                max_guidance_scale=self.wm_args.guidance_scale,
                fps=self.wm_args.fps,
                motion_bucket_id=self.wm_args.motion_bucket_id,
                mask=None,
                output_type="frame",
                return_dict=False,
                frame_level_cond=self.wm_args.frame_level_cond,
                his_cond_zero=self.wm_args.his_cond_zero,
            )

            pred_float = np.asarray(pred_frames[0], dtype=np.float32)
            pred_uint8 = uint8_from_float01(pred_float)
            pred_tensor = torch.from_numpy(pred_float).permute(0, 3, 1, 2).to(
                device=self.device,
                dtype=self.dtype,
            )
            future_action_latent = action_latent[:, -num_future:, :]
            reward_actions = future_action_latent.reshape(-1, future_action_latent.shape[-1])
            # Q(o_t, u_t) sees the anchor before the first command and the
            # preceding prediction for every following command. Training uses
            # the same [-1, 1] image range.
            proxy_images = torch.cat((current_image.to(device=self.device, dtype=self.dtype),
                                      pred_tensor[:-1] * 2 - 1), dim=0)
            pred_scores = _get_inner_module(self.model).reward_classifier.predict_score(
                proxy_images,
                reward_actions.to(device=self.device, dtype=self.dtype),
            )
            pred_scores_np = pred_scores.detach().to(torch.float32).cpu().numpy()

        metadata = {
            "pred_latent_shape": tuple(pred_latents.shape) if hasattr(pred_latents, "shape") else None,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "num_future_frames": num_future,
        }
        return pred_uint8, pred_scores_np, metadata

