# from diffusers import StableVideoDiffusionPipeline
import datetime
import json
import os
from typing import Optional, Tuple
import sys

script_path = os.path.abspath(__file__)
wm_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(wm_dir)

import einops
import mediapy
import numpy as np
import swanlab
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from decord import VideoReader, cpu
try:
    from modules.ctrl_world.model_loading import (
        prepare_local_hf_model_dir,
        rethrow_hf_loading_error,
    )
except ImportError:
    from model_loading import prepare_local_hf_model_dir, rethrow_hf_loading_error
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.reward_model import VisionActionClassifier
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from tqdm.auto import tqdm


# ---- utils: positional encoding ----
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def self_forcing_from_pred_and_gt(
    gt_latent: torch.Tensor,
    pred_latent: torch.Tensor,
    gradient_mask: Optional[torch.Tensor] = None,
    normalization: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, dict]:
    """
    DMD-style self-forcing loss using GT latent as implicit teacher.

    Args:
        gt_latent:   [B, F, C, H, W], ground-truth latent (from real env)
        pred_latent: [B, F, C, H, W], predicted latent (from world model)
        gradient_mask: optional bool mask, same shape, True = include
        normalization: whether to normalize grad per-sample
    Returns:
        loss: scalar tensor
        log_dict
    """
    assert gt_latent.shape == pred_latent.shape
    B = gt_latent.shape[0]

    # ------------------------------------------------------------
    # Step 1: define "desired gradient" (implicit score difference)
    # ------------------------------------------------------------
    grad = pred_latent - gt_latent  # [B, F, C, H, W]
    if normalization:
        # per-sample stabilizer (similar role to |p_real| in DMD)
        norm_factor = (
            gt_latent.abs().mean(dim=[1, 2, 3, 4], keepdim=True).clamp_min(eps)
        )
        grad = grad / norm_factor
    else:
        norm_factor = torch.ones(
            (B, 1, 1, 1, 1),
            device=pred_latent.device,
            dtype=pred_latent.dtype,
        )
    grad = torch.nan_to_num(grad)

    # ------------------------------------------------------------
    # Step 2: construct DMD-style target (CRITICAL)
    # ------------------------------------------------------------
    # This is what enforces: dL/d(pred_latent) = grad
    target = (pred_latent - grad).detach()

    # ------------------------------------------------------------
    # Step 3: compute loss
    # ------------------------------------------------------------
    if gradient_mask is not None:
        mask = gradient_mask.to(dtype=torch.bool)
        diff = (pred_latent - target)[mask]
        if diff.numel() == 0:
            loss = torch.zeros((), device=pred_latent.device, dtype=pred_latent.dtype)
        else:
            loss = 0.5 * diff.pow(2).mean()
    else:
        loss = 0.5 * (pred_latent - target).pow(2).mean()

    # ------------------------------------------------------------
    # logs
    # ------------------------------------------------------------
    log_dict = {
        "sf_grad_abs_mean": float(grad.abs().mean().detach().cpu()),
        "sf_norm_factor_mean": float(norm_factor.mean().detach().cpu()),
    }

    return loss, log_dict


# ---- modules ----
class Action_encoder2(nn.Module):
    def __init__(self, action_dim, action_num, hidden_size, text_cond=True):
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_size = hidden_size
        self.text_cond = text_cond

        input_dim = int(action_dim)
        self.action_encode = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
        )
        # kaiming initialization
        nn.init.kaiming_normal_(
            self.action_encode[0].weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.action_encode[2].weight, mode="fan_in", nonlinearity="relu"
        )

    def forward(
        self,
        action,
        texts=None,
        text_tokinizer=None,
        text_encoder=None,
        frame_level_cond=True,
    ):
        # action: (B, action_num, action_dim)
        B, T, D = action.shape
        if not frame_level_cond:
            action = einops.rearrange(action, "b t d -> b 1 (t d)")
        action = self.action_encode(action)

        if texts is not None and self.text_cond:
            # with 50% probability, add text condition
            with torch.no_grad():
                inputs = text_tokinizer(
                    texts, padding="max_length", return_tensors="pt", truncation=True
                ).to(text_encoder.device)
                outputs = text_encoder(**inputs)
                hidden_text = outputs.text_embeds  # (B, 512)
                hidden_text = einops.repeat(
                    hidden_text, "b c -> b 1 (n c)", n=2
                )  # (B, 1, 1024)

            action = action + hidden_text  # (B, T, hidden_size)
        return action  # (B, 1, hidden_size) or (B, T, hidden_size) if frame_level_cond


class CtrlWorld(nn.Module):
    def __init__(self, args):
        super(CtrlWorld, self).__init__()

        self.args = args

        svd_model_path, svd_pretrained_kwargs, _ = prepare_local_hf_model_dir(
            args.svd_model_path,
            component_name="Ctrl-World SVD backbone",
        )
        clip_model_path, clip_pretrained_kwargs, _ = prepare_local_hf_model_dir(
            args.clip_model_path,
            component_name="Ctrl-World CLIP text encoder",
        )

        # load from pretrained stable video diffusion
        try:
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                svd_model_path,
                **svd_pretrained_kwargs,
            )
        except Exception as exc:
            rethrow_hf_loading_error(
                exc,
                component_name="Ctrl-World SVD backbone",
                model_path=svd_model_path,
            )
        # repalce the unet to support frame_level pose condition
        print("replace the unet to support action condition and frame_level pose!")
        unet = UNetSpatioTemporalConditionModel()
        unet.load_state_dict(self.pipeline.unet.state_dict(), strict=False)
        self.pipeline.unet = unet
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.image_encoder = self.pipeline.image_encoder
        self.scheduler = self.pipeline.scheduler

        # freeze vae, image_encoder, enable unet gradient ckpt
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        self.unet.enable_gradient_checkpointing()

        # SVD is a img2video model, load a clip text encoder
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        try:
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
                clip_model_path,
                **clip_pretrained_kwargs,
            )
        except Exception as exc:
            rethrow_hf_loading_error(
                exc,
                component_name="Ctrl-World CLIP text encoder",
                model_path=clip_model_path,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            clip_model_path, use_fast=False
        )
        self.text_encoder.requires_grad_(False)

        # initialize an action projector
        self.action_encoder = Action_encoder2(
            action_dim=args.action_dim,
            action_num=int(args.num_history + args.num_frames),  # T
            hidden_size=1024,
            text_cond=args.text_cond,
        )

        self.reward_classifier = VisionActionClassifier(
            latent_dim=512,
            action_dim=1024,
            num_classes=2,
            num_action_tokens=4,
            num_heads=8,
            mlp_ratio=2.0,
            dropout=0.1,
            freeze_vision=True,
            in_channels=3,
            # A full simulator checkpoint contains the reward backbone too.
            pretrained_backbone=not bool(getattr(args, "load_from_ckpt", False)),
        )

    def encode_img_to_latent(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, T, 3, h, w)
        return: (B, T, 4, 32, 32)
        """
        B, T, C, H, W = img.shape
        img = img.view(B * T, C, H, W)
        latents: torch.Tensor = (
            self.vae.encode(img)
            .latent_dist.sample()
            .mul_(self.vae.config.scaling_factor)
        )  # [B * T, 4, 24, 40]
        latents = latents.view(B, T, *latents.shape[1:])
        return latents

    def compute_self_forcing_loss(
        self,
        original_latent: torch.Tensor,
        predicted_latent: torch.Tensor,
        gradient_mask: Optional[torch.Tensor] = None,
        use_normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute self-forcing loss between ground truth latent and predicted latent.
        - original_latent: ground truth latents [B, F, C, H, W]
        - predicted_latent: model predicted latents (same shape)
        Returns loss, log_dict.
        """
        return self_forcing_from_pred_and_gt(
            gt_latent=original_latent,
            pred_latent=predicted_latent,
            gradient_mask=gradient_mask,
            normalization=use_normalization,
        )

    def compute_time_weights(
        self,
        num_future,
        scheme="linear",
        alpha=2.0,
        beta=3.0,
        normalize=False,
        device="cpu",
    ):
        """
        返回形状 (num_future,) 的权重张量，归一化（可选）使得 mean=1
        """
        if num_future == 0:
            return torch.empty(0, device=device)
        t = torch.arange(num_future, dtype=torch.float32, device=device)
        if scheme == "linear":
            w = 1.0 + alpha * (t / (num_future - 1 + 1e-8))
        elif scheme == "exp":
            w = torch.exp(beta * (t / (num_future - 1 + 1e-8)))
        elif scheme == "power":
            w = (1.0 + t) ** alpha
        else:
            w = torch.ones_like(t)
        if normalize:
            w = w * (num_future / w.sum())  # 归一化到 mean = 1
        return w

    def forward(self, batch):
        dtype = self.unet.dtype
        device = self.unet.device
        P_mean = 0.7
        P_std = 1.6
        noise_aug_strength = 0.0

        # prepare inputs
        if self.args.is_img_pregenerated and batch.get("latent", None) is not None:
            img = batch["img"]  # (B, T, 3, h, w)
            latents = batch["latent"]  # (B, T, 4, 72, 40)
        elif not self.args.is_img_pregenerated and batch.get("img", None) is not None:
            img = batch["img"]  # (B, T, 3, h, w)
            latents = self.encode_img_to_latent(img)  # [B, T, 4, 24, 40]
        else:
            raise NotImplementedError("no obs img or latent provided!")
        texts = batch["text"]  # (B)
        action = batch["action"]  # (B, f, 7)
        reward = batch["reward"].to(dtype=torch.int64)  # (B, f)

        num_history = self.args.num_history
        latents = latents.to(device)  # [B, num_history + num_future, 4, 32, 32]

        # current img as condition image to stack at channel wise, add random noise to current image, noise strength 0.0~0.2
        current_img = latents[:, num_history : (num_history + 1)]  # (B, 1, 4, 32, 32)
        bsz, num_frames = latents.shape[:2]  # (B, T)
        current_img = current_img[:, 0]  # (B, 4, 32, 32)

        # blur current image
        sigma = torch.rand([bsz, 1, 1, 1], device=device) * 0.2
        c_in = 1 / (sigma**2 + 1) ** 0.5
        current_img = c_in * (current_img + torch.randn_like(current_img) * sigma)

        condition_latent = einops.repeat(
            current_img, "b c h w -> b f c h w", f=num_frames
        )  # (8, T, 12, 32, 32)
        if self.args.his_cond_zero:
            condition_latent[:, :num_history] = (
                0.0  # (B, num_history+num_frames, 4, 32, 32)
            )

        # action condition
        action = action.to(device=device, dtype=dtype)
        action_hidden = self.action_encoder(
            action,
            texts,
            self.tokenizer,
            self.text_encoder,
            frame_level_cond=self.args.frame_level_cond,
        )  # (B, T, 1024)

        #! calculate reward
        img_ = einops.rearrange(img, "b t c h w -> (b t) c h w")
        action_hidden_ = einops.rearrange(action_hidden, "b t c -> (b t) c")
        pred_rew_logit, pred_rew_prob = self.reward_classifier(
            img_, action_hidden_
        )  # [BT, 2]
        reward_ = einops.rearrange(reward, "b t -> (b t)")
        loss_reward = F.cross_entropy(pred_rew_logit, reward_)

        # for classifier-free guidance, with 5% probability, set action_hidden to 0
        uncond_hidden_states = torch.zeros_like(action_hidden)
        text_mask = (
            (torch.rand(action_hidden.shape[0], device=device) > 0.05)
            .unsqueeze(1)
            .unsqueeze(2)
        )
        action_hidden = action_hidden * text_mask + uncond_hidden_states * (~text_mask)

        # diffusion forward process on future latent
        rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        c_skip = 1 / (sigma**2 + 1)
        c_out = -sigma / (sigma**2 + 1) ** 0.5
        c_in = 1 / (sigma**2 + 1) ** 0.5
        c_noise = (sigma.log() / 4).reshape([bsz])
        loss_weight = (sigma**2 + 1) / sigma**2
        noisy_latents = latents + torch.randn_like(latents) * sigma

        # add 0~0.3 noise to history, history as condition
        sigma_h = torch.randn([bsz, num_history, 1, 1, 1], device=device) * 0.3
        history = latents[:, :num_history]  # (B, num_history, 4, 32, 32)
        noisy_history = (
            1
            / (sigma_h**2 + 1) ** 0.5
            * (history + sigma_h * torch.randn_like(history))
        )  # (B, num_history, 4, 32, 32)
        input_latents = torch.cat(
            [noisy_history, c_in * noisy_latents[:, num_history:]], dim=1
        )  # (B, num_history+num_frames, 4, 32, 32)

        # svd stack a img at channel wise
        input_latents = torch.cat(
            [input_latents, condition_latent / self.vae.config.scaling_factor], dim=2
        )  # 输入特征 = 原始特征加噪结果 || 编码后的条件特征
        motion_bucket_id = self.args.motion_bucket_id
        fps = self.args.fps
        added_time_ids = self.pipeline._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            action_hidden.dtype,
            bsz,
            1,
            False,
        )
        added_time_ids = added_time_ids.to(device)

        # forward unet
        model_pred = self.unet(
            input_latents,
            c_noise,
            encoder_hidden_states=action_hidden,
            added_time_ids=added_time_ids,
            frame_level_cond=self.args.frame_level_cond,
        ).sample  # 预测噪声
        predict_x0 = c_out * model_pred + c_skip * noisy_latents

        # only calculate loss on future frames
        # loss_noise = (
        #     (predict_x0[:, num_history:] - latents[:, num_history:]) ** 2 * loss_weight
        # ).mean()
        #! to solve seam jitter
        # 假设 predict_x0 和 latents 形状 (B, T_total, C, H, W)
        num_future = predict_x0.shape[1] - num_history
        w = self.compute_time_weights(
            num_future, scheme="exp", beta=3.0, device=predict_x0.device
        )  # (T_f,)
        time_weight = w.view(1, -1, 1, 1, 1)  # expand 到帧维（B, T_f, 1, 1, 1）
        loss_noise = (predict_x0[:, num_history:] - latents[:, num_history:]) ** 2
        loss_noise = loss_noise * loss_weight
        loss_noise = loss_noise * time_weight
        loss_noise = loss_noise.mean()

        loss_dict = {
            "loss_noise": loss_noise,
            "loss_reward": loss_reward,
        }

        # -------------------
        # Add self-forcing loss (config keys below)
        # -------------------
        # config keys (add them to args or config world_model):
        #   args.self_forcing_weight (float, default 0.0 -> disable)
        #   args.self_forcing_normalize (bool, default True)
        sf_weight = getattr(self.args, "self_forcing_weight", 1.0)
        sf_normalize = getattr(self.args, "self_forcing_normalize", True)
        if sf_weight is not None and sf_weight > 0.0:
            # only compute on future frames (same as loss_noise)
            pred_future = predict_x0[
                :, num_history:
            ]  # predicted clean for future frames
            gt_future = latents[:, num_history:]  # GT latents for future frames

            sf_loss, sf_log = self.compute_self_forcing_loss(
                original_latent=gt_future,
                predicted_latent=pred_future,
                gradient_mask=None,
                use_normalization=sf_normalize,
            )

            # add to loss_dict and to final loss (weight it)
            loss_dict["loss_self_forcing"] = sf_loss
            # if you want combined loss add weighted to final scalar return (if you return scalar)
            # your training loop expects loss_dict["loss_noise"], so ensure outer code adds them correctly.
            # e.g., if your trainer sums losses: total_loss = loss_noise + sf_weight * sf_loss
        else:
            sf_log = {}

        return loss_dict, torch.tensor(0.0, device=device, dtype=dtype)


# class RewardModel(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args

#     def forward(self, batch):
#         obs_latent = batch["latent"]
#         action = batch["action"]
#         out_reward = batch["reward"]
