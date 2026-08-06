import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import einops
import mediapy
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from config import wm_args
from dataset.dataset_droid_exp33_new import Dataset_mix
from dataset.dataset_libero import DatasetLibero
from models.ctrl_world_new import CtrlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from tqdm import tqdm

# ------------------------------
# utils
# ------------------------------


def _get_inner_module(m):
    """解除 DDP 包装"""
    return m.module if hasattr(m, "module") else m


def encode_img_to_latent(model, img, device):
    """
    img: (T, 3, H, W)
    return latents: (T, 4, 32, 32)
    """
    model = _get_inner_module(model)
    vae = _get_inner_module(model.vae)
    img = img.to(device)

    with torch.no_grad():
        latent = vae.encode(img).latent_dist.sample()
        latent = latent.mul_(vae.config.scaling_factor).cpu()

    return latent


# ------------------------------
# 核心验证函数（视频生成 + 奖励指标）
# ------------------------------
# def validate_video_generation(
#     model, val_dataset, args, step, save_dir, dir_name, accelerator
# ):
#     device = accelerator.device
#     pipeline = _get_inner_module(model).pipeline

#     # 随机选 val 样本
#     total_val_samples = len(val_dataset)
#     sample_num = min(args.videos_col, total_val_samples)
#     ids = random.sample(range(total_val_samples), k=sample_num)

#     batch_list = [val_dataset[i] for i in ids]

#     rewards = torch.cat([b["reward"][None] for b in batch_list], dim=0).to(device)
#     if torch.all(rewards == 0.0) and torch.all(torch.isfinite(rewards)):
#         # print("All rewards are zero, skip this step")
#         return None

#     # GT latent
#     if args.is_img_pregenerated:
#         video_gt = torch.cat([b["latent"][None] for b in batch_list], dim=0).to(device)
#     else:
#         video_gt = torch.cat(
#             [encode_img_to_latent(model, b["img"], device)[None] for b in batch_list],
#             dim=0,
#         ).to(device)

#     text = [b["text"] for b in batch_list]
#     actions = torch.cat([b["action"][None] for b in batch_list], dim=0).to(device)
#     imgs = torch.cat([b["img"][None] for b in batch_list], dim=0).to(device)

#     his_latent_gt = video_gt[:, : args.num_history]
#     future_latent_gt = video_gt[:, args.num_history :]
#     current_latent = future_latent_gt[:, 0]

#     # ------------------------------
#     # encode actions → latent
#     # ------------------------------
#     with torch.no_grad():
#         bsz = actions.shape[0]
#         actions = actions.to(
#             device=device, dtype=_get_inner_module(model).unet.dtype
#         )  # [1, 11, 7], b t d_a

#         action_latent = _get_inner_module(model).action_encoder(
#             actions,
#             text,
#             _get_inner_module(model).tokenizer,
#             _get_inner_module(model).text_encoder,
#             args.frame_level_cond,
#         )  # [1, 11, 1024], b t d'_a

#         # history = None时，action_latent 只能取 Future_frames

#         _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
#             pipeline,
#             image=current_latent,  # current observation, [1, 4, 24, 40], b c h w
#             text=action_latent,  # action embeddings, history + future
#             width=args.width,
#             height=args.height * args.num_views,
#             num_frames=args.num_frames,
#             history=his_latent_gt,  # history observations, [1, 6, 4, 24, 40]
#             # history=None,
#             num_inference_steps=args.num_inference_steps,
#             decode_chunk_size=args.decode_chunk_size,
#             max_guidance_scale=args.guidance_scale,
#             fps=args.fps,
#             motion_bucket_id=args.motion_bucket_id,
#             mask=None,
#             output_type="latent",
#             return_dict=False,
#             frame_level_cond=args.frame_level_cond,
#             his_cond_zero=args.his_cond_zero,
#         )
#         # pred_latents: [1, 5, 4, 24, 40]

#         # 奖励预测
#         imgs_flat = einops.rearrange(imgs, "b t c h w -> (b t) c h w")
#         act_lat_flat = einops.rearrange(action_latent, "b t c -> (b t) c")

#         # pred_logits, _ = _get_inner_module(model).reward_classifier(
#         #     imgs_flat,
#         #     act_lat_flat,
#         # )
#         # pred_rewards = torch.softmax(pred_logits, dim=-1)[:, 1]
#         # pred_rewards = pred_rewards.reshape(bsz, args.num_frames + args.num_history)
#         pred_score = _get_inner_module(model).reward_classifier.predict_score(
#             imgs_flat,
#             act_lat_flat,
#         )
#         pred_rewards = pred_score.reshape(bsz, args.num_frames + args.num_history)

#     # ------------------------------
#     # Reward Metrics
#     # ------------------------------
#     true_r = rewards.flatten().float()
#     pred_r = pred_rewards.flatten()

#     mse = torch.nn.functional.mse_loss(pred_r, true_r)
#     mae = torch.nn.functional.l1_loss(pred_r, true_r)

#     def pearson(x, y, eps=1e-8):
#         xm = x - x.mean()
#         ym = y - y.mean()
#         min_div = torch.clamp(xm.norm() * ym.norm(), min=eps)
#         return (xm * ym).sum() / min_div

#     corr = pearson(pred_r, true_r)

#     print("------ Reward Metrics ------")
#     print(f"Tasks: {text}")
#     print("MSE:", mse.item())
#     print("MAE:", mae.item())
#     print("Correlation:", corr.item())
#     print(f"GT_rewards: {true_r}")
#     print(f"Pred_rewards: {pred_r}")
#     print("---------------------------")

#     # ------------------------------
#     # 视频 decode + 保存
#     # ------------------------------

#     pred_latents = einops.rearrange(
#         pred_latents, "b t c (m h) (n w) -> (b m n) t c h w", m=args.num_views, n=1
#     )

#     video_gt = torch.cat([his_latent_gt, future_latent_gt], dim=1)
#     video_gt = einops.rearrange(
#         video_gt, "b t c (m h) (n w) -> (b m n) t c h w", m=args.num_views, n=1
#     )

#     # decode latents → RGB
#     def decode_latents(z):
#         z = z.flatten(0, 1)  # (B*T, C, H, W)
#         out = []
#         decode_kwargs = {}

#         for i in range(0, z.shape[0], args.decode_chunk_size):
#             chunk = z[i : i + args.decode_chunk_size]
#             chunk = chunk / pipeline.vae.config.scaling_factor

#             decode_kwargs["num_frames"] = chunk.shape[0]
#             out.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)

#         out = torch.cat(out, dim=0)
#         # reshape 回 (B*num_views, T, C, H, W)
#         return out.reshape(bsz * args.num_views, -1, *out.shape[1:])

#     video_gt = decode_latents(video_gt)  # [Bv, T, C, H, W]
#     pred_video = decode_latents(pred_latents)  # [Bv, T_f, C, H, W]

#     # to uint8
#     video_gt = ((video_gt / 2 + 0.5).clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
#     pred_video = (
#         ((pred_video / 2 + 0.5).clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
#     )

#     # 纵向 concat：上 GT，下 pred
#     T_his = args.num_history
#     T_pred = args.num_frames

#     # 构造完整预测序列（history + pred）
#     video_pred_full = np.concatenate(
#         [video_gt[:, :T_his], pred_video],  # (Bv, 6, 3, H, W)  # (Bv, 5, 3, H, W)
#         axis=1,
#     )  # → (Bv, 11, 3, H, W)
#     video_pred_full = einops.rearrange(video_pred_full, "b t c h w -> b t h w c")

#     # 拼成上下结构（GT 在上，Pred 在下）
#     video_gt = einops.rearrange(video_gt, "b t c h w -> b t h w c")
#     stacked = np.concatenate(
#         [video_gt, video_pred_full], axis=2
#     )  # shape: (Bv, T, H*2, W, 3)
#     # 将 batch 按横向拼接（拼在 width 方向）
#     hcat = np.concatenate([s for s in stacked], axis=2)  # shape: (T, H*2, W*(Bv), 3)

#     # mediapy 需要 (T, H, W, 3) 的 uint8 视频帧
#     vis = hcat.astype(np.uint8)

#     os.makedirs(f"{save_dir}/{dir_name}", exist_ok=True)
#     save_path = f"{save_dir}/{dir_name}/val_steps_{step}.mp4"

#     mediapy.write_video(save_path, vis, fps=2)
#     print(f"[Saved] {save_path}")

#     return {
#         "MSE": mse.item(),
#         "MAE": mae.item(),
#         "Correlation": corr.item(),
#     }


def validate_video_generation(
    model, val_dataset, args, step, save_dir, dir_name, accelerator
):
    device = accelerator.device
    pipeline = _get_inner_module(model).pipeline

    # 随机选 val 样本
    total_val_samples = len(val_dataset)
    sample_num = min(args.videos_col, total_val_samples)
    ids = random.sample(range(total_val_samples), k=sample_num)

    batch_list = [val_dataset[i] for i in ids]

    rewards = torch.cat([b["reward"][None] for b in batch_list], dim=0).to(device)
    if torch.all(rewards == 0.0) and torch.all(torch.isfinite(rewards)):
        # print("All rewards are zero, skip this step")
        return None

    # GT latent
    if args.is_img_pregenerated:
        video_gt = torch.cat([b["latent"][None] for b in batch_list], dim=0).to(device)
    else:
        video_gt = torch.cat(
            [encode_img_to_latent(model, b["img"], device)[None] for b in batch_list],
            dim=0,
        ).to(device)

    text = [b["text"] for b in batch_list]
    actions = torch.cat([b["action"][None] for b in batch_list], dim=0).to(device)
    imgs = torch.cat([b["img"][None] for b in batch_list], dim=0).to(device)

    his_latent_gt = video_gt[:, : args.num_history]
    future_latent_gt = video_gt[:, args.num_history :]
    current_latent = future_latent_gt[:, 0]

    # ------------------------------
    # encode actions → latent
    # ------------------------------
    with torch.no_grad():
        bsz = actions.shape[0]
        actions = actions.to(
            device=device, dtype=_get_inner_module(model).unet.dtype
        )  # [1, 11, 7], b t d_a

        action_latent = _get_inner_module(model).action_encoder(
            actions,
            text,
            _get_inner_module(model).tokenizer,
            _get_inner_module(model).text_encoder,
            args.frame_level_cond,
        )  # [1, 11, 1024], b t d'_a

        # history = None时，action_latent 只能取 Future_frames

        pred_frames, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,  # current observation, [1, 4, 24, 40], b c h w
            text=action_latent,  # action embeddings, history + future
            width=args.width,
            height=args.height * args.num_views,
            num_frames=args.num_frames,  # This is the number of future frames to predict
            history=his_latent_gt,  # history observations, [1, 6, 4, 24, 40]
            # history=None,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type="frame",
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )
        # pred_latents: [1, 5, 4, 24, 40] -> [bsz, num_future_frames, c, h, w]

        # --- 奖励预测 ---
        # 1. 解码预测的未来视频潜码
        # pred_latents shape: [bsz, num_future_frames, c, h, w]
        pred_latents_flat = einops.rearrange(pred_latents, "b t c h w -> (b t) c h w")

        # Decode to pixel space
        decoded_frames = []
        decode_kwargs = {}
        for i in range(0, pred_latents_flat.shape[0], args.decode_chunk_size):
            chunk = pred_latents_flat[i : i + args.decode_chunk_size]
            chunk = chunk / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_chunk = pipeline.vae.decode(chunk, **decode_kwargs).sample
            decoded_frames.append(decoded_chunk)

        decoded_frames = torch.cat(decoded_frames, dim=0)
        # Reshape back to [bsz, num_future_frames, c, h, w]
        pred_video = decoded_frames.reshape(
            bsz, args.num_frames, *decoded_frames.shape[1:]
        )
        # pred_video is now the decoded future video: [bsz, num_future_frames, c, h, w]

        # 2. 提取未来动作 (与预测的未来视频帧对应)
        future_actions = actions[
            :, -args.num_frames :, ...
        ]  # [bsz, num_future_frames, action_dim]
        future_action_latent = action_latent[
            :, -args.num_frames :, ...
        ]  # [bsz, num_future_frames, action_latent_dim]
        # Reshape for reward model: [bsz * num_future_frames, action_latent_dim]
        future_action_latent_flat = einops.rearrange(
            future_action_latent, "b t c -> (b t) c"
        )

        # 3. 使用解码后的未来视频和未来动作预测奖励
        # pred_video shape: [bsz, num_future_frames, c, h, w]
        # future_action_latent_flat shape: [bsz * num_future_frames, action_latent_dim]
        pred_video_flat = einops.rearrange(
            pred_video, "b t c h w -> (b t) c h w"
        )  # [bsz * num_future_frames, c, h, w]

        pred_score = _get_inner_module(model).reward_classifier.predict_score(
            pred_video_flat,
            future_action_latent_flat,
        )
        # Reshape back to [bsz, num_future_frames]
        pred_rewards = pred_score.reshape(
            bsz, args.num_frames
        )  # args.num_frames == num_future_frames
        # pred_rewards = F.softmax(pred_rewards, dim=-1)  # 再对frames进行softmax
        # for b in range(pred_rewards.shape[0]):
        #     pred_rewards[b] = pred_rewards[b] / pred_rewards[b].sum()  # 对frames进行归一化

        # 4. 提取GT未来奖励用于比较
        future_rewards = rewards[:, -args.num_frames :]  # [bsz, num_future_frames]

    # ------------------------------
    # Reward Metrics (使用未来奖励)
    # ------------------------------
    true_r = future_rewards.flatten().float()
    pred_r = pred_rewards.flatten()

    mse = torch.nn.functional.mse_loss(pred_r, true_r)
    mae = torch.nn.functional.l1_loss(pred_r, true_r)

    def pearson(x, y, eps=1e-8):
        xm = x - x.mean()
        ym = y - y.mean()
        min_div = torch.clamp(xm.norm() * ym.norm(), min=eps)
        return (xm * ym).sum() / min_div

    corr = pearson(pred_r, true_r)

    print(
        "------ Future Reward Metrics (from decoded predicted video and future actions) ------"
    )
    print(f"Tasks: {text}")
    print("MSE:", mse.item())
    print("MAE:", mae.item())
    print("Correlation:", corr.item())
    print(f"GT_Future_Rewards: {true_r}")
    print(f"Pred_Future_Rewards: {pred_r}")
    print("---------------------------")

    # ------------------------------
    # 视频 decode + 保存 (这部分需要解码 pred_latents 和 video_gt)
    # pred_video (用于奖励计算) 已经在上面解码好了，但视频保存需要原始的 pred_latents 和 video_gt
    # ------------------------------

    # 重新解码 pred_latents for visualization, using the same logic as video_gt
    pred_latents_vis = einops.rearrange(
        pred_latents, "b t c (m h) (n w) -> (b m n) t c h w", m=args.num_views, n=1
    )

    video_gt = torch.cat([his_latent_gt, future_latent_gt], dim=1)
    video_gt = einops.rearrange(
        video_gt, "b t c (m h) (n w) -> (b m n) t c h w", m=args.num_views, n=1
    )

    # decode latents → RGB
    def decode_latents(z):
        z = z.flatten(0, 1)  # (B*T, C, H, W)
        out = []
        decode_kwargs = {}

        for i in range(0, z.shape[0], args.decode_chunk_size):
            chunk = z[i : i + args.decode_chunk_size]
            chunk = chunk / pipeline.vae.config.scaling_factor

            decode_kwargs["num_frames"] = chunk.shape[0]
            out.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)

        out = torch.cat(out, dim=0)
        # reshape 回 (B*num_views, T, C, H, W)
        return out.reshape(bsz * args.num_views, -1, *out.shape[1:])

    video_gt = decode_latents(video_gt)  # [Bv, T, C, H, W]
    pred_video_vis = decode_latents(pred_latents_vis)  # [Bv, T_f, C, H, W]

    # to uint8
    video_gt = ((video_gt / 2 + 0.5).clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
    pred_video_vis = (
        ((pred_video_vis / 2 + 0.5).clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
    )

    # 纵向 concat：上 GT，下 pred
    T_his = args.num_history
    T_pred = args.num_frames

    # 构造完整预测序列（history + pred） for visualization
    # Note: history part for pred_video_vis is not available from prediction, so we reuse GT history
    video_pred_full = np.concatenate(
        [video_gt[:, :T_his], pred_video_vis],  # (Bv, 6, 3, H, W)  # (Bv, 5, 3, H, W)
        axis=1,
    )  # → (Bv, 11, 3, H, W)
    video_pred_full = einops.rearrange(video_pred_full, "b t c h w -> b t h w c")

    # 拼成上下结构（GT 在上，Pred 在下）
    video_gt = einops.rearrange(video_gt, "b t c h w -> b t h w c")
    stacked = np.concatenate(
        [video_gt, video_pred_full], axis=2
    )  # shape: (Bv, T, H*2, W, 3)
    # 将 batch 按横向拼接（拼在 width 方向）
    hcat = np.concatenate([s for s in stacked], axis=2)  # shape: (T, H*2, W*(Bv), 3)

    # mediapy 需要 (T, H, W, 3) 的 uint8 视频帧
    vis = hcat.astype(np.uint8)

    os.makedirs(f"{save_dir}/{dir_name}", exist_ok=True)
    save_path = f"{save_dir}/{dir_name}/val_steps_{step}.mp4"

    mediapy.write_video(save_path, vis, fps=2)
    print(f"[Saved] {save_path}")

    return {
        "MSE": mse.item(),
        "MAE": mae.item(),
        "Correlation": corr.item(),
    }


# ------------------------------
# 主函数：加载模型并验证
# ------------------------------
def main_val():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--val_model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="droid")
    parser.add_argument("--dataset_root_path", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--dirname", type=str, default="samples_val_1202")

    args_cli = parser.parse_args()

    args = wm_args()
    args.videos_col = 1
    args.val_model_path = args_cli.val_model_path
    args.dataset_root_path = args_cli.dataset_root_path

    # dataset
    if args_cli.dataset == "libero":
        val_dataset = DatasetLibero(args, mode="val")
    else:
        val_dataset = Dataset_mix(args, mode="val")

    accelerator = Accelerator()

    # load model
    model = CtrlWorld(args)
    print(f"[Load] checkpoint: {args.val_model_path}")
    model.load_state_dict(torch.load(args.val_model_path, map_location="cpu"))
    model.to(accelerator.device)
    model.eval()

    mean_metrics = {
        "MSE": 0,
        "MAE": 0,
        "Correlation": 0,
    }
    corr_count = 0
    idx = 0
    while idx < args_cli.num_steps:
        rm_metrics = validate_video_generation(
            model=model,
            val_dataset=val_dataset,
            args=args,
            step=idx,
            save_dir="./output_val",
            dir_name=args_cli.dirname,
            accelerator=accelerator,
        )
        if not rm_metrics:
            continue

        print(f"[Step {idx}] collected.")
        idx += 1
        for key in mean_metrics:
            mean_metrics[key] += rm_metrics[key]
            if key == "Correlation" and rm_metrics[key] > 0.0:
                corr_count += 1

    print(
        f"\n------ Mean Reward Metrics of {args_cli.num_steps * args.videos_col} samples------"
    )
    for key in mean_metrics:
        if key != "Correlation":
            mean_metrics[key] /= args_cli.num_steps
        else:
            if corr_count == 0:
                mean_metrics[key] = 0
            else:
                mean_metrics[key] /= corr_count
        print(f"{key}: {mean_metrics[key]:.4f}")
    print("---------------------------\n")


if __name__ == "__main__":
    main_val()
