# validate_new.py
import json
import os
import sys
import warnings

script_path = os.path.abspath(__file__)
wm_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(wm_dir)
root_dir = os.path.dirname(os.path.dirname(wm_dir))
sys.path.append(root_dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import random
from pathlib import Path

import einops
import mediapy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from accelerate import Accelerator

# from config import wm_args
from configs.wm_offline_config import wm_args
from dataset.dataset_droid_exp33_new import Dataset_mix
from dataset.dataset_libero import DatasetLibero
from models.ctrl_world_new import CtrlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from torchvision import transforms
from tqdm import tqdm

# try optional imports
try:
    import lpips
except Exception:
    lpips = None

try:
    import clip as clip_pkg
except Exception:
    clip_pkg = None

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None

try:
    from torchvision.models.video import r3d_18
    from torchvision.transforms import Resize
except Exception:
    r3d_18 = None

try:
    from scipy import linalg as scipy_linalg
except Exception:
    scipy_linalg = None


# ------------------------------
# utils
# ------------------------------
def _get_inner_module(m):
    """Unwrap the DDP packaging"""
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


# ---------- 新增：模型加载器（放在 utils 区域） ----------
def _load_lpips_model(lpips_path, device):
    """
    lpips_path: None or str/path to checkpoint (.pt/.pth)
    Returns instantiated LPIPS module on device or None if lpips package missing.
    """
    if lpips is None:
        print("[LPIPS] lpips package not installed; LPIPS will be skipped.")
        return None
    model = lpips.LPIPS(net="vgg")
    if lpips_path:
        # try to load a state_dict
        try:
            ck = torch.load(lpips_path, map_location="cpu")
            if isinstance(ck, dict) and "state_dict" in ck:
                state = ck["state_dict"]
            else:
                state = ck
            # Try to inject weights into inner net if necessary
            try:
                model.net.load_state_dict(state, strict=False)
            except Exception as e:
                print(
                    f"[LPIPS] load_state_dict strict=False failed: {e}; trying direct load to module"
                )
                try:
                    model.load_state_dict(state, strict=False)
                except Exception as e2:
                    print(f"[LPIPS] fallback load failed: {e2}")
        except Exception as e:
            print(f"[LPIPS] failed to load local checkpoint '{lpips_path}': {e}")
    model = model.to(device)
    model.eval()
    return model


def _load_clip_model(clip_path, device):
    """
    Try several strategies:
      1) clip.load(clip_path, device)  # if clip_path is a model name or a local model .pt supported by clip.load
      2) torch.jit.load(clip_path)       # if user provided traced/scripted model
      3) fallback to clip.load("ViT-B/32")
    Returns (model, preprocess) or (None, None)
    """
    if clip_pkg is None:
        print("[CLIP] clip package not available; CLIP metrics will be skipped.")
        return None, None

    # try clip.load with provided path (clip supports local weight files in some setups)
    if clip_path:
        try:
            m, pre = clip_pkg.load(clip_path, device=device)  # try direct
            m.eval()
            print(f"[CLIP] loaded model from {clip_path} via clip.load")
            return m, pre
        except Exception as e:
            print(
                f"[CLIP] clip.load('{clip_path}') failed: {e}; trying torch.jit.load fallback"
            )
            try:
                traced = torch.jit.load(clip_path, map_location=device)
                traced.eval()
                # cannot get preprocess from traced, fallback to clip default preprocess for the architecture
                _, pre = clip_pkg.load("ViT-B/32", device=device)
                print(f"[CLIP] loaded traced model via torch.jit.load({clip_path})")
                return traced, pre
            except Exception as e2:
                print(f"[CLIP] torch.jit.load('{clip_path}') failed: {e2}")

    # fallback to default
    try:
        m, pre = clip_pkg.load("ViT-B/32", device=device)
        m.eval()
        print("[CLIP] Using default ViT-B/32 (downloaded or cached).")
        return m, pre
    except Exception as e:
        print(f"[CLIP] default clip.load failed: {e}")
        return None, None


def _load_r3d_model(r3d_path, device):
    """
    Load r3d_18 model. If r3d_path provided, load weights from that path.
    Returns model or None.
    """
    try:
        # create model skeleton
        model = r3d_18(pretrained=False)
    except Exception:
        # torchvision may not have r3d_18 in this env
        try:
            from torchvision.models.video import r3d_18 as _r3d

            model = _r3d(pretrained=False)
        except Exception as e:
            print(f"[R3D] r3d_18 not available in torchvision: {e}")
            return None

    if r3d_path:
        try:
            ck = torch.load(r3d_path, map_location="cpu")
            # ck could be dict with state_dict key
            if isinstance(ck, dict) and "state_dict" in ck:
                state = ck["state_dict"]
            else:
                state = ck
            model.load_state_dict(state, strict=False)
            print(f"[R3D] loaded weights from {r3d_path}")
        except Exception as e:
            print(
                f"[R3D] failed to load weights from {r3d_path}: {e}; continuing with random init"
            )
    model = model.to(device)
    model.eval()
    return model


def _load_inception_model(inception_path, device):
    """
    Load inception_v3 to extract image features (pool features 2048).
    If inception_path provided, try to load state dict. Return model or None.
    """
    try:
        inception = torchvision.models.inception_v3(pretrained=False, aux_logits=False)
    except Exception as e:
        print(f"[INCEPTION] torchvision inception_v3 not available: {e}")
        return None

    if inception_path:
        try:
            ck = torch.load(inception_path, map_location="cpu")
            state = (
                ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
            )
            inception.load_state_dict(state, strict=False)
            print(f"[INCEPTION] loaded weights from {inception_path}")
        except Exception as e:
            print(
                f"[INCEPTION] failed to load inception weights: {e}; will try with random init"
            )

    # convert inception to feature extractor (remove final fc)
    # We'll extract features from the last pooling layer (before fc)
    inception.fc = torch.nn.Identity()
    inception.eval()
    inception = inception.to(device)
    return inception


# ---------- metric helpers ----------
def psnr_from_mse(mse, max_val=1.0, eps=1e-8):
    return 10.0 * torch.log10((max_val**2) / (mse + eps))


def compute_psnr_batch(pred, gt):
    """
    pred, gt: float tensors in [0,1], shape [B, T, C, H, W]
    returns mean_psnr (float)
    """
    assert pred.shape == gt.shape
    b, t, c, h, w = pred.shape
    pred_flat = pred.reshape(b * t, c, h, w)
    gt_flat = gt.reshape(b * t, c, h, w)
    mse = F.mse_loss(pred_flat, gt_flat, reduction="none")
    mse = mse.view(mse.shape[0], -1).mean(dim=1)  # per-frame MSE
    psnr_per_frame = 10.0 * torch.log10(1.0 / (mse + 1e-8))
    return float(psnr_per_frame.mean().item())


def compute_lpips_batch(pred, gt, device, lpips_model=None):
    """
    pred, gt: [B,T,C,H,W] float in [0,1]
    lpips_model: an instantiated lpips.LPIPS on device (or None)
    """
    if lpips is None or lpips_model is None:
        warnings.warn("lpips model not available; skipping LPIPS.")
        return None
    Loss = lpips_model
    b, t, c, h, w = pred.shape
    pred = (pred * 2.0 - 1.0).to(device)
    gt = (gt * 2.0 - 1.0).to(device)
    total = []
    with torch.no_grad():
        for bi in range(b):
            for ti in range(t):
                d = Loss(pred[bi, ti], gt[bi, ti])
                total.append(d.cpu().item())
    return float(np.mean(total))


def _compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # mu: numpy vectors
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # compute sqrt of product
    covmean = None
    if scipy_linalg is not None:
        covmean = scipy_linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    else:
        # fallback: use numpy + eigen decomposition (less stable but try)
        try:
            from numpy.linalg import eigh

            eigvals, eigvecs = eigh(sigma1.dot(sigma2))
            sqrt_eigvals = np.sqrt(np.clip(eigvals, a_min=0, a_max=None))
            covmean = (eigvecs * sqrt_eigvals).dot(eigvecs.T)
        except Exception:
            covmean = np.zeros_like(sigma1)

    tr_covmean = np.trace(covmean)
    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(np.real(fd))


#  A) if inception_model provided -> extract features with it
#  B) else try torchmetrics FrechetInceptionDistance
def compute_fid_images(pred, gt, device, inception_model=None):
    """
    pred, gt: [B, T, C, H, W] float in [0,1]
    inception_model: if provided, use it to extract features
    """
    # flatten to images
    pred_flat = einops.rearrange(pred, "b t c h w -> (b t) c h w")
    gt_flat = einops.rearrange(gt, "b t c h w -> (b t) c h w")

    # Option A: use provided inception_model to extract features
    if inception_model is not None:
        # preprocess: inception expects 299x299, normalized as imagenet
        preprocess = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def _extract_feat(x):
            # x: (N,C,H,W) float [0,1] tensor on cpu or device
            x = x.to(device)
            # resize & normalize per batch (do in float)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = preprocess(x)
            with torch.no_grad():
                f = inception_model(x)
                if isinstance(f, torch.Tensor):
                    return f.cpu().numpy()
                else:
                    return np.array(f)

        try:
            feat_pred = _extract_feat(pred_flat)
            feat_gt = _extract_feat(gt_flat)
            mu1 = np.mean(feat_pred, axis=0)
            mu2 = np.mean(feat_gt, axis=0)
            sigma1 = np.cov(feat_pred, rowvar=False)
            sigma2 = np.cov(feat_gt, rowvar=False)
            return _compute_frechet_distance(mu1, sigma1, mu2, sigma2)
        except Exception as e:
            warnings.warn(
                f"[FID] feature extraction with inception_model failed: {e}; fallback to torchmetrics if available."
            )

    # Option B: fallback to torchmetrics FID
    if FrechetInceptionDistance is None:
        warnings.warn(
            "torchmetrics FrechetInceptionDistance not available; skipping FID."
        )
        return None
    fid = FrechetInceptionDistance(feature=2048).to(device)
    # note: torchmetrics expects float in [0,1] or uint8 images; we'll convert to floats
    try:
        fid.update((pred_flat).to(device), real=False)
        fid.update((gt_flat).to(device), real=True)
        score = fid.compute().item()
        return score
    except Exception as e:
        warnings.warn(f"[FID] torchmetrics FID compute failed: {e}")
        return None


def compute_video_features_r3d(videos, device, r3d_model=None):
    """
    videos: [B, T, C, H, W] float in [0,1]
    r3d_model: optional preloaded model on device
    returns: numpy array [B, feat_dim] or None
    """
    if r3d_model is None:
        # try to create default one (fallback)
        if r3d_18 is None:
            warnings.warn("r3d_18 model unavailable; skipping FVD approx.")
            return None
        r3d_model = r3d_18(pretrained=True).to(device)
    r3d_model.eval()
    # extract feature extractor (remove final fc)
    feat_extractor = torch.nn.Sequential(*(list(r3d_model.children())[:-2])).to(device)
    v = videos.permute(0, 2, 1, 3, 4).to(device)
    with torch.no_grad():
        feats = feat_extractor(v)
        feats = feats.mean(dim=[2, 3, 4])
    return feats.cpu().numpy()


def compute_fvd_approx(pred, gt, device):
    """
    pred, gt: float tensors in [0,1], shape [B, T, C, H, W]
    returns float or None
    """
    feats_pred = compute_video_features_r3d(pred, device)
    feats_gt = compute_video_features_r3d(gt, device)
    if feats_pred is None or feats_gt is None:
        return None

    mu1 = np.mean(feats_pred, axis=0)
    mu2 = np.mean(feats_gt, axis=0)
    sigma1 = np.cov(feats_pred, rowvar=False)
    sigma2 = np.cov(feats_gt, rowvar=False)
    try:
        fd = _compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    except Exception as e:
        warnings.warn(f"FVD approximation failed: {e}")
        fd = None
    return fd


def compute_fvd_approx_with_model(pred, gt, device, r3d_model):
    feats_pred = compute_video_features_r3d(pred, device, r3d_model)
    feats_gt = compute_video_features_r3d(gt, device, r3d_model)
    if feats_pred is None or feats_gt is None:
        return None
    mu1 = np.mean(feats_pred, axis=0)
    mu2 = np.mean(feats_gt, axis=0)
    sigma1 = np.cov(feats_pred, rowvar=False)
    sigma2 = np.cov(feats_gt, rowvar=False)
    return _compute_frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_clip_score(pred, gt, device, clip_model=None, clip_preprocess=None):
    """
    pred, gt: float [0,1] tensors [B,T,C,H,W]
    clip_model, clip_preprocess: from clip.load
    """
    if clip_pkg is None or clip_model is None or clip_preprocess is None:
        warnings.warn("CLIP model not available; skipping CLIP score.")
        return None

    model = clip_model
    preprocess = clip_preprocess
    b, t, c, h, w = pred.shape
    video_embeds_pred = []
    video_embeds_gt = []
    with torch.no_grad():
        for bi in range(b):
            frames_pred_imgs = []
            frames_gt_imgs = []
            for ti in range(t):
                # clip preprocess expects PIL or tensor in [0,1], but preprocess from clip returns tensor after transform
                # convert each frame to PIL-like via numpy -> PIL
                pil_pred = mediapy.Image(pred[bi, ti].cpu().numpy().transpose(1, 2, 0))
                pil_gt = mediapy.Image(gt[bi, ti].cpu().numpy().transpose(1, 2, 0))
                p_pred = preprocess(pil_pred).unsqueeze(0).to(device)
                p_gt = preprocess(pil_gt).unsqueeze(0).to(device)
                emb_pred = model.encode_image(p_pred).float()
                emb_gt = model.encode_image(p_gt).float()
                frames_pred_imgs.append(emb_pred)
                frames_gt_imgs.append(emb_gt)
            emb_p = torch.mean(torch.cat(frames_pred_imgs, dim=0), dim=0, keepdim=True)
            emb_g = torch.mean(torch.cat(frames_gt_imgs, dim=0), dim=0, keepdim=True)
            video_embeds_pred.append(emb_p)
            video_embeds_gt.append(emb_g)
        pred_stack = torch.cat(video_embeds_pred, dim=0)
        gt_stack = torch.cat(video_embeds_gt, dim=0)
        pred_stack = pred_stack / pred_stack.norm(dim=-1, keepdim=True)
        gt_stack = gt_stack / gt_stack.norm(dim=-1, keepdim=True)
        cos = (pred_stack * gt_stack).sum(dim=-1)
        return float(cos.mean().cpu().item())


def normalize_metric(name, value):
    """
    Normalize a metric into [0,1] where higher is better.
    name: one of "PSNR","LPIPS","FID","FVD","CLIP"
    """
    if value is None:
        return None
    if name == "PSNR":
        # PSNR higher better: map to (0,1) by psnr/(psnr+1)
        return float(value / (value + 1.0))
    if name == "CLIP":
        # CLIP cosine in [-1,1] -> map to [0,1]
        return float((value + 1.0) / 2.0)
    # lower is better -> map to 1/(1+value)
    return float(1.0 / (1.0 + value))


def compute_metrics_for_videos(
    pred_video_torch, gt_video_torch, device, args, models=None
):
    """
    models: dict with optional keys:
       lpips_model, clip_model, clip_preprocess, r3d_model, inception_model
    """
    # ... 与之前实现完全相同，区别在于调用 compute_lpips_batch / compute_fid_images / compute_video_features_r3d / compute_clip_score
    lpips_model = models.get("lpips_model") if models else None
    clip_model = models.get("clip_model") if models else None
    clip_preprocess = models.get("clip_preprocess") if models else None
    r3d_model = models.get("r3d_model") if models else None
    inception_model = models.get("inception_model") if models else None

    # convert to [0,1] as before
    def to_0_1(x):
        if x.min() < 0.0:
            return (x + 1.0) / 2.0
        return x

    pred = to_0_1(pred_video_torch.float().cpu()).to(device)
    gt = to_0_1(gt_video_torch.float().cpu()).to(device)

    results = {}
    # PSNR (unchanged)
    try:
        results["PSNR"] = compute_psnr_batch(pred, gt)
    except Exception as e:
        print(f"[PSNR] failed: {e}")
        results["PSNR"] = None

    # LPIPS
    try:
        results["LPIPS"] = compute_lpips_batch(
            pred, gt, device, lpips_model=lpips_model
        )
    except Exception as e:
        print(f"[LPIPS] failed: {e}")
        results["LPIPS"] = None

    # FID
    try:
        results["FID"] = compute_fid_images(
            pred, gt, device, inception_model=inception_model
        )
    except Exception as e:
        print(f"[FID] failed: {e}")
        results["FID"] = None

    # FVD
    try:
        results["FVD"] = (
            compute_fvd_approx(pred, gt, device)
            if r3d_model is None
            else compute_fvd_approx_with_model(pred, gt, device, r3d_model)
        )
    except Exception as e:
        print(f"[FVD] failed: {e}")
        results["FVD"] = None

    # CLIP
    try:
        results["CLIP"] = compute_clip_score(
            (pred * 255).byte(),
            (gt * 255).byte(),
            device,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
        )
    except Exception as e:
        print(f"[CLIP] failed: {e}")
        results["CLIP"] = None

    # normalize & combine as before
    norm = {}
    for k, v in results.items():
        norm[k] = normalize_metric(k, v)
    vals = [v for v in norm.values() if v is not None]
    combined = float(np.mean(vals)) if len(vals) > 0 else None
    return {"raw": results, "norm": norm, "combined_score": combined}


# ------------------------------
# Core verification function (video generation + reward metrics + video quality metrics)
# ------------------------------
def validate_video_generation(
    model,
    val_dataset,
    args,
    step,
    save_dir,
    dir_name,
    accelerator,
    models=None,
):
    device = accelerator.device
    pipeline = _get_inner_module(model).pipeline

    # Randomly select val samples
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
        )  # [B, total_t, d_a]

        action_latent = _get_inner_module(model).action_encoder(
            actions,
            text,
            _get_inner_module(model).tokenizer,
            _get_inner_module(model).text_encoder,
            args.frame_level_cond,
        )  # [B, total_t, d']

        # Use pipeline to generate predicted frames (both frames and latents)
        pred_frames, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,  # current observation, [B, 4, h, w]
            text=action_latent,  # action embeddings
            width=args.width,
            height=args.height * args.num_views,
            num_frames=args.num_frames,
            history=his_latent_gt,
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

        # --- Reward Prediction (using decoded pred_frames as input) ---
        # pred_frames: List[np.ndarray], get pred_frames tensor:
        pred_frames_tensor = []
        for i in range(len(pred_frames)):
            pred_frames_tensor.append(torch.from_numpy(pred_frames[i]))
        pred_frames_tensor = torch.stack(
            pred_frames_tensor
        )  # [B, num_future_frames, H, W, C]
        pred_frames_tensor = einops.rearrange(
            pred_frames_tensor, "b t h w c -> b t c h w"
        ).to(device)
        # pred_frames shape: [B, num_future_frames, C, H, W] in model pixel range (likely [-1,1])

        pred_video_for_reward = pred_frames_tensor  # keep torch tensor
        future_actions = actions[:, -args.num_frames :, ...]
        future_action_latent = action_latent[:, -args.num_frames :, ...]
        future_action_latent_flat = einops.rearrange(
            future_action_latent, "b t c -> (b t) c"
        )
        pred_video_flat = einops.rearrange(
            pred_video_for_reward, "b t c h w -> (b t) c h w"
        )
        pred_score = _get_inner_module(model).reward_classifier.predict_score(
            pred_video_flat,
            future_action_latent_flat,
        )
        pred_rewards = pred_score.reshape(bsz, args.num_frames)
        future_rewards = rewards[:, -args.num_frames :]

    # ------------------------------
    # Reward Metrics (using future rewards)
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
    # Video decode + save (this part needs to decode pred_latents and video_gt)
    # ------------------------------
    pred_latents_vis = einops.rearrange(
        pred_latents, "b t c (m h) (n w) -> (b m n) t c h w", m=args.num_views, n=1
    )
    video_gt_latents = torch.cat([his_latent_gt, future_latent_gt], dim=1)
    video_gt_latents = einops.rearrange(
        video_gt_latents, "b t c (m h) (n w) -> (b m n) t c h w", m=args.num_views, n=1
    )

    # decode latents → RGB (torch tensor, likely in [-1,1])
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
        return out.reshape(bsz * args.num_views, -1, *out.shape[1:])

    video_gt_decoded = decode_latents(video_gt_latents)  # [Bv, T, C, H, W]
    pred_video_vis = decode_latents(pred_latents_vis)  # [Bv, T_f, C, H, W]

    # prepare tensors for metrics: we want predicted future and GT full sequence (history+future)
    # For metrics we compare predicted future frames vs GT future frames
    # Convert decoded to float in [-1,1] (as produced), then to [0,1]
    def to_0_1_tensor(x):
        x = x.float()
        if x.min() < 0.0:
            return (x + 1.0) / 2.0
        else:
            return x

    # gt future frames: video_gt_decoded[:, T_his:, ...]
    video_gt_decoded = to_0_1_tensor(video_gt_decoded).cpu()
    pred_video_vis = to_0_1_tensor(pred_video_vis).cpu()

    # For computing metrics, align shapes: both [Bv, T_pred, C, H, W]
    gt_future = video_gt_decoded[:, args.num_history :, ...]  # [Bv, T_pred, C, H, W]
    pred_future = pred_video_vis  # already [Bv, T_pred, C, H, W]

    # Ensure shapes match time dimension; if not, trim or pad
    if pred_future.shape[1] != gt_future.shape[1]:
        min_t = min(pred_future.shape[1], gt_future.shape[1])
        pred_future = pred_future[:, :min_t]
        gt_future = gt_future[:, :min_t]

    # compute video-level metrics
    metrics_res = compute_metrics_for_videos(
        pred_future,
        gt_future,
        device,
        args,
        models,
    )

    # to uint8 for visualization saving
    video_gt_np = ((video_gt_decoded / 1.0).clamp(0, 1) * 255).numpy().astype(np.uint8)
    pred_video_np = ((pred_video_vis / 1.0).clamp(0, 1) * 255).numpy().astype(np.uint8)

    # Vertical concat: GT on top, pred at the bottom
    T_his = args.num_history
    T_pred = args.num_frames

    video_pred_full = np.concatenate([video_gt_np[:, :T_his], pred_video_np], axis=1)
    video_pred_full = einops.rearrange(video_pred_full, "b t c h w -> b t h w c")
    video_gt_np = einops.rearrange(video_gt_np, "b t c h w -> b t h w c")
    stacked = np.concatenate([video_gt_np, video_pred_full], axis=2)
    hcat = np.concatenate([s for s in stacked], axis=2)
    vis = hcat.astype(np.uint8)

    os.makedirs(f"{save_dir}/{dir_name}", exist_ok=True)
    save_path = f"{save_dir}/{dir_name}/val_steps_{step}.mp4"
    mediapy.write_video(save_path, vis, fps=2)
    print(f"[Saved] {save_path}")

    # Save metrics json
    metrics_save = {
        "step": step,
        "reward_metrics": {
            "MSE": mse.item(),
            "MAE": mae.item(),
            "Correlation": corr.item(),
        },
        "video_metrics": metrics_res,
    }
    json_path = f"{save_dir}/{dir_name}/metrics_step_{step}.json"
    with open(json_path, "w") as f:
        json.dump(metrics_save, f, indent=2)
    print(f"[Saved metrics] {json_path}")

    # return reward metrics merged with video metrics summary for external aggregation
    out = {
        "MSE": mse.item(),
        "MAE": mae.item(),
        "Correlation": corr.item(),
        "VideoMetrics": metrics_res,
    }
    return out


# ------------------------------
# Main function: Load the model and verify it
# ------------------------------
def main_val():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--val_model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="droid")
    parser.add_argument("--dataset_root_path", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--dirname", type=str, default="samples_val_1202")

    parser.add_argument(
        "--lpips_path",
        type=str,
        default=None,
        help="local lpips checkpoint path (.pth/.pt)",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        help="local CLIP model path (.pt) or model name",
    )
    parser.add_argument(
        "--r3d_path", type=str, default=None, help="local r3d weights path (.pth/.pt)"
    )
    parser.add_argument(
        "--inception_path",
        type=str,
        default=None,
        help="local inception_v3 weights path (.pth/.pt) for FID",
    )

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
    device = accelerator.device
    models = {}
    models["lpips_model"] = _load_lpips_model(args_cli.lpips_path, device)
    clip_model, clip_preprocess = _load_clip_model(args_cli.clip_path, device)
    models["clip_model"] = clip_model
    models["clip_preprocess"] = clip_preprocess
    models["r3d_model"] = _load_r3d_model(args_cli.r3d_path, device)
    models["inception_model"] = _load_inception_model(args_cli.inception_path, device)

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
            models=models,  # 新增参数
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
