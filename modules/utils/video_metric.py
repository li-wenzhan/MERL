"""
A unified, reusable video evaluation module.

Supported metrics:
- PSNR   (frame-level)
- SSIM   (frame-level)
- LPIPS  (frame-level perceptual)
- CLIP   (frame-level semantic, PIL-based)
- FID    (frame-level distribution)
- FVD    (video-level distribution via 3D CNN features)

Design principles:
- Internally, all video tensors use shape [B, T, C, H, W]
- External inputs may be [T, C, H, W] or [B, T, C, H, W]
- Explicit, deterministic, reviewer-friendly behavior
"""

import contextlib
import os
from typing import List, Optional, Sequence, Tuple

import math
import numpy as np
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
from torchvision.models.video import r3d_18

from PIL import Image

import lpips
import clip
import pytorch_fid.inception as fid_inception_module
from pytorch_fid.inception import InceptionV3
from skimage.metrics import structural_similarity as ssim

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

_mnt_dir = os.environ.get("MNT_CACHE_DIR", None)
if _mnt_dir:
    os.environ["TORCH_HOME"] = os.path.join(_mnt_dir, "torch")

_FID_INCEPTION_FILENAME = "pt_inception-2015-12-05-6726825d.pth"
_R3D18_FILENAME = "r3d_18-b3b3357e.pth"

# -----------------------------------------------------------------------------
# Global cached models (lazy init)
# -----------------------------------------------------------------------------
_lpips_models = {}
_clip_model = None
_clip_preprocess = None
_fid_model = None
_r3d_models = {}


def _resolve_local_checkpoint(
    filename: str,
    *,
    explicit_env_keys: Sequence[str] = (),
) -> Optional[str]:
    candidates = []

    for env_key in explicit_env_keys:
        raw_value = str(os.environ.get(env_key, "") or "").strip()
        if not raw_value:
            continue
        expanded = os.path.abspath(os.path.expanduser(raw_value))
        if os.path.isdir(expanded):
            candidates.append(os.path.join(expanded, filename))
        else:
            candidates.append(expanded)

    raw_torch_home = str(os.environ.get("TORCH_HOME", "") or "").strip()
    raw_mnt_cache_dir = str(os.environ.get("MNT_CACHE_DIR", "") or "").strip()
    if raw_mnt_cache_dir:
        candidates.append(
            os.path.join(raw_mnt_cache_dir, "torch", "hub", "checkpoints", filename)
        )
    if raw_torch_home:
        candidates.append(os.path.join(raw_torch_home, "hub", "checkpoints", filename))

    try:
        candidates.append(os.path.join(torch.hub.get_dir(), "checkpoints", filename))
    except Exception:
        pass

    candidates.append(
        os.path.join(os.path.expanduser("~/.cache/torch/hub/checkpoints"), filename)
    )

    seen = set()
    for candidate in candidates:
        normalized = os.path.abspath(os.path.expanduser(str(candidate)))
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isfile(normalized):
            return normalized
    return None


def _load_checkpoint_state(local_path: str):
    state = torch.load(local_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


@contextlib.contextmanager
def _patch_fid_weight_download(local_path: Optional[str]):
    if not local_path:
        yield
        return

    def _load_from_local(*args, **kwargs):
        return _load_checkpoint_state(local_path)

    patched = []

    def _replace_attr(obj, attr_name: str):
        if obj is None or not hasattr(obj, attr_name):
            return
        patched.append((obj, attr_name, getattr(obj, attr_name)))
        setattr(obj, attr_name, _load_from_local)

    _replace_attr(torch.hub, "load_state_dict_from_url")
    _replace_attr(fid_inception_module, "load_state_dict_from_url")
    fid_model_zoo = getattr(fid_inception_module, "model_zoo", None)
    _replace_attr(fid_model_zoo, "load_url")
    torch_model_zoo = getattr(torch.utils, "model_zoo", None)
    _replace_attr(torch_model_zoo, "load_url")

    try:
        yield
    finally:
        for obj, attr_name, original in reversed(patched):
            setattr(obj, attr_name, original)


def _build_r3d18_feature_extractor(device: torch.device) -> nn.Module:
    device = torch.device(device)
    local_r3d_path = _resolve_local_checkpoint(
        _R3D18_FILENAME,
        explicit_env_keys=("MERL_R3D18_PATH", "R3D18_PATH"),
    )

    if local_r3d_path is not None:
        try:
            model = r3d_18(weights=None)
        except TypeError:
            model = r3d_18(pretrained=False)
        model.load_state_dict(_load_checkpoint_state(local_r3d_path), strict=False)
    else:
        try:
            model = r3d_18(weights="DEFAULT")
        except TypeError:
            model = r3d_18(pretrained=True)

    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_5d_video(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure video tensor has shape [B, T, C, H, W].

    Accepts:
      - [T, C, H, W]
      - [B, T, C, H, W]
    """
    if x.dim() == 4:
        return x.unsqueeze(0)
    elif x.dim() == 5:
        return x
    else:
        raise ValueError(f"Video tensor must be 4D or 5D, got {x.shape}")


# -----------------------------------------------------------------------------
# Frame utilities
# -----------------------------------------------------------------------------
def frames_to_tensor(
    frames: List[Image.Image],
    device: torch.device,
    size: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Convert a list of PIL frames to a tensor.

    Args:
        frames: list of PIL.Image
        device: torch device
        size: optional (W, H) resize

    Returns:
        Tensor [T, 3, H, W] in [0, 1]
    """
    if size is not None:
        frames = [f.resize(size, Image.BICUBIC) for f in frames]

    tf = T.ToTensor()
    return torch.stack([tf(f.convert("RGB")) for f in frames]).to(device)


# -----------------------------------------------------------------------------
# PSNR
# -----------------------------------------------------------------------------
def compute_psnr(gen: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Frame-level PSNR.

    Args:
        gen, gt: [T, C, H, W] or [B, T, C, H, W] in [0,1]
    """
    gen = ensure_5d_video(gen)
    gt = ensure_5d_video(gt)

    mse = F.mse_loss(gen, gt, reduction="none")
    mse = mse.mean(dim=[2, 3, 4])  # [B, T]
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.mean().item()


# -----------------------------------------------------------------------------
# SSIM
# -----------------------------------------------------------------------------
def compute_ssim(gen: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Frame-level SSIM.

    Args:
        gen, gt: [T, C, H, W] or [B, T, C, H, W] in [0,1]
    """
    gen = ensure_5d_video(gen)
    gt = ensure_5d_video(gt)

    gen = gen.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    B, T, C, H, W = gen.shape
    scores = []

    for b in range(B):
        for t in range(T):
            # compute SSIM per-channel, then average
            ssim_c = []
            for c in range(C):
                s = ssim(
                    gen[b, t, c],
                    gt[b, t, c],
                    data_range=1.0,
                )
                ssim_c.append(s)
            scores.append(sum(ssim_c) / C)

    return float(sum(scores) / len(scores))


# -----------------------------------------------------------------------------
# LPIPS
# -----------------------------------------------------------------------------
def _metric_device_key(device: torch.device) -> str:
    device = torch.device(device)
    if device.type != "cuda":
        return device.type
    if device.index is None:
        return "cuda"
    return f"cuda:{device.index}"


def compute_lpips(gen: torch.Tensor, gt: torch.Tensor, device: torch.device) -> float:
    """
    Frame-level LPIPS.

    Args:
        gen, gt: [T, C, H, W] or [B, T, C, H, W] in [0,1]
    """
    global _lpips_models

    device = torch.device(device)
    device_key = _metric_device_key(device)

    if device_key not in _lpips_models:
        _lpips_models[device_key] = lpips.LPIPS(net="alex").to(device)
        _lpips_models[device_key].eval()

    lpips_model = _lpips_models[device_key]

    gen = ensure_5d_video(gen)
    gt = ensure_5d_video(gt)

    # LPIPS expects [-1, 1]
    gen = gen * 2 - 1
    gt = gt * 2 - 1

    B, T = gen.shape[:2]
    dists = []

    with torch.no_grad():
        for b in range(B):
            for t in range(T):
                d = lpips_model(gen[b, t : t + 1], gt[b, t : t + 1])
                dists.append(d.item())

    return float(sum(dists) / len(dists))


# -----------------------------------------------------------------------------
# CLIP (single-video, PIL-based)
# -----------------------------------------------------------------------------
def compute_clip_score(
    gen_frames: List[Image.Image],
    real_frames: List[Image.Image],
    device: torch.device,
) -> float:
    """
    Frame-wise CLIP similarity for a single video.
    Args:
        gen_frames, real_frames: list[PIL.Image]
    """
    assert len(gen_frames) == len(real_frames)
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import os
        mnt_dir = os.environ.get("MNT_CACHE_DIR", None)
        default_clip_pt = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")
        # 三级 Fallback 路由机制
        if os.path.isfile(default_clip_pt):
            # 1. 默认路径命中（本地开发机常用）
            clip_path = default_clip_pt
            download_root = None
        elif mnt_dir and os.path.isfile(os.path.join(mnt_dir, "clip", "ViT-B-32.pt")):
            # 2. 挂载路径命中（集群 rjob 常用）
            clip_path = os.path.join(mnt_dir, "clip", "ViT-B-32.pt")
            download_root = None
        else:
            # 3. 未命中任何绝对路径，触发网络下载流程（如果配置了 mnt 则下载到 mnt 目录）
            clip_path = "ViT-B/32"
            download_root = os.path.join(mnt_dir, "clip") if mnt_dir else None
        try:
            _clip_model, _clip_preprocess = clip.load(clip_path, device=device, download_root=download_root)
            _clip_model.eval()
        except Exception as e:
            raise RuntimeError(
                f"CLIP 模型初始化失败。可能处于无网集群且挂载路径无权重。\n"
                f"当前读取的 MNT_CACHE_DIR 为: {mnt_dir}\n"
                f"原始报错: {str(e)}"
            )
    sims = []
    # ... [保留原代码后续的 with torch.no_grad(): 及特征提取逻辑完全不变] ...

    with torch.no_grad():
        for g, r in zip(gen_frames, real_frames):
            g_in = _clip_preprocess(g).unsqueeze(0).to(device)
            r_in = _clip_preprocess(r).unsqueeze(0).to(device)

            fg = _clip_model.encode_image(g_in)
            fr = _clip_model.encode_image(r_in)

            fg = fg / fg.norm(dim=-1, keepdim=True)
            fr = fr / fr.norm(dim=-1, keepdim=True)

            sims.append((fg * fr).sum().item())

    return float(sum(sims) / len(sims))


# -----------------------------------------------------------------------------
# FID (frame-level)
# -----------------------------------------------------------------------------
def compute_fid(gen: torch.Tensor, gt: torch.Tensor, device: torch.device) -> float:
    global _fid_model
    if _fid_model is None:
        local_fid_path = _resolve_local_checkpoint(
            _FID_INCEPTION_FILENAME,
            explicit_env_keys=("MERL_FID_INCEPTION_PATH", "FID_INCEPTION_PATH"),
        )
        try:
            with _patch_fid_weight_download(local_fid_path):
                _fid_model = InceptionV3([3]).to(device)
        except Exception as exc:
            if local_fid_path is not None:
                raise RuntimeError(
                    "FID 模型初始化失败：已找到本地 pt_inception 权重但加载失败。"
                    f" path={local_fid_path}; error={exc}"
                ) from exc
            raise RuntimeError(
                "FID 模型初始化失败，且未找到本地 pt_inception 权重。"
                " 请确认 MNT_CACHE_DIR/TORCH_HOME 指向包含"
                f" {_FID_INCEPTION_FILENAME} 的缓存目录，"
                "或显式设置 MERL_FID_INCEPTION_PATH。"
                f" 原始报错: {exc}"
            ) from exc
        _fid_model.eval()

    gen = ensure_5d_video(gen)
    gt = ensure_5d_video(gt)

    B, T = gen.shape[:2]
    gen = gen.reshape(B * T, *gen.shape[2:])
    gt = gt.reshape(B * T, *gt.shape[2:])

    def get_feats(x):
        with torch.no_grad():
            feats = _fid_model(x)[0]
            return feats.squeeze(-1).squeeze(-1)

    gen_feats = get_feats(gen)
    gt_feats = get_feats(gt)

    # convert to numpy
    gen_feats_np = gen_feats.cpu().numpy()
    gt_feats_np = gt_feats.cpu().numpy()

    # means
    mu1, mu2 = gen_feats_np.mean(0), gt_feats_np.mean(0)

    # covariances (bias=True for small samples)
    sigma1 = np.cov(gen_feats_np, rowvar=False, bias=True)
    sigma2 = np.cov(gt_feats_np, rowvar=False, bias=True)

    # numerical stability
    eps = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    # sqrtm
    covmean = linalg.sqrtm(sigma1 @ sigma2)

    # take real part if complex
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # final FID
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


# ---------------------------------------------------------------------------
# FVD (video-level)
# ---------------------------------------------------------------------------
class VideoFeatureExtractor(nn.Module):
    """Video-level feature extractor using cached 3D CNN weights when available."""

    def __init__(self, device: torch.device):
        super().__init__()
        device = torch.device(device)
        device_key = _metric_device_key(device)
        if device_key not in _r3d_models:
            _r3d_models[device_key] = _build_r3d18_feature_extractor(device)
        self.model = _r3d_models[device_key]

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            videos: [B, T, C, H, W]
        Returns:
            features: [B, 512]
        """
        videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        return self.model(videos)


def _compute_stats(features: np.ndarray):
    """
    Compute mean and covariance in a numerically stable way.
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(
        features, rowvar=False, bias=True
    )  # bias=True -> stable for small samples
    eps = 1e-6
    sigma += np.eye(sigma.shape[0]) * eps  # ensure positive definite
    return mu, sigma


def _frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    Frechet distance with numerical stability for small samples.
    """
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)

    if not np.isfinite(covmean).all():
        eps = 1e-6
        sigma1 += np.eye(sigma1.shape[0]) * eps
        sigma2 += np.eye(sigma2.shape[0]) * eps
        covmean = linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)


def compute_fvd(
    gen_videos: torch.Tensor, gt_videos: torch.Tensor, device: torch.device
) -> float:
    """
    Compute video-level FVD in a numerically stable way.
    Args:
        gen_videos, gt_videos: [B, T, C, H, W] or [T, C, H, W]
    """
    gen_videos = ensure_5d_video(gen_videos)
    gt_videos = ensure_5d_video(gt_videos)

    extractor = VideoFeatureExtractor(device)
    with torch.no_grad():
        gen_feats = extractor(gen_videos).cpu().numpy()
        gt_feats = extractor(gt_videos).cpu().numpy()

    mu_g, sigma_g = _compute_stats(gen_feats)
    mu_r, sigma_r = _compute_stats(gt_feats)

    return float(_frechet_distance(mu_g, sigma_g, mu_r, sigma_r))


def compute_all_metrics(
    gen_videos,
    real_videos,
    device,
    metrics_to_compute=None,
    *,
    strict=True,
    synchronize_cuda=False,
    enable_lpips_cpu_fallback=False,
):
    """
    Compute all video metrics and return as a dict.
    Args:
        gen_videos (torch.Tensor): (B, T, C, H, W), range [0, 1]
        real_videos (torch.Tensor): (B, T, C, H, W), range [0, 1]
        device (torch.device)
    Returns:
        dict containing successful metric values. When strict=False, failed metrics are
        recorded as '<metric>_error' instead of raising.
    """
    device = torch.device(device)
    gen_videos = ensure_5d_video(gen_videos)
    real_videos = ensure_5d_video(real_videos)

    if metrics_to_compute is None:
        metrics_to_compute = ["psnr", "ssim", "lpips", "fid", "fvd", "clips"]
    metrics_to_compute = [str(name).lower() for name in metrics_to_compute]

    metrics = {}

    def _maybe_sync():
        if synchronize_cuda and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _record_failure(metric_name: str, exc: Exception):
        if strict:
            raise exc
        metrics[f"{metric_name}_error"] = f"{type(exc).__name__}: {exc}"

    metric_fns = {
        "psnr": lambda: float(compute_psnr(gen_videos, real_videos)),
        "ssim": lambda: float(compute_ssim(gen_videos, real_videos)),
        "lpips": lambda: float(compute_lpips(gen_videos, real_videos, device)),
        "fid": lambda: float(compute_fid(gen_videos, real_videos, device)),
        "fvd": lambda: float(compute_fvd(gen_videos, real_videos, device)),
    }

    for metric_name in metrics_to_compute:
        if metric_name == "clips":
            continue
        if metric_name not in metric_fns:
            _record_failure(metric_name, ValueError(f"Unsupported metric '{metric_name}'"))
            continue
        try:
            _maybe_sync()
            metrics[metric_name] = metric_fns[metric_name]()
            _maybe_sync()
        except Exception as exc:
            if (
                metric_name == "lpips"
                and enable_lpips_cpu_fallback
                and device.type == "cuda"
            ):
                try:
                    cpu_device = torch.device("cpu")
                    metrics[metric_name] = float(
                        compute_lpips(gen_videos.cpu(), real_videos.cpu(), cpu_device)
                    )
                    metrics["lpips_cpu_fallback"] = 1
                    continue
                except Exception as fallback_exc:
                    _record_failure(metric_name, fallback_exc)
                    continue
            _record_failure(metric_name, exc)

    if "clips" in metrics_to_compute:
        try:
            _maybe_sync()
            B, T, _, _, _ = gen_videos.shape
            clip_scores = []
            for b in range(B):
                gen_frames = [
                    Image.fromarray(
                        (
                            gen_videos[b, t]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                            * 255
                        ).astype(np.uint8)
                    )
                    for t in range(T)
                ]
                real_frames = [
                    Image.fromarray(
                        (
                            real_videos[b, t]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                            * 255
                        ).astype(np.uint8)
                    )
                    for t in range(T)
                ]
                clip_scores.append(
                    float(compute_clip_score(gen_frames, real_frames, device))
                )
            metrics["clips"] = float(np.mean(clip_scores)) if clip_scores else math.nan
            _maybe_sync()
        except Exception as exc:
            _record_failure("clips", exc)

    return metrics


# -----------------------------------------------------------------------------
# Simple test (for sanity check)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # stable random video data
    B = 4  # batch
    T = 16  # frames
    C = 3
    H = 64
    W = 64

    gen_videos = torch.rand(B, T, C, H, W, device=device)
    gt_videos = torch.rand(B, T, C, H, W, device=device)

    print("===== Sanity Test: video_metric.py =====")
    print(f"PSNR: {compute_psnr(gen_videos, gt_videos):.4f}")
    print(f"SSIM: {compute_ssim(gen_videos, gt_videos):.4f}")
    print(f"LPIPS: {compute_lpips(gen_videos, gt_videos, device):.4f}")
    print(f"FID (frame-level): {compute_fid(gen_videos, gt_videos, device):.4f}")
    print(f"FVD (video-level): {compute_fvd(gen_videos, gt_videos, device):.4f}")

    # PIL frames for CLIP
    gen_frames = [
        Image.fromarray(
            (gen_videos[0, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        for t in range(T)
    ]
    gt_frames = [
        Image.fromarray(
            (gt_videos[0, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        for t in range(T)
    ]
    print(f"CLIP score: {compute_clip_score(gen_frames, gt_frames, device):.4f}")
    print("===== Sanity Test Finished =====")
