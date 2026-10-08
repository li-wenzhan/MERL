"""Soft binary success-to-go supervision with an explicit transition mask."""

import torch
from torch.nn import functional as F


def soft_progress_loss(logits, target, valid):
    if logits.shape != (*target.shape, 2) or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError("expected logits [...,2], target [...], and boolean transition mask")
    if not valid.any() or not torch.isfinite(target[valid]).all():
        raise ValueError("proxy supervision requires finite valid targets")
    if ((target[valid] < 0) | (target[valid] > 1)).any():
        raise ValueError("success-to-go targets must be in [0,1]")
    selected = logits[valid].float()
    y = target[valid].float()
    return -(torch.stack((1 - y, y), -1) * F.log_softmax(selected, -1)).sum(-1).mean()
