"""Small action-chunk/token boundary helpers shared with policy rollout."""

import torch
from torch import Tensor


def valid_response_tokens(responses: Tensor, finish_steps: Tensor,
                          chunk_size: int, dummy_chunks: Tensor = None) -> Tensor:
    """Convert environment-step counts to a contiguous valid action-token prefix.

    `responses` is [B, policy_calls, C*A]. `finish_steps` is in environment
    steps, not policy calls. A placeholder invalidates its chunk and the suffix:
    a single prefix length cannot represent holes without admitting dummy tokens.
    """
    if responses.ndim != 3 or chunk_size < 1 or responses.shape[2] % chunk_size:
        raise ValueError("responses must be [B, calls, C*A] with an integral token/action ratio")
    b, calls, per_chunk = responses.shape
    if per_chunk < 1 or finish_steps.numel() != b:
        raise ValueError("invalid response dimensions or finish-step batch")
    steps = finish_steps.reshape(b).to(responses.device)
    if not torch.isfinite(steps).all() or not torch.equal(steps, steps.round()) or (steps < 0).any():
        raise ValueError("finish_steps must be finite nonnegative integers")
    valid_calls = torch.full((b,), calls, dtype=torch.long, device=responses.device)
    if dummy_chunks is not None:
        if dummy_chunks.shape != (b, calls):
            raise ValueError("dummy mask must align with policy calls")
        dummy = dummy_chunks.to(device=responses.device, dtype=torch.bool)
        indices = torch.arange(calls, device=responses.device).expand(b, calls)
        valid_calls = torch.where(dummy, indices, calls).amin(dim=1)
    available_steps = valid_calls * chunk_size
    return torch.minimum(steps.long(), available_steps) * (per_chunk // chunk_size)
