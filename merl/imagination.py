"""Recursive imagination boundary: there is deliberately no environment handle.

Production adapters supply a policy, a frozen simulator and an encoder. The core
keeps policy queries on predicted observations and returns chunk-level trust.
"""

from dataclasses import dataclass
from typing import Callable, List

import torch
from torch import Tensor

from .trust import ChunkFeatures, ResidualPredictor, TrustConfig, TrustScores, trust_scores


@dataclass(frozen=True)
class Prediction:
    observations: Tensor  # [C, ...], decoded frames in the policy input convention
    latents: Tensor  # [C, D], frozen encoder convention used for calibration
    proxy: Tensor  # [C]


@dataclass(frozen=True)
class ImaginedChunk:
    anchor: Tensor
    actions: Tensor
    prediction: Prediction
    features: ChunkFeatures
    residuals: Tensor


@torch.no_grad()
def imagine(
    *, anchor: Tensor, instruction: str, horizon: int, chunk_size: int,
    policy: Callable[[Tensor, str], Tensor],
    simulator: Callable[[Tensor, Tensor, str], Prediction],
    encode_anchor: Callable[[Tensor], Tensor],
    predictor: ResidualPredictor, stage_context: Tensor,
    simulator_revision: str, stage: int,
) -> List[ImaginedChunk]:
    """Generate local recursive chunks; the last chunk is explicitly masked.

    The simulator adapter is responsible for its historical conditioning and RNG;
    those states must start from the stored grounded anchor, never live env reads.
    The core calls no simulator after reaching the requested environment-step cap.
    """
    if horizon < 1 or chunk_size < 1 or stage_context.ndim != 1:
        raise ValueError("positive horizon/chunk size and vector stage context required")
    if predictor.model is None or predictor.revision != simulator_revision or predictor.stage != stage:
        raise ValueError("a current calibrated predictor is required before rollout")
    chunks = []
    current = anchor.detach().clone()
    elapsed = 0
    while elapsed < horizon:
        actions = policy(current.clone(), instruction).detach()
        if actions.ndim != 2 or actions.shape[0] != chunk_size or not torch.isfinite(actions).all():
            raise ValueError("policy must return finite actions [C, A]")
        count = min(chunk_size, horizon - elapsed)
        prediction = simulator(current.clone(), actions[:count].clone(), instruction)
        if (prediction.observations.shape[0] != count or prediction.latents.ndim != 2
                or prediction.latents.shape[0] != count or prediction.proxy.shape != (count,)):
            raise ValueError("simulator outputs must align with requested actions")
        for value in (prediction.observations, prediction.latents, prediction.proxy):
            if not torch.isfinite(value).all():
                raise ValueError("non-finite simulator output")
        if ((prediction.proxy < 0) | (prediction.proxy > 1)).any():
            raise ValueError("simulator proxy must be in [0, 1]")
        latent = encode_anchor(current.clone()).detach()
        future = prediction.latents.detach()
        device = latent.device
        valid = torch.arange(chunk_size, device=device)[None, :] < count
        padded = torch.zeros(chunk_size, future.shape[-1], device=device)
        padded[:count] = future.to(device)
        padded_actions = actions.to(device).clone()
        padded_actions[count:] = 0
        features = ChunkFeatures(
            latent[None], padded[None], padded_actions[None],
            torch.tensor([len(chunks) + 1], device=device),
            stage_context.detach().to(device)[None], valid,
        )
        residuals = predictor.predict(features, simulator_revision=simulator_revision, stage=stage)
        chunks.append(ImaginedChunk(current.clone(), actions[:count].clone(), prediction, features, residuals))
        current = prediction.observations[-1].detach().clone()
        elapsed += count
    return chunks


def score_chunks(chunks: List[ImaginedChunk], config: TrustConfig = TrustConfig()) -> TrustScores:
    if not chunks:
        raise ValueError("cannot score an empty imagined population")
    return trust_scores(torch.cat([chunk.residuals for chunk in chunks]), config)


def sample_chunks(scores: TrustScores, count: int, *, generator: torch.Generator) -> Tensor:
    """Eq. 22, draws with replacement; no uniform-replay importance correction."""
    if count < 1:
        raise ValueError("sample count must be positive")
    return torch.multinomial(scores.probability.cpu(), count, replacement=True, generator=generator)


def mixed_policy_loss(real_chunk_loss: Tensor, imagined_chunk_loss: Tensor,
                      imagined_weights: Tensor, ratio: float) -> Tensor:
    """Eqs. 24, 27, 28; inputs are clipped loss sums for sampled valid chunks.

    Advantages must already be computed within the appropriate GRPO groups.
    Dividing by sum(weights) would cancel trust suppression and is incorrect.
    """
    if not 0 <= ratio <= 1:
        raise ValueError("ratio must be in [0, 1]")
    if real_chunk_loss.ndim != 1 or imagined_chunk_loss.ndim != 1 or imagined_weights.shape != imagined_chunk_loss.shape:
        raise ValueError("losses and imagined weights must be aligned chunk vectors")
    if (ratio < 1 and not real_chunk_loss.numel()) or (ratio > 0 and not imagined_chunk_loss.numel()):
        raise ValueError("nonzero mixture branches require valid chunks")
    for value in (real_chunk_loss, imagined_chunk_loss, imagined_weights):
        if not torch.isfinite(value).all():
            raise ValueError("losses and weights must be finite")
    if ((imagined_weights < 0) | (imagined_weights > 1)).any():
        raise ValueError("trust weights must be in [0, 1]")
    real = real_chunk_loss.sum() / max(real_chunk_loss.numel(), 1)
    imagined = (imagined_chunk_loss * imagined_weights.detach()).sum() / max(imagined_chunk_loss.numel(), 1)
    return (1 - ratio) * real + ratio * imagined
