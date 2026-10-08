"""Camera-ready contracts shared by the driver, actor and simulator worker.

This module has no Ray, simulator, or model-loading imports. Observation o[t]
precedes executed action u[t]; o[t+1] is its outcome. Categorical policy tokens
are retained independently of the continuous commands sent to the environment.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .stored_calibration import StoredTrajectory
from .trust import success_to_go


@dataclass(frozen=True)
class PaperConfig:
    seed: int = 0
    grounded_trajectories: int = 6
    grounded_step_cap: int = 512
    chunk_size: int = 8
    history_size: int = 8
    simulator_steps: int = 50
    proxy_discount: float = 0.99
    proxy_horizon: int = 32
    proxy_loss_weight: float = 1.0
    calibration_windows_per_trajectory: int = 1
    calibration_depth: int = 4
    residual_hidden_dim: int = 64
    residual_fit_steps: int = 200
    residual_learning_rate: float = 0.003
    error_beta: float = 0.8
    reference_error: float = 1.0
    scheduler_kappa: float = 1.0
    ratio_min: float = 0.05
    ratio_max: float = 0.95
    horizon_min: int = 8
    horizon_max: int = 32
    fixed_ratio: float = 0.5
    fixed_horizon: int = 16
    alpha_obs: float = 1.0
    alpha_proxy: float = 1.0
    priority_epsilon: float = 0.001
    priority_exponent: float = 1.0
    weight_eta: float = 1.0
    weight_min: float = 0.05
    imagined_group_size: int = 4
    real_chunks_per_update: int = 48
    imagined_chunks_per_update: int = 48
    stage_trust: bool = True
    chunk_trust: bool = True

    def __post_init__(self):
        integer_positive = ("grounded_trajectories", "grounded_step_cap", "chunk_size", "history_size",
                            "simulator_steps", "proxy_horizon", "calibration_windows_per_trajectory",
                            "calibration_depth", "residual_hidden_dim", "residual_fit_steps",
                            "horizon_min", "horizon_max", "fixed_horizon", "imagined_group_size",
                            "real_chunks_per_update", "imagined_chunks_per_update")
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in integer_positive):
            raise ValueError("paper counts and horizons must be positive integers")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("paper seed must be a nonnegative integer")
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("paper configuration must be finite")
        if not (self.chunk_size <= self.horizon_min <= self.horizon_max <= 4 * self.chunk_size):
            raise ValueError("paper imagination must remain within one to four action chunks")
        if not self.chunk_size <= self.fixed_horizon <= 4 * self.chunk_size:
            raise ValueError("fixed baseline horizon must remain within one to four chunks")
        if not 0 <= self.ratio_min <= self.ratio_max <= 1 or not 0 <= self.fixed_ratio <= 1:
            raise ValueError("invalid paper mixture ratios")
        if not 0 < self.proxy_discount <= 1 or self.proxy_loss_weight < 0:
            raise ValueError("invalid success-to-go/proxy loss configuration")
        if not 0 <= self.error_beta < 1 or min(self.reference_error, self.scheduler_kappa,
                                              self.residual_learning_rate, self.priority_epsilon) <= 0:
            raise ValueError("invalid scheduler or residual fitting parameters")
        if min(self.alpha_obs, self.alpha_proxy, self.priority_exponent, self.weight_eta) < 0:
            raise ValueError("trust coefficients must be nonnegative")
        if self.alpha_obs + self.alpha_proxy <= 0 or not 0 < self.weight_min <= 1:
            raise ValueError("invalid trust aggregation/weight floor")
        if self.grounded_step_cap > 512 or self.imagined_group_size < 2 or self.calibration_depth > 4:
            raise ValueError("grounded cap is at most 512; GRPO needs at least two candidates")

    @classmethod
    def from_dict(cls, config):
        return cls(**dict(config))

    def for_mode(self, mode):
        from dataclasses import replace
        if mode in ("MBRL", "ONLINE_MBRL", "MFRL"):
            return replace(self, stage_trust=False, chunk_trust=False)
        if mode in ("MERL", "STATIC_TRUST"):
            return self
        raise ValueError(f"unsupported camera-ready control: {mode}")


def save_grounded_trajectory(path, *, observations, executed_actions, instruction,
                             success, task_id, trial_id, stage):
    """Atomic, non-pickled trajectory export. Invalid episodes are rejected."""
    path = Path(path)
    observations = np.asarray(observations)
    actions = np.asarray(executed_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not 0 < len(actions) <= 512:
        raise ValueError("grounded trajectory requires 1..512 executed 7D commands")
    if (observations.dtype != np.uint8 or observations.ndim != 4
            or len(observations) != len(actions) + 1 or observations.shape[-1] != 3):
        raise ValueError("grounded export requires T+1 RGB observations for T actions")
    if not np.isfinite(actions).all() or not instruction.strip():
        raise ValueError("grounded actions and instruction must be valid")
    metadata = dict(trajectory_id=f"stage:{stage}/task:{task_id}/trial:{trial_id}/{path.stem}",
                    instruction=instruction, success=bool(success), task_id=int(task_id),
                    trial_id=int(trial_id), stage=int(stage), split="calibration",
                    action_convention="libero_executed_xyz_axis_angle_gripper",
                    observation_alignment="o[t] before u[t]; o[t+1] after u[t]")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, observations=observations, actions=actions,
                            metadata=np.asarray(json.dumps(metadata)))
    temporary.replace(path)
    return str(path.resolve())


def load_grounded_trajectory(path, config: PaperConfig):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        observations = torch.from_numpy(data["observations"].copy())
        actions = torch.from_numpy(data["actions"].copy())
    if metadata["split"] != "calibration":
        raise ValueError("held-out/evaluation trajectories cannot enter online calibration")
    t = len(actions)
    target = success_to_go(torch.tensor([int(metadata["success"])]), torch.tensor([t]),
                           torch.arange(t)[None], config.proxy_discount, config.proxy_horizon)[0]
    item = StoredTrajectory(metadata["trajectory_id"], metadata["instruction"],
                            observations, actions, target, metadata["split"])
    item.validate()
    return item


def grouped_advantages(scores: Tensor, groups) -> Tensor:
    """GRPO statistics are computed before trust sampling, within each branch."""
    if scores.ndim != 1 or len(groups) != len(scores) or not torch.isfinite(scores).all():
        raise ValueError("finite group-aligned scalar scores required")
    result = torch.zeros_like(scores, dtype=torch.float32)
    for key in sorted(set(groups)):
        indices = [i for i, group in enumerate(groups) if group == key]
        if len(indices) < 2:
            raise ValueError("each GRPO group requires at least two candidates")
        selected = scores[indices].float()
        result[indices] = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-6)
    return result.detach()


def clipped_chunk_loss(new_logp: Tensor, old_logp: Tensor, advantages: Tensor,
                       valid_tokens: Tensor, clip_low: float, clip_high: float) -> Tensor:
    """Paper Eq. 26: SUM categorical-token terms in each valid action chunk."""
    if new_logp.ndim != 2 or any(x.shape != new_logp.shape for x in (old_logp, advantages, valid_tokens)):
        raise ValueError("log probabilities, advantages and masks must be [N,L]")
    if valid_tokens.dtype != torch.bool or min(clip_low, clip_high) < 0:
        raise ValueError("a boolean action-token mask and nonnegative clipping are required")
    for value in (new_logp, old_logp, advantages):
        if not torch.isfinite(value[valid_tokens]).all():
            raise FloatingPointError("non-finite valid policy term")
    # Replace masked terms before exp: NaN padding cannot poison gradients.
    difference = torch.where(valid_tokens, new_logp - old_logp.detach(), 0.)
    ratio = difference.exp()
    if not torch.isfinite(ratio[valid_tokens]).all():
        raise FloatingPointError("categorical policy ratio overflow")
    advantage = torch.where(valid_tokens, advantages.detach(), 0.)
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    return -torch.minimum(ratio * advantage, clipped * advantage).sum(-1)


def branch_coefficients(is_imagined: Tensor, weights: Tensor, ratio: float,
                        real_count: int, imagined_count: int) -> Tensor:
    """Independent branch normalization; never normalize by summed trust."""
    if is_imagined.dtype != torch.bool or is_imagined.shape != weights.shape:
        raise ValueError("branch labels and weights must be aligned")
    if not 0 <= ratio <= 1 or (ratio < 1 and real_count < 1) or (ratio > 0 and imagined_count < 1):
        raise ValueError("nonzero mixture branches need valid chunk counts")
    if not torch.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
        raise ValueError("trust weights must be finite in [0,1]")
    return torch.where(is_imagined, ratio * weights.detach() / max(imagined_count, 1),
                       torch.full_like(weights, (1 - ratio) / max(real_count, 1))).detach()
