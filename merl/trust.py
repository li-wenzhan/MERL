"""No-oracle residual estimation and the trust rules in paper Appendix A.

Only CalibrationBatch contains grounded futures. Inference accepts ChunkFeatures
and a simulator revision, so future observations cannot enter the scoring API.
The caller must produce exact pairs by replaying *stored* actions, never by
stepping an environment to label a policy-generated imagined trajectory.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor, nn


def _finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ChunkFeatures:
    """Inference-only features; N chunks, C steps, pooled visual dimension D.

    Latents must come from the same frozen visual encoder at calibration and
    inference. Spatially pool each latent frame before constructing this object.
    stage_context is a declared numeric context (e.g. normalized online stage).
    depth is one-based; valid_steps masks padding, including partial last chunks.
    """

    anchor_latent: Tensor  # [N, D], current grounded or imagined anchor
    imagined_latents: Tensor  # [N, C, D]
    actions: Tensor  # [N, C, A], executed-coordinate convention
    depth: Tensor  # [N]
    stage_context: Tensor  # [N, S]
    valid_steps: Tensor  # bool [N, C]

    def validate(self) -> None:
        if self.imagined_latents.ndim != 3:
            raise ValueError("imagined_latents must be [N, C, D]")
        n, c, d = self.imagined_latents.shape
        if min(n, c, d) < 1 or self.anchor_latent.shape != (n, d):
            raise ValueError("empty chunks or incompatible anchor_latent")
        if self.actions.ndim != 3 or self.actions.shape[:2] != (n, c):
            raise ValueError("actions must be [N, C, A]")
        if self.actions.shape[2] < 1:
            raise ValueError("action dimension must be positive")
        if self.depth.shape != (n,) or (self.depth < 1).any():
            raise ValueError("depth must be one-based [N]")
        if not torch.equal(self.depth, self.depth.round()):
            raise ValueError("depth must be integral")
        if self.stage_context.ndim != 2 or self.stage_context.shape[0] != n:
            raise ValueError("stage_context must be [N, S]")
        if self.valid_steps.shape != (n, c) or self.valid_steps.dtype != torch.bool:
            raise ValueError("valid_steps must be bool [N, C]")
        if not self.valid_steps.any(dim=1).all():
            raise ValueError("each chunk needs at least one valid step")
        # Partial chunks are prefixes, not a way to hide interior bad predictions.
        if (self.valid_steps[:, 1:] & ~self.valid_steps[:, :-1]).any():
            raise ValueError("valid_steps must be a contiguous prefix")
        for name in ("anchor_latent", "imagined_latents", "actions", "depth", "stage_context"):
            value = getattr(self, name)
            if value.device != self.anchor_latent.device:
                raise ValueError("all feature tensors must use the same device")
            _finite(name, value)
        if self.valid_steps.device != self.anchor_latent.device:
            raise ValueError("valid_steps must use the feature device")

    def matrix(self) -> Tensor:
        self.validate()
        mask = self.valid_steps[..., None]
        count = mask.sum(dim=1).float()
        future = self.imagined_latents.detach().float()
        action = self.actions.detach().float()
        mean = (future * mask).sum(dim=1) / count
        variance = ((future - mean[:, None]).square() * mask).sum(dim=1) / count
        # Flatten actions to preserve order; mask padding, retaining fixed C.
        return torch.cat((
            self.anchor_latent.detach().float(), mean, variance,
            (action * mask).flatten(1), self.valid_steps.float(),
            self.depth.detach().float()[:, None], self.stage_context.detach().float(),
        ), dim=1)

    @property
    def signature(self) -> Tuple[int, int, int, int]:
        self.validate()
        return (self.imagined_latents.shape[1], self.imagined_latents.shape[2],
                self.actions.shape[2], self.stage_context.shape[1])


@dataclass(frozen=True)
class CalibrationBatch:
    features: ChunkFeatures
    grounded_latents: Tensor  # [N, C, D], same encoder as imagined latents
    predicted_proxy: Tensor  # [N, C], on imagined observation/action
    target_proxy: Tensor  # [N, C], stored success-to-go labels
    grounded_actions: Tensor  # [N, C, A]
    trajectory_ids: Tuple[str, ...]
    anchor_ids: Tuple[str, ...]
    predicted_anchor_ids: Tuple[str, ...]
    simulator_revision: str  # identifies BOTH video predictor and reward proxy
    split: str = "calibration"

    def residuals(self) -> Tensor:
        """Eq. 20, with uniform normalized temporal weights (masked mean)."""
        f = self.features
        f.validate()
        n, c, _ = f.imagined_latents.shape
        if self.split != "calibration":
            raise ValueError("held-out/evaluation futures cannot train the estimator")
        if not self.simulator_revision:
            raise ValueError("a simulator revision is required")
        if any(len(ids) != n or any(not item for item in ids) for ids in
               (self.trajectory_ids, self.anchor_ids, self.predicted_anchor_ids)):
            raise ValueError("each window needs trajectory and anchor provenance")
        if self.anchor_ids != self.predicted_anchor_ids:
            raise ValueError("grounded and predicted windows have different anchors")
        # Stored action sequences can be recursively replayed through predicted
        # observations. Their recorded futures remain labels, never features.
        # Whole-trajectory evaluation splits are excluded above at every depth.
        if self.grounded_latents.shape != f.imagined_latents.shape:
            raise ValueError("grounded/predicted latent shapes differ")
        if self.grounded_actions.shape != f.actions.shape:
            raise ValueError("grounded/predicted action shapes differ")
        for name in ("grounded_latents", "predicted_proxy", "target_proxy", "grounded_actions"):
            value = getattr(self, name)
            if value.device != f.anchor_latent.device:
                raise ValueError("calibration tensors must use the feature device")
            _finite(name, value)
        if not torch.equal(self.grounded_actions[f.valid_steps], f.actions[f.valid_steps]):
            raise ValueError("exact pairs require identical action continuations")
        for name in ("predicted_proxy", "target_proxy"):
            value = getattr(self, name)
            if value.shape != (n, c) or ((value < 0) | (value > 1)).any():
                raise ValueError(f"{name} must be bounded [N, C] progress scores")
        mask = f.valid_steps.float()
        obs = (f.imagined_latents.detach().float() - self.grounded_latents.detach().float()).square().mean(-1)
        proxy = (self.predicted_proxy.detach().float() - self.target_proxy.detach().float()).abs()
        return torch.stack(((obs * mask).sum(1), (proxy * mask).sum(1)), dim=1) / mask.sum(1)[:, None]


def success_to_go(success: Tensor, length: Tensor, times: Tensor,
                  gamma: float = 0.99, horizon: int = 32) -> Tensor:
    """Eq. 9 at explicit zero-based grounded timestamps; no implicit alignment."""
    if not 0 < gamma <= 1 or horizon < 1:
        raise ValueError("invalid success-to-go discount or horizon")
    if success.ndim != 1 or length.shape != success.shape or times.ndim != 2 or times.shape[0] != len(success):
        raise ValueError("expected success/length [N] and times [N, C]")
    for name, value in (("success", success), ("length", length), ("times", times)):
        _finite(name, value)
    if not ((success == 0) | (success == 1)).all():
        raise ValueError("success must be binary")
    if not torch.equal(length, length.round()) or not torch.equal(times, times.round()):
        raise ValueError("length and timestamps must be integral")
    if (length < 1).any() or (times < 0).any() or (times >= length[:, None]).any():
        raise ValueError("timestamps must refer to stored trajectory steps")
    distance = (length[:, None] - 1 - times).clamp(max=horizon - 1)
    return success.float()[:, None] * gamma ** distance.float()


@dataclass(frozen=True)
class TrustConfig:
    alpha_obs: float = 1.0
    alpha_proxy: float = 1.0
    priority_epsilon: float = 1e-3
    priority_exponent: float = 1.0
    weight_eta: float = 1.0
    weight_min: float = 0.05

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("trust parameters must be finite")
        if min(self.alpha_obs, self.alpha_proxy, self.priority_exponent, self.weight_eta) < 0:
            raise ValueError("trust scales and exponents must be nonnegative")
        if self.alpha_obs + self.alpha_proxy == 0 or self.priority_epsilon <= 0 or not 0 < self.weight_min <= 1:
            raise ValueError("invalid trust weights or priority epsilon")


@dataclass(frozen=True)
class TrustScores:
    residuals: Tensor
    error: Tensor
    probability: Tensor
    weight: Tensor


def trust_scores(residuals: Tensor, config: TrustConfig = TrustConfig()) -> TrustScores:
    """Eqs. 19, 21--23 over a candidate CHUNK population, without IS correction."""
    if residuals.ndim != 2 or residuals.shape[1] != 2 or len(residuals) == 0:
        raise ValueError("residuals must be nonempty [N, 2]")
    residuals = residuals.detach().float()
    _finite("residuals", residuals)
    if (residuals < 0).any():
        raise ValueError("residuals must be nonnegative")
    error = config.alpha_obs * residuals[:, 0] + config.alpha_proxy * residuals[:, 1]
    _finite("combined error", error)
    log_priority = -config.priority_exponent * torch.log(error.double() + config.priority_epsilon)
    probability = torch.softmax(log_priority, dim=0)
    weight = torch.exp(-config.weight_eta * error).clamp(config.weight_min, 1.0)
    return TrustScores(residuals, error, probability, weight)


@dataclass
class StageScheduler:
    """Eqs. 15--18; call once after calibration, before candidate generation."""
    beta: float = 0.8
    reference_error: float = 1.0
    kappa: float = 1.0
    ratio_min: float = 0.05
    ratio_max: float = 0.95
    horizon_min: int = 8
    horizon_max: int = 32
    smoothed_error: float = None

    def update(self, measured_error: float) -> Tuple[float, int]:
        values = (measured_error, self.beta, self.reference_error, self.kappa,
                  self.ratio_min, self.ratio_max, self.horizon_min, self.horizon_max)
        if not all(math.isfinite(v) for v in values) or measured_error < 0:
            raise ValueError("scheduler inputs must be finite and error nonnegative")
        if not 0 <= self.beta < 1 or self.reference_error <= 0 or self.kappa <= 0:
            raise ValueError("invalid scheduler smoothing or confidence parameters")
        if not 0 <= self.ratio_min <= self.ratio_max <= 1 or not 1 <= self.horizon_min <= self.horizon_max:
            raise ValueError("invalid ratio or horizon bounds")
        if self.smoothed_error is not None and (not math.isfinite(self.smoothed_error) or self.smoothed_error < 0):
            raise ValueError("invalid restored scheduler state")
        self.smoothed_error = (measured_error if self.smoothed_error is None else
                               self.beta * self.smoothed_error + (1 - self.beta) * measured_error)
        # Stable inverse-power sigmoid, including zero error and large exponents.
        log_odds = (-math.inf if self.smoothed_error == 0 else
                    self.kappa * (math.log(self.smoothed_error) - math.log(self.reference_error)))
        confidence = (1.0 / (1.0 + math.exp(log_odds)) if log_odds <= 0 else
                      math.exp(-log_odds) / (1.0 + math.exp(-log_odds)))
        return (self.ratio_min + (self.ratio_max - self.ratio_min) * confidence,
                math.floor(self.horizon_min + (self.horizon_max - self.horizon_min) * confidence))


class ResidualPredictor:
    """A small two-output MLP, re-fit atomically after each simulator update.

    Fit is deterministic on CPU and does not alter policy RNG streams. Published
    snapshots are frozen. This is a new implementation of the stated mechanism,
    not a recovered checkpoint or evidence of the paper's reported accuracy.
    """

    def __init__(self, hidden_dim: int = 64, seed: int = 0):
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.model = None
        self.revision = None
        self.stage = None
        self.signature = None

    def _network(self, dimension: int) -> nn.Module:
        return nn.Sequential(nn.Linear(dimension, self.hidden_dim), nn.SiLU(),
                             nn.Linear(self.hidden_dim, 2), nn.Softplus())

    def fit(self, batch: CalibrationBatch, *, stage: int, steps: int = 200,
            learning_rate: float = 3e-3) -> dict:
        if stage < 0 or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("invalid fit configuration")
        targets = batch.residuals().cpu()
        features = batch.features.matrix().cpu()
        if len(features) < 2:
            raise ValueError("at least two calibration windows are required")
        mean = features.mean(0)
        scale = features.std(0, unbiased=False)
        # Constant context (e.g. depth=1) must not amplify an unseen depth by
        # 1000x through an epsilon denominator. This does not solve depth OOD.
        scale = torch.where(scale < 1e-3, torch.ones_like(scale), scale)
        target_scale = targets.mean(0).clamp_min(1e-3)
        x = (features - mean) / scale
        y = targets / target_scale
        # Only a completed fit replaces the last valid frozen snapshot.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            model = self._network(features.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite residual fit; snapshot was not replaced")
                loss.backward()
                optimizer.step()
        model.eval().requires_grad_(False)
        with torch.no_grad():
            mae = (model(x) * target_scale - targets).abs().mean(0)
        _finite("fitted residuals", mae)
        self.model = model
        self.mean, self.scale, self.target_scale = mean, scale, target_scale
        self.revision, self.stage = batch.simulator_revision, stage
        self.signature = batch.features.signature
        self.trajectory_ids = tuple(sorted(set(batch.trajectory_ids)))
        return {"windows": len(features), "trajectories": len(self.trajectory_ids),
                "train_obs_mae": float(mae[0]), "train_proxy_mae": float(mae[1]),
                "stage": stage, "simulator_revision": self.revision}

    @torch.no_grad()
    def predict(self, features: ChunkFeatures, *, simulator_revision: str, stage: int) -> Tensor:
        if self.model is None:
            raise RuntimeError("calibrate before imagination; an untrained predictor is not trust")
        if simulator_revision != self.revision or stage != self.stage:
            raise ValueError("stale residual estimator: recalibrate after simulator/stage changes")
        if features.signature != self.signature:
            raise ValueError("feature encoder/action/context signature changed")
        x = (features.matrix().cpu() - self.mean) / self.scale
        self.model.eval()
        result = self.model(x) * self.target_scale
        _finite("predicted residuals", result)
        return result.to(features.anchor_latent.device)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("cannot save an uncalibrated predictor")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format_version=1, hidden_dim=self.hidden_dim, seed=self.seed,
                       model=self.model.state_dict(), mean=self.mean, scale=self.scale,
                       target_scale=self.target_scale, revision=self.revision, stage=self.stage,
                       signature=self.signature, trajectory_ids=self.trajectory_ids)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["format_version"] != 1:
            raise ValueError("unsupported residual checkpoint format")
        result = cls(data["hidden_dim"], data["seed"])
        with torch.random.fork_rng(devices=[]):
            result.model = result._network(len(data["mean"]))
        result.model.load_state_dict(data["model"], strict=True)
        result.model.eval().requires_grad_(False)
        for key in ("mean", "scale", "target_scale", "revision", "stage", "signature", "trajectory_ids"):
            setattr(result, key, data[key])
        return result
