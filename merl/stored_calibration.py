"""Exact offline calibration pairs and recursive simulator history.

Observation o[t+1] follows executed action a[t]. The last historical observation
is the current anchor. Simulator callbacks receive past context and proposed
actions only; future observations and progress labels stay in the exporter.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from .imagination import Prediction
from .trust import CalibrationBatch, ChunkFeatures


@dataclass(frozen=True)
class StoredTrajectory:
    trajectory_id: str
    instruction: str
    observations: Tensor  # [T+1, ...], one explicit observation per transition
    actions: Tensor  # [T,A], executed coordinates (including gripper conversion)
    target_proxy: Tensor  # [T], target attached to the observation AFTER action
    split: str

    def validate(self):
        if not self.trajectory_id or not self.instruction.strip():
            raise ValueError("trajectory identity and instruction are required")
        if self.actions.ndim != 2 or min(self.actions.shape) < 1:
            raise ValueError("actions must be nonempty [T,A]")
        t = len(self.actions)
        if self.observations.ndim < 2 or len(self.observations) != t + 1:
            raise ValueError("a trajectory needs T+1 observations for T actions")
        if self.target_proxy.shape != (t,):
            raise ValueError("one explicit post-action target is required per action")
        for value in (self.observations, self.actions, self.target_proxy):
            if value.device != self.actions.device or not torch.isfinite(value).all():
                raise ValueError("trajectory tensors must be finite and on one device")
        if ((self.target_proxy < 0) | (self.target_proxy > 1)).any():
            raise ValueError("proxy targets must be in [0,1]")


@dataclass(frozen=True)
class PastContext:
    # Observations here are the outcomes of history_actions, including the anchor.
    observations: Tensor  # [H,...]
    actions: Tensor  # [H,A]
    instruction: str
    anchor_id: str

    @property
    def anchor(self):
        return self.observations[-1]

    def copy(self):
        return PastContext(self.observations.detach().clone(), self.actions.detach().clone(),
                           self.instruction, self.anchor_id)


class HistorySimulator:
    """Adapt a past-only simulator step to the recursive imagination callback.

    step(context, executed_actions) returns (predicted_observations, proxy_scores).
    encode(frames) is the same frozen deterministic [N,...] -> [N,D] encoder used
    by calibration. Returned denoising latents are deliberately not used as trust
    features: decoded predictions and grounded frames pass through one encoder.
    """

    def __init__(self, context: PastContext, step: Callable, encode: Callable):
        if (context.observations.ndim < 2 or context.actions.ndim != 2
                or len(context.observations) != len(context.actions) or not len(context.actions)):
            raise ValueError("history must contain aligned, nonempty transitions")
        if (not context.instruction.strip() or not context.anchor_id
                or not torch.isfinite(context.observations).all()
                or not torch.isfinite(context.actions).all()):
            raise ValueError("history must be finite and identified")
        self.context = context.copy()
        self.step = step
        self.encode = encode

    @torch.no_grad()
    def __call__(self, current: Tensor, actions: Tensor, instruction: str) -> Prediction:
        if instruction != self.context.instruction or not torch.equal(current, self.context.anchor):
            raise ValueError("simulator history does not match the current policy observation")
        if (actions.ndim != 2 or actions.shape[1] != self.context.actions.shape[1]
                or not len(actions) or not torch.isfinite(actions).all()):
            raise ValueError("future actions must be finite nonempty [C,A]")
        observations, proxy = self.step(self.context.copy(), actions.detach().clone())
        if (observations.shape != (len(actions), *current.shape)
                or proxy.shape != (len(actions),)):
            raise ValueError("simulator outputs must align with proposed transitions")
        if observations.dtype != current.dtype or observations.device != current.device:
            raise ValueError("predicted observations must preserve the anchor dtype and device")
        if not torch.isfinite(observations).all() or not torch.isfinite(proxy).all():
            raise ValueError("non-finite simulator output")
        if ((proxy < 0) | (proxy > 1)).any():
            raise ValueError("proxy predictions must be in [0,1]")
        latents = self.encode(observations.detach().clone()).detach()
        if latents.ndim != 2 or len(latents) != len(actions) or not torch.isfinite(latents).all():
            raise ValueError("encoder must return finite [C,D] latents")
        result = Prediction(observations.detach().clone(), latents, proxy.detach().clone())
        h = len(self.context.actions)
        self.context = PastContext(
            torch.cat((self.context.observations, observations.to(current.device)))[-h:].detach().clone(),
            torch.cat((self.context.actions, actions.to(self.context.actions.device)))[-h:].detach().clone(),
            instruction, self.context.anchor_id,
        )
        return result

    @torch.no_grad()
    def encode_anchor(self, observation):
        result = self.encode(observation[None].detach().clone()).detach()
        if result.ndim != 2 or len(result) != 1:
            raise ValueError("encoder must preserve the frame batch dimension")
        return result[0]


def context_at(trajectory: StoredTrajectory, start: int, history_size: int) -> PastContext:
    """Construct a stored anchor after start transitions, with no future access."""
    trajectory.validate()
    if history_size < 1 or start < history_size or start >= len(trajectory.actions):
        raise ValueError("window requires sufficient stored history and at least one future action")
    return PastContext(
        trajectory.observations[start - history_size + 1:start + 1].detach().clone(),
        trajectory.actions[start - history_size:start].detach().clone(),
        trajectory.instruction, f"{trajectory.trajectory_id}@transition:{start}",
    )


@torch.no_grad()
def build_calibration_batch(
    trajectories: Sequence[StoredTrajectory], windows: Sequence[tuple[int, int]], *,
    history_size: int, chunk_size: int, stage_context: Tensor,
    simulator_revision: str, step: Callable, encode: Callable,
) -> CalibrationBatch:
    """Replay only stored actions; windows are (trajectory index, start transition).

    Declare the whole-trajectory split before choosing windows. A history ending
    at o[start] predicts o[start+1:start+C+1]. A partial final window is padded and
    masked identically across features and labels. No live environment is queried.
    """
    if (not trajectories or not windows or chunk_size < 1 or stage_context.ndim != 1
            or not simulator_revision):
        raise ValueError("nonempty trajectories/windows, context and revision are required")
    ids = [item.trajectory_id for item in trajectories]
    if len(ids) != len(set(ids)):
        raise ValueError("trajectory IDs must be unique")
    # Reject held-out data before any simulator invocation or window construction.
    for item in trajectories:
        item.validate()
        if item.split != "calibration":
            raise ValueError("only pre-partitioned calibration trajectories are allowed")
    if len(windows) != len(set(windows)):
        raise ValueError("duplicate calibration windows are not allowed")
    contexts = []
    for index, start in windows:
        if not 0 <= index < len(trajectories):
            raise ValueError("trajectory index is out of bounds")
        contexts.append(context_at(trajectories[index], start, history_size))

    anchors, predictions, actions_all, masks = [], [], [], []
    grounded, proxies, targets, trajectory_ids, anchor_ids = [], [], [], [], []
    for (index, start), context in zip(windows, contexts):
        item = trajectories[index]
        actions = item.actions[start:start + chunk_size].detach().clone()
        count = len(actions)
        simulator = HistorySimulator(context, step, encode)
        prediction = simulator(context.anchor, actions, item.instruction)
        anchor = simulator.encode_anchor(context.anchor).cpu().float()
        truth = encode(item.observations[start + 1:start + count + 1].detach().clone()).detach().cpu().float()
        predicted = prediction.latents.cpu().float()
        if truth.shape != predicted.shape or truth.shape[1:] != anchor.shape:
            raise ValueError("grounded, predicted and anchor encoder dimensions must match")

        def pad(value):
            result = torch.zeros((chunk_size, *value.shape[1:]), dtype=torch.float32)
            result[:count] = value.detach().cpu().float()
            return result

        anchors.append(anchor)
        predictions.append(pad(predicted))
        grounded.append(pad(truth))
        actions_all.append(pad(actions))
        proxies.append(pad(prediction.proxy))
        targets.append(pad(item.target_proxy[start:start + count]))
        masks.append(torch.arange(chunk_size) < count)
        trajectory_ids.append(item.trajectory_id)
        anchor_ids.append(context.anchor_id)
    f = ChunkFeatures(torch.stack(anchors), torch.stack(predictions), torch.stack(actions_all),
                      torch.ones(len(windows), dtype=torch.long),
                      stage_context.detach().cpu().float()[None].repeat(len(windows), 1),
                      torch.stack(masks))
    result = CalibrationBatch(f, torch.stack(grounded), torch.stack(proxies), torch.stack(targets),
                              f.actions.clone(), tuple(trajectory_ids), tuple(anchor_ids),
                              tuple(anchor_ids), simulator_revision)
    result.residuals()  # Validate the complete pair before exposing it to fitting.
    return result
