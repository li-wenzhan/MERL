# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface

MERL memory patch:
1. Keep replay pools bounded and store cloned PPO-ready tensors only.
2. Keep rollout videos on workers / shard files instead of Ray driver pools.
3. Preserve MERL/MBRL/MFRL batch contracts for mixed PPO updates.
"""

import copy
import gc
import glob
import json
import os
import shutil
import statistics
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pprint import pprint
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
import torch.distributed as dist

import numpy as np
import ray
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.dataset.rob_dataset import BufferedDataLoader
from verl.workers.fsdp_workers import (
    FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT,
    FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT,
    HF_FULL_STATE_DICT_CHECKPOINT_FORMAT,
    WorldModelTrainer,
)
from verl.trainer.ppo.priority_pool import (
    PrioritizedPool,
    prioritized_mixed_sample,
    compute_priorities_from_dataproto_list,
)
from verl.trainer.ppo.dataproto_filter import DataProtoFilter

WorkerType = Type[Worker]


PPO_REPLAY_BATCH_KEYS = {
    "task_id",
    "trial_id",
    "trial_seed",
    "responses",
    "input_ids",
    "attention_mask",
    "position_ids",
    "pixel_values",
    "proprio",
    "is_dummy",
    "complete",
    "env_complete",
    "wm_proxy_complete",
    "wm_proxy_score",
    "anchor_reward",
    "has_anchor_reward",
    "finish_step",
    "action",
    "rm_scores",
    "gt_scores",
    "token_level_scores",
    "token_level_rewards",
    "old_log_probs",
    "ref_log_prob",
    "advantages",
    "returns",
    "is_weight",
    "is_wm",
    "is_weight_token",
    "is_wm_token_mask",
    "acc",
    "format_correctness",
    "wm_pred_valid",
    "wm_obs_error",
    "wm_done_error",
    "valid_response_tokens",
    "wm_valid_response_tokens",
    "wm_dummy_step_count",
    "wm_placeholder_step_count",
}

ROLLOUT_MEDIA_BATCH_KEYS = {"video", "env_video", "env_dones"}
ROLLOUT_PROMPT_BATCH_KEYS = {"task_id", "trial_id", "trial_seed"}
ROLLOUT_PROMPT_NON_TENSOR_KEYS = {"task_suite_name", "uid", "data_source"}
ROLLOUT_VOLATILE_META_KEYS = {"task_descriptions"}


def clone_dataproto_for_replay(dp: DataProto) -> DataProto:
    """Clone a bounded PPO/prompt-ready DataProto without rollout media storage."""
    if dp is None or len(dp) == 0:
        return dp

    tensors = {}
    for key, value in dp.batch.items():
        if key in ROLLOUT_MEDIA_BATCH_KEYS:
            continue
        if key not in PPO_REPLAY_BATCH_KEYS:
            continue
        if isinstance(value, torch.Tensor):
            tensors[key] = value.detach().cpu().clone().contiguous()

    if len(tensors) == 0:
        raise RuntimeError(
            "Cannot build replay DataProto: no PPO replay tensor keys were found."
        )

    non_tensors = {}
    for key, value in dp.non_tensor_batch.items():
        if isinstance(value, np.ndarray) and len(value) == len(dp):
            non_tensors[key] = np.asarray(value, dtype=object).copy()

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=copy.deepcopy(dp.meta_info),
    )


def inject_actor_token_contract(
    dp: DataProto,
    *,
    action_token_len: int,
    strict: bool = False,
    require_wm_anchor_reward: bool = False,
    context: str = "actor_input",
) -> Dict[str, float]:
    """Attach token-level real/imagined PPO masks consumed by the actor."""
    empty_stats = {
        "sample_count": 0.0,
        "imag_sample_count": 0.0,
        "imag_valid_sample_count": 0.0,
        "imag_token_count": 0.0,
        "real_token_count": 0.0,
        "imag_weight_mean": 0.0,
        "imag_anchor_coverage": 0.0,
        "imag_unanchored_sample_count": 0.0,
        "response_token_len": 0.0,
    }
    if dp is None or len(dp) == 0 or getattr(dp, "batch", None) is None:
        return empty_stats

    batch = dp.batch
    if "responses" not in batch or "finish_step" not in batch:
        if strict:
            raise RuntimeError(
                f"[WM CONTRACT] {context} is missing responses/finish_step."
            )
        return empty_stats

    B = len(dp)
    device = next(iter(batch.values())).device
    responses = batch["responses"]
    response_token_len = int(responses[0].numel()) if B > 0 else 0

    def _read_sample_field(
        key: str,
        *,
        default: float,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if key not in batch:
            return torch.full((B,), default, device=device, dtype=dtype)
        value = batch[key]
        if value.dim() > 1:
            value = value.reshape(B, -1)[:, 0]
        return value.to(device=device, dtype=dtype)

    finish_step = _read_sample_field("finish_step", default=0.0, dtype=torch.long)
    actor_valid_tokens = finish_step.clamp_min(0) * max(1, int(action_token_len))
    actor_valid_tokens = actor_valid_tokens.clamp(max=max(response_token_len, 0))

    token_valid_counts = DataProtoFilter._per_sample_valid_response_tokens(
        batch=batch,
        B=B,
        device=device,
        action_token_len=action_token_len,
    )
    valid_tokens = torch.minimum(actor_valid_tokens, token_valid_counts)

    if "is_dummy" in batch:
        has_real_step = DataProtoFilter._per_sample_has_real_step(
            is_dummy=batch["is_dummy"],
            B=B,
            device=device,
        )
        valid_tokens = torch.where(
            has_real_step,
            valid_tokens,
            torch.zeros_like(valid_tokens),
        )

    is_wm_sample = _read_sample_field("is_wm", default=0.0) > 0.5
    is_weight_sample = _read_sample_field("is_weight", default=1.0).clamp_min(0.0)
    has_anchor_reward = (
        _read_sample_field(
            "has_anchor_reward",
            default=0.0 if require_wm_anchor_reward else 1.0,
        )
        > 0.5
    )
    if require_wm_anchor_reward:
        is_weight_sample = torch.where(
            is_wm_sample & (~has_anchor_reward),
            torch.zeros_like(is_weight_sample),
            is_weight_sample,
        )
    batch["is_wm"] = is_wm_sample.to(dtype=torch.float32).contiguous()
    batch["is_weight"] = is_weight_sample.to(dtype=torch.float32).contiguous()

    if "wm_pred_valid" in batch:
        wm_pred_valid = _read_sample_field(
            "wm_pred_valid", default=1.0, dtype=torch.float32
        )
        valid_tokens = torch.where(
            (~is_wm_sample) | (wm_pred_valid > 0.5),
            valid_tokens,
            torch.zeros_like(valid_tokens),
        )

    steps = torch.arange(response_token_len, device=device).unsqueeze(0)
    response_mask = steps < valid_tokens.unsqueeze(1)
    is_wm_token_mask = response_mask & is_wm_sample.unsqueeze(1)
    is_weight_token = is_weight_sample.unsqueeze(1).expand(-1, response_token_len)
    is_weight_token = torch.where(
        response_mask,
        is_weight_token,
        torch.zeros_like(is_weight_token),
    ).contiguous()

    batch["is_wm_token_mask"] = is_wm_token_mask.contiguous()
    batch["is_weight_token"] = is_weight_token.to(dtype=torch.float32).contiguous()

    imag_weights = batch["is_weight_token"][is_wm_token_mask]
    imag_token_count = int(is_wm_token_mask.sum().detach().item())
    real_token_count = int((response_mask & (~is_wm_sample.unsqueeze(1))).sum().item())
    imag_valid_sample_count = int((is_wm_sample & (valid_tokens > 0)).sum().item())
    imag_sample_count = int(is_wm_sample.sum().detach().item())
    imag_anchor_count = int((is_wm_sample & has_anchor_reward).sum().detach().item())
    imag_weight_mean = (
        float(imag_weights.float().mean().detach().item())
        if imag_weights.numel() > 0
        else 0.0
    )

    return {
        "sample_count": float(B),
        "imag_sample_count": float(imag_sample_count),
        "imag_valid_sample_count": float(imag_valid_sample_count),
        "imag_token_count": float(imag_token_count),
        "real_token_count": float(real_token_count),
        "imag_weight_mean": float(imag_weight_mean),
        "imag_anchor_coverage": (
            float(imag_anchor_count) / float(imag_sample_count)
            if imag_sample_count > 0
            else 0.0
        ),
        "imag_unanchored_sample_count": float(
            max(0, imag_sample_count - imag_anchor_count)
        ),
        "response_token_len": float(response_token_len),
    }


def union_prompt_and_rollout_output(
    prompt_batch: DataProto,
    rollout_output: DataProto,
    *,
    context: str = "rollout",
) -> DataProto:
    """Merge rollout tensors while keeping prompt-side identity fields canonical."""
    if rollout_output is None:
        return prompt_batch
    if prompt_batch is None:
        return rollout_output

    if (
        getattr(prompt_batch, "batch", None) is not None
        and getattr(rollout_output, "batch", None) is not None
    ):
        for key in ROLLOUT_PROMPT_BATCH_KEYS:
            if key not in prompt_batch.batch or key not in rollout_output.batch:
                continue
            try:
                same_value = prompt_batch.batch[key].shape == rollout_output.batch[
                    key
                ].shape and prompt_batch.batch[key].equal(rollout_output.batch[key])
            except Exception:
                same_value = False
            if not same_value:
                print(
                    f"[DataProto union] Drop rollout-side '{key}' in {context}; "
                    "prompt-side identity is kept canonical.",
                    flush=True,
                )
            rollout_output.batch.pop(key)

    if hasattr(prompt_batch, "non_tensor_batch") and hasattr(
        rollout_output, "non_tensor_batch"
    ):
        # The environment may perturb the instruction. Keep the actual
        # sample-aligned rollout instruction, rather than the prompt's stale
        # placeholder, while retaining prompt-side task/trial identity below.
        if "task_descriptions" in rollout_output.non_tensor_batch:
            from verl.utils.task_description_contract import normalize_task_descriptions

            descriptions = normalize_task_descriptions(
                rollout_output.non_tensor_batch["task_descriptions"],
                len(rollout_output),
                context=context,
            )
            rollout_output.non_tensor_batch["task_descriptions"] = np.asarray(
                descriptions, dtype=object
            )
            prompt_batch.non_tensor_batch.pop("task_descriptions", None)
            prompt_batch.meta_info.pop("task_descriptions", None)
        for key in ROLLOUT_PROMPT_NON_TENSOR_KEYS:
            if (
                key in prompt_batch.non_tensor_batch
                and key in rollout_output.non_tensor_batch
            ):
                del rollout_output.non_tensor_batch[key]

    if hasattr(rollout_output, "meta_info") and rollout_output.meta_info is not None:
        prompt_meta_info = getattr(prompt_batch, "meta_info", None) or {}
        for key in ROLLOUT_VOLATILE_META_KEYS:
            if key in rollout_output.meta_info:
                if key in prompt_meta_info:
                    print(
                        f"[DataProto union] Drop rollout-side meta '{key}' in {context}; "
                        "per-sample values are kept in non_tensor_batch.",
                        flush=True,
                    )
                rollout_output.meta_info.pop(key, None)

    return prompt_batch.union(rollout_output)


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(
    data: DataProto,
    kl_ctrl: core_algos.AdaptiveKLController,
    kl_penalty="kl",
    action_token_len=7,
    action_chunks_len=8,
    config=None,
):
    responses = data.batch["responses"]

    traj_length = responses.size(1) * action_chunks_len
    action_length = action_token_len  # next fix
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    # attention_mask = data.batch['attention_mask']
    finish_step = data.batch["finish_step"] * action_length

    steps = torch.arange(
        traj_length * action_length, device=data.batch["responses"].device
    )  # (traj_len,)
    steps_expanded = steps.unsqueeze(0).expand(data.batch["responses"].size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)

    disable_wm_kl = False
    if config is not None:
        try:
            train_mode = str(getattr(config.trainer, "train_mode", "MERL")).upper()
        except Exception:
            train_mode = "MERL"
        try:
            wm_cfg = getattr(config.actor_rollout_ref, "world_model", None)
        except Exception:
            wm_cfg = None
        try:
            use_wm_reward_proxy = bool(
                getattr(wm_cfg, "use_wm_reward_proxy", train_mode != "MERL")
            )
        except Exception:
            use_wm_reward_proxy = train_mode != "MERL"
        default_disable = False
        try:
            disable_wm_kl = bool(
                getattr(wm_cfg, "disable_wm_kl_penalty", default_disable)
            )
        except Exception:
            disable_wm_kl = default_disable

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"],
            data.batch["ref_log_prob"],
            kl_penalty=kl_penalty,
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    kl_metric_mask = response_mask
    wm_kl_disabled_tokens = 0.0
    if disable_wm_kl and "is_wm" in data.batch:
        is_wm = _first_sample_column(data.batch["is_wm"], batch_size).to(
            device=response_mask.device, dtype=torch.bool
        )
        non_wm_mask = (~is_wm).unsqueeze(1)
        wm_token_mask = response_mask & is_wm.unsqueeze(1)
        wm_kl_disabled_tokens = float(wm_token_mask.sum().detach().item())
        kld = kld * non_wm_mask.to(dtype=kld.dtype)
        kl_metric_mask = response_mask & non_wm_mask

    token_level_rewards = token_level_scores - beta * kld

    kl_counts = kl_metric_mask.sum(axis=-1)
    kl_denom = kl_counts.clamp_min(1)
    current_kl_per_sample = (kld * kl_metric_mask).sum(axis=-1) / kl_denom
    valid_kl_rows = kl_counts > 0
    if bool(valid_kl_rows.any().item()):
        current_kl = torch.mean(current_kl_per_sample[valid_kl_rows], dim=0).item()
    else:
        current_kl = 0.0

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {
        "critic/kl": current_kl,
        "critic/kl_coeff": beta,
        "critic/wm_kl_disabled": 1.0 if disable_wm_kl else 0.0,
        "critic/wm_kl_disabled_tokens": wm_kl_disabled_tokens,
    }

    return data, metrics


def compute_advantage(data: DataProto, gamma, lam, adv_estimator, config):

    responses = data.batch["responses"]
    response_length = responses.size(1) * responses.size(2)
    # attention_mask = data.batch['attention_mask']
    finish_step = (
        data.batch["finish_step"] * config.actor_rollout_ref.model.action_token_len
    )
    steps = torch.arange(
        response_length, device=data.batch["responses"].device
    )  # (traj_len,)
    steps_expanded = steps.unsqueeze(0).expand(data.batch["responses"].size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)

    token_level_rewards = (
        data.batch["token_level_rewards"]
        if "token_level_rewards" in list(data.batch.keys())
        else data.batch["token_level_scores"]
    )

    # TODO: add other ways to estimate advantages
    if adv_estimator == "rloo":
        # prompt_ids = data.batch['prompts']
        # prompt_length = prompt_ids.shape[-1]
        # valid_response_length = data.batch['attention_mask'][:,prompt_length:].sum(-1)
        advantages, returns = core_algos.compute_rloo_returns(
            data=data,
            eos_mask=response_mask,
            n_samples=config.data.n_samples,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == "gae":
        values = data.batch["values"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch["token_level_rewards"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == "grpo":
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch.get(
            "grpo_uid",
            data.non_tensor_batch.get("group_uid", data.non_tensor_batch["uid"]),
        )
        responses = data.batch["responses"]
        response_length = responses.size(1) * responses.size(2)
        finish_step = (
            data.batch["finish_step"] * config.actor_rollout_ref.model.action_token_len
        )
        steps = torch.arange(
            response_length, device=data.batch["responses"].device
        )  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(data.batch["responses"].size(0), -1)
        response_mask = steps_expanded < finish_step.unsqueeze(
            1
        )  # (batch_size, traj_len)
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == "reinforce_plus_plus":
        token_level_rewards = data.batch["token_level_rewards"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == "remax":
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch["reward_baselines"]

        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=token_level_rewards,
            reward_baselines=reward_baselines,
            eos_mask=response_mask,
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _get_positive_int_attr(container, key: str, default: int = 0) -> int:
    try:
        value = getattr(container, key)
    except Exception:
        try:
            value = container.get(key, default)
        except Exception:
            value = default
    if value is None:
        return int(default)
    if isinstance(value, str) and value.strip() == "":
        return int(default)
    return max(0, int(value))


def _first_sample_column(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.reshape(1).expand(batch_size)
    if tensor.shape[0] != batch_size:
        return tensor.reshape(batch_size, -1)[:, 0]
    return tensor.reshape(batch_size, -1)[:, 0]


def _add_masked_success_metrics(
    metrics: dict,
    prefix: str,
    success: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    success = success.to(dtype=torch.float32)
    if mask is not None:
        mask = mask.to(device=success.device, dtype=torch.bool)
        success = success[mask]
    count = int(success.numel())
    success_sum = float(success.sum().detach().item()) if count > 0 else 0.0
    metrics[f"{prefix}_num_samples"] = float(count)
    metrics[f"{prefix}_success_count"] = success_sum
    metrics[f"{prefix}_success_rate"] = success_sum / float(count) if count > 0 else 0.0


def compute_data_metrics(batch, config):
    # TODO: add response length
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]
    # add
    finish_step = (
        batch.batch["finish_step"] * config.actor_rollout_ref.model.action_token_len
    )
    steps = torch.arange(
        batch.batch["responses"].size(1) * batch.batch["responses"].size(2),
        device=advantages.device,
    )  # (traj_len,)
    steps_expanded = steps.unsqueeze(0).expand(batch.batch["responses"].size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
    #
    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": masked_mean(advantages, response_mask)
        .detach()
        .item(),
        "critic/advantages/max": torch.max(advantages[response_mask.bool()])
        .detach()
        .item(),
        "critic/advantages/min": torch.min(advantages[response_mask.bool()])
        .detach()
        .item(),
        # returns
        "critic/returns/mean": masked_mean(returns, response_mask).detach().item(),
        "critic/returns/max": torch.max(returns[response_mask.bool()]).detach().item(),
        "critic/returns/min": torch.min(returns[response_mask.bool()]).detach().item(),
        # response length
    }
    responses = batch.batch["responses"]
    action_chunks_len = int(
        getattr(config.actor_rollout_ref.model, "action_chunks_len", 1)
    )
    represented_horizon = int(responses.size(1) * max(1, action_chunks_len))
    raw_finish_step = batch.batch["finish_step"].view(len(batch), -1)[:, 0].float()
    metrics.update(
        {
            "rollout/represented_horizon": float(represented_horizon),
            "rollout/finish_step_mean": raw_finish_step.mean().detach().item(),
            "rollout/finish_step_max": raw_finish_step.max().detach().item(),
            "rollout/full_horizon_ratio": (
                raw_finish_step >= max(1, represented_horizon)
            )
            .float()
            .mean()
            .detach()
            .item(),
        }
    )
    if "complete" in batch.batch:
        complete = batch.batch["complete"].view(len(batch), -1)[:, 0].float()
        if "env_complete" in batch.batch:
            env_complete = (
                batch.batch["env_complete"].view(len(batch), -1)[:, 0].float()
            )
        else:
            env_complete = complete
        if "wm_proxy_complete" in batch.batch:
            wm_proxy_complete = (
                batch.batch["wm_proxy_complete"].view(len(batch), -1)[:, 0].float()
            )
        else:
            wm_proxy_complete = complete

        metrics["rollout/success_rate"] = env_complete.mean().detach().item()
        metrics["rollout/success_count"] = env_complete.sum().detach().item()
        metrics["rollout/num_samples"] = float(env_complete.numel())
        metrics["rollout/env_success_rate"] = metrics["rollout/success_rate"]
        metrics["rollout/env_success_count"] = metrics["rollout/success_count"]
        metrics["rollout/legacy_complete_rate"] = complete.mean().detach().item()
        if "wm_proxy_score" in batch.batch:
            proxy_score = (
                batch.batch["wm_proxy_score"].view(len(batch), -1)[:, 0].float()
            )
            metrics["rollout/wm_proxy_score_mean"] = (
                proxy_score.mean().detach().item()
            )
            metrics["rollout/wm_proxy_score_max"] = (
                proxy_score.max().detach().item()
            )
        is_wm = batch.batch.get("is_wm", None)
        if is_wm is not None:
            is_wm = _first_sample_column(is_wm, len(batch)).to(
                device=env_complete.device, dtype=torch.bool
            )
            real_mask = ~is_wm
            _add_masked_success_metrics(
                metrics, "rollout/real", env_complete, real_mask
            )
            _add_masked_success_metrics(
                metrics, "rollout/wm_proxy", wm_proxy_complete, is_wm
            )
            metrics["train/real_success_rate"] = metrics["rollout/real_success_rate"]
            metrics["train/wm_proxy_success_rate"] = metrics[
                "rollout/wm_proxy_success_rate"
            ]
        else:
            _add_masked_success_metrics(metrics, "rollout/real", env_complete)
            metrics["train/real_success_rate"] = metrics["rollout/real_success_rate"]
    if "is_dummy" in batch.batch:
        is_dummy = batch.batch["is_dummy"].reshape(len(batch), -1)
        metrics["rollout/dummy_step_ratio"] = is_dummy.float().mean().detach().item()
        metrics["rollout/dummy_sample_ratio"] = (
            is_dummy.all(dim=1).float().mean().detach().item()
        )
    return metrics


def debug_dummy(dp, name):
    if "is_dummy" not in dp.batch:
        print(f"[{name}] no is_dummy")
        return
    if len(dp) == 0:
        print(f"[{name}] empty")
        return
    is_dummy = dp.batch["is_dummy"].view(len(dp), -1)
    print(
        f"[{name}] dummy stats:",
        "B =",
        len(dp),
        "all_dummy =",
        (is_dummy.all(dim=1)).sum().item(),
        "has_real =",
        ((~is_dummy).any(dim=1)).sum().item(),
    )


def inspect_dataproto_keys(dp: DataProto, name="DataProto"):
    print(f"\n[{name}] len = {len(dp)}")

    if dp.batch is None:
        print("  batch: None")
    else:
        print("  batch keys:", list(dp.batch.keys()))
        for k, v in dp.batch.items():
            print(
                f"    {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}"
            )

    if hasattr(dp, "non_tensor_batch"):
        print("  non_tensor_batch keys:", list(dp.non_tensor_batch.keys()))
        for k, v in dp.non_tensor_batch.items():
            print(f"    {k}: len={len(v)}")

    if hasattr(dp, "meta_info"):
        print("  meta_info keys:", list(dp.meta_info.keys()))


def _get_concat_pad_fill_value(
    key: str, tensor: torch.Tensor, for_missing_key: bool = False
):
    if for_missing_key:
        return False if tensor.dtype == torch.bool else 0
    if key == "is_dummy":
        return True if tensor.dtype == torch.bool else 1
    if key == "env_dones":
        return False if tensor.dtype == torch.bool else 0
    return 0


def _pad_tensor_for_concat(
    key: str, tensor: torch.Tensor, target_tail_shape: Tuple[int, ...]
) -> torch.Tensor:
    target_shape = (tensor.shape[0],) + tuple(target_tail_shape)
    if tuple(tensor.shape) == target_shape:
        return tensor

    padded = tensor.new_full(
        target_shape,
        _get_concat_pad_fill_value(key, tensor),
    )
    copy_slices = [slice(0, tensor.shape[0])]
    for current_size, target_size in zip(tensor.shape[1:], target_tail_shape):
        copy_slices.append(slice(0, min(current_size, target_size)))
    copy_slices = tuple(copy_slices)
    padded[copy_slices] = tensor[copy_slices]
    return padded


def align_dataproto_list_for_concat(dataprotos: list[DataProto]) -> list[DataProto]:
    dataprotos = [dp for dp in dataprotos if dp is not None]
    if len(dataprotos) <= 1:
        return dataprotos

    all_batch_keys = set()
    all_non_tensor_keys = set()
    all_meta_keys = set()
    for dp in dataprotos:
        if dp.batch is not None:
            all_batch_keys.update(dp.batch.keys())
        all_non_tensor_keys.update(dp.non_tensor_batch.keys())
        all_meta_keys.update(dp.meta_info.keys())

    for key in all_batch_keys:
        template_tensor = None
        target_tail_shape = None
        target_ndim = None
        for dp in dataprotos:
            if dp.batch is None or key not in dp.batch:
                continue
            tensor = dp.batch[key]
            if template_tensor is None:
                template_tensor = tensor
                target_ndim = tensor.ndim
                target_tail_shape = tuple(tensor.shape[1:])
                continue
            if tensor.ndim != target_ndim:
                raise RuntimeError(
                    f"Cannot align key '{key}' for concat: ndim mismatch {target_ndim} vs {tensor.ndim}."
                )
            target_tail_shape = tuple(
                max(current, incoming)
                for current, incoming in zip(target_tail_shape, tensor.shape[1:])
            )

        if template_tensor is None:
            continue

        for dp in dataprotos:
            if dp.batch is None:
                continue
            if key not in dp.batch:
                shape = (len(dp),) + tuple(target_tail_shape)
                dp.batch[key] = template_tensor.new_full(
                    shape,
                    _get_concat_pad_fill_value(
                        key,
                        template_tensor,
                        for_missing_key=True,
                    ),
                )
                continue

            tensor = dp.batch[key]
            if tensor.ndim != target_ndim:
                raise RuntimeError(
                    f"Cannot align key '{key}' for concat: ndim mismatch {target_ndim} vs {tensor.ndim}."
                )
            if tuple(tensor.shape[1:]) != tuple(target_tail_shape):
                dp.batch[key] = _pad_tensor_for_concat(
                    key,
                    tensor,
                    target_tail_shape,
                )

    for key in all_non_tensor_keys:
        for dp in dataprotos:
            if key not in dp.non_tensor_batch:
                dp.non_tensor_batch[key] = np.array([None] * len(dp), dtype=object)

    for key in all_meta_keys:
        reference_value = None
        for dp in dataprotos:
            if key in dp.meta_info:
                reference_value = dp.meta_info[key]
                break
        if reference_value is None:
            continue
        for dp in dataprotos:
            if key not in dp.meta_info:
                dp.meta_info[key] = reference_value

    return dataprotos


def align_keys_between_pools(
    dp1: DataProto, dp2: DataProto
) -> Tuple[DataProto, DataProto]:
    """
    Align batch keys, non_tensor_batch keys, and meta_info keys between two DataProto objects.
    Missing keys are filled with zeros (for tensors) or None (for arrays/objects).
    Returns the aligned dp1, dp2.
    """
    aligned = align_dataproto_list_for_concat([dp1, dp2])
    return aligned[0], aligned[1]


def get_or_create_wm_trainer(
    wm_cfg_path,
    rollout_base_dir,
    wm_options,
    trainer_rank=0,
    wm_overrides=None,
):
    try:
        return ray.get_actor("world_model_trainer")
    except ValueError:
        print(f"Create world model trainer on {trainer_rank}.")
        return WorldModelTrainer.options(**wm_options).remote(
            wm_cfg_path,
            rollout_base_dir,
            trainer_rank,
            wm_overrides,
        )


class RayTrainer(object):

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"{role_worker_mapping.keys()}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = (
            Role.RefPolicy in role_worker_mapping
            and config.algorithm.kl_ctrl.kl_coef > 0
        )
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        # _prepare_resume_for_worker_init() runs before init_workers() sets this,
        # so initialize it here to avoid attribute-order bugs.
        self.use_critic = self.config.algorithm.adv_estimator == "gae"

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == "fixed":
                self.kl_ctrl = core_algos.FixedKLController(
                    kl_coef=config.algorithm.kl_ctrl.kl_coef
                )
            elif config.algorithm.kl_ctrl.type == "adaptive":
                assert (
                    config.algorithm.kl_ctrl.horizon > 0
                ), f"horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}"
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon,
                )
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        self._create_dataloader()

        self.update_wm = self.config.actor_rollout_ref.world_model.get(
            "fine_tune", False
        )
        self.wm_trainer = None
        self._resume_state = None
        self._resume_state_source = None
        # Prioritized replay pools (one for real, one for wm)
        if not hasattr(self, "real_prioritized_pool"):
            train_batch_size = int(getattr(self.config.data, "train_batch_size", 1))
            n_samples = int(getattr(self.config.data, "n_samples", 1))
            default_pool_capacity = max(1, train_batch_size * n_samples * 8)
            trainer_cfg = getattr(self.config, "trainer", None)
            wm_cfg = getattr(self.config.actor_rollout_ref, "world_model", None)
            real_capacity = int(
                getattr(
                    self.config,
                    "prio_real_capacity",
                    getattr(
                        trainer_cfg,
                        "prio_real_capacity",
                        getattr(
                            wm_cfg,
                            "prio_real_capacity",
                            default_pool_capacity,
                        ),
                    ),
                )
            )
            wm_capacity = int(
                getattr(
                    self.config,
                    "prio_wm_capacity",
                    getattr(
                        trainer_cfg,
                        "prio_wm_capacity",
                        getattr(
                            wm_cfg,
                            "prio_wm_capacity",
                            default_pool_capacity,
                        ),
                    ),
                )
            )
            real_capacity = max(1, real_capacity)
            wm_capacity = max(1, wm_capacity)
            self.real_prioritized_pool = PrioritizedPool(capacity=real_capacity)
            self.wm_prioritized_pool = PrioritizedPool(capacity=wm_capacity)
            print(
                f"[ReplayPool] real_capacity={real_capacity}, wm_capacity={wm_capacity}, "
                f"default_capacity={default_pool_capacity}",
                flush=True,
            )

        #! rebuild last rollout video path
        # exp_name = self.config.actor_rollout_ref.rollout.experiment_name
        # rollout_path = os.path.join("/path/to/rollouts", exp_name)
        # if os.path.exists(rollout_path):
        #     print("[WARN] rollout_path exists, remove it.")
        #     shutil.rmtree(rollout_path)
        # os.makedirs(rollout_path, exist_ok=True)
        #! rebuild last rollout video path
        rollout_path = os.path.abspath(self.config.actor_rollout_ref.rollout_base_dir)
        with open_dict(self.config):
            self.config.actor_rollout_ref.rollout_base_dir = rollout_path
        preserve_rollout_base_dir = bool(
            self.config.trainer.get("preserve_rollout_base_dir", False)
        )
        if (
            (not self.config.trainer.get("val_only", False))
            and (not preserve_rollout_base_dir)
            and os.path.exists(rollout_path)
        ):
            print("[WARN] rollout_path exists, remove it.")
            shutil.rmtree(rollout_path)
        os.makedirs(rollout_path, exist_ok=True)

    def _create_dataloader(self):  # next fix
        from torch.utils.data import DataLoader
        from verl.utils.libero_path import ensure_libero_pro_root, ensure_libero_root

        rollout_cfg = getattr(self.config.actor_rollout_ref, "rollout", None)
        use_libero_pro = bool(getattr(rollout_cfg, "use_libero_pro", False))
        if use_libero_pro:
            ensure_libero_pro_root(
                evaluation_config_path=getattr(
                    rollout_cfg, "libero_pro_eval_config_path", None
                )
            )
        else:
            ensure_libero_root()

        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rob_dataset import (
            LIBERO_Dataset,
            Robotwin_Dataset,
            collate_fn,
        )

        if "libero" in self.config.data.task_suite_name:
            self.train_dataset = LIBERO_Dataset(
                self.config.data.task_suite_name,
                num_trials_per_task=self.config.data.num_trials_per_task,
                train_val="train",
                task_ids=getattr(rollout_cfg, "allowed_task_ids", None),
            )
            self.val_dataset = LIBERO_Dataset(
                self.config.data.task_suite_name,
                num_trials_per_task=self.config.data.num_trials_per_task,
                train_val="valid",
                task_ids=getattr(rollout_cfg, "allowed_task_ids", None),
                trial_offset=int(self.config.data.get("eval_trial_offset", 0)),
            )
            self.rollout_dataset = LIBERO_Dataset(
                self.config.data.task_suite_name,
                num_trials_per_task=self.config.data.num_trials_per_task,
                train_val="rollout",
                task_ids=getattr(rollout_cfg, "allowed_task_ids", None),
            )

        elif "robotwin" in self.config.data.task_suite_name:
            # (cjh) We assume here that data set names are "robotwin_{task_name}" or "robotwin_all"
            self.train_dataset = Robotwin_Dataset(
                self.config.data.task_suite_name,
                num_trials_per_task=self.config.data.num_trials_per_task,
                train_val="train",
            )
            self.val_dataset = Robotwin_Dataset(
                self.config.data.task_suite_name,
                num_trials_per_task=self.config.data.num_trials_per_task,
                train_val="valid",
            )
        else:
            raise ValueError(
                f"Unsupported task suite name: {self.config.data.task_suite_name}"
            )

        # self.config.data.oversample_factor = 1
        self.train_dataloader = BufferedDataLoader(
            DataLoader(
                dataset=self.train_dataset,
                batch_size=int(
                    self.config.data.train_batch_size
                    * self.config.data.oversample_factor
                ),
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
            )
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        self.rollout_dataloader = DataLoader(
            dataset=self.rollout_dataset,
            batch_size=self.config.data.rollout_batch_size,
            collate_fn=collate_fn,
        )

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")
        print(f"Size of rollout dataloader: {len(self.rollout_dataloader)}")

        if self.config.trainer.get("rollout_before_train", False):
            if len(self.rollout_dataset) % int(self.config.trainer.n_gpus_per_node):
                raise ValueError("Collection trials must divide evenly across actor GPUs; use --actor-gpus 1")
        elif not self.config.trainer.get("val_only", False):
            assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1
        assert len(self.rollout_dataloader) >= 1

        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )
        step_limit = self.config.trainer.get("total_training_steps")
        if step_limit is not None:
            if isinstance(step_limit, bool) or not isinstance(step_limit, int) or step_limit < 1:
                raise ValueError("trainer.total_training_steps must be a positive integer or null")
            total_training_steps = min(total_training_steps, step_limit)

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )
            self.config.critic.optim.total_training_steps = total_training_steps

    def _training_step_limit_reached(self, global_steps):
        limit = self.config.trainer.get("total_training_steps")
        if limit is not None and global_steps >= limit:
            return True
        budget = float(self.config.trainer.get("max_training_seconds", 0) or 0)
        if budget < 0:
            raise ValueError("max_training_seconds cannot be negative")
        started = getattr(self, "_training_started", None)
        return budget > 0 and started is not None and time.monotonic() - started >= budget

    @staticmethod
    def _safe_int(value, default=0):
        # Resume values may come from Hydra/JSON/log strings; normalize defensively.
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _safe_float(value, default=None):
        # Keep only finite floats so NaN/Inf from logs do not poison resume state.
        if value is None:
            return default
        try:
            value = float(value)
            if np.isfinite(value):
                return value
        except Exception:
            pass
        return default

    @staticmethod
    def _extract_global_step_from_dir(path: str):
        name = os.path.basename(os.path.normpath(path))
        if not name.startswith("global_step_"):
            return None
        try:
            return int(name.split("global_step_")[-1])
        except Exception:
            return None

    @staticmethod
    def _load_checkpoint_meta(checkpoint_dir: str):
        if not checkpoint_dir:
            return None
        meta_path = os.path.join(checkpoint_dir, "checkpoint_meta.json")
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            return payload if isinstance(payload, dict) else None
        except Exception as exc:
            print(
                f"[resume] Failed to load checkpoint metadata from {meta_path}: {exc}"
            )
            return None

    @staticmethod
    def _hf_checkpoint_has_weights(checkpoint_dir: str):
        direct_weight_files = (
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
            "adapter_model.bin",
            "adapter_model.safetensors",
        )
        for filename in direct_weight_files:
            if os.path.isfile(os.path.join(checkpoint_dir, filename)):
                return True

        shard_patterns = (
            "pytorch_model-*.bin",
            "model-*.safetensors",
            "adapter_model-*.bin",
            "adapter_model-*.safetensors",
        )
        for pattern in shard_patterns:
            if len(glob.glob(os.path.join(checkpoint_dir, pattern))) > 0:
                return True
        return False

    @classmethod
    def _inspect_checkpoint_dir(cls, checkpoint_dir: str, component: str = None):
        if not checkpoint_dir or (not os.path.isdir(checkpoint_dir)):
            return None, None, "missing_directory"

        checkpoint_meta = cls._load_checkpoint_meta(checkpoint_dir)

        if component == "world_model":
            world_model_ckpt = os.path.join(checkpoint_dir, "world_model.pth")
            if os.path.isfile(world_model_ckpt):
                return "world_model_state_dict", checkpoint_meta, None
            return None, checkpoint_meta, "missing_world_model_pth"

        if isinstance(checkpoint_meta, dict):
            checkpoint_format = str(checkpoint_meta.get("format", "")).strip().lower()

            if checkpoint_format == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
                return None, checkpoint_meta, "legacy_local_state_dict_unsupported"

            if checkpoint_format == FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
                shard_paths = sorted(
                    glob.glob(os.path.join(checkpoint_dir, "rank_*.pt"))
                )
                if len(shard_paths) == 0:
                    return None, checkpoint_meta, "missing_local_state_shards"
                expected_world_size = checkpoint_meta.get("world_size", None)
                try:
                    expected_world_size = (
                        int(expected_world_size)
                        if expected_world_size is not None
                        else None
                    )
                except Exception:
                    expected_world_size = None
                if (
                    expected_world_size is not None
                    and expected_world_size > 0
                    and len(shard_paths) < expected_world_size
                ):
                    return (
                        None,
                        checkpoint_meta,
                        f"incomplete_local_state_shards:{len(shard_paths)}/{expected_world_size}",
                    )
                return checkpoint_format, checkpoint_meta, None

            if checkpoint_format == HF_FULL_STATE_DICT_CHECKPOINT_FORMAT:
                if cls._hf_checkpoint_has_weights(checkpoint_dir):
                    return checkpoint_format, checkpoint_meta, None
                return None, checkpoint_meta, "missing_hf_weight_files"

        if os.path.isfile(os.path.join(checkpoint_dir, "config.json")):
            if cls._hf_checkpoint_has_weights(checkpoint_dir):
                return "hf_full_state_dict", checkpoint_meta, None
            return None, checkpoint_meta, "missing_hf_weight_files"

        return None, checkpoint_meta, "missing_checkpoint_markers"

    @classmethod
    def _detect_checkpoint_format(cls, checkpoint_dir: str):
        checkpoint_format, checkpoint_meta, _ = cls._inspect_checkpoint_dir(
            checkpoint_dir=checkpoint_dir
        )
        return checkpoint_format, checkpoint_meta

    def _cleanup_stale_rollout_shards(self, keep_from_global_steps: int):
        if self.wm_trainer is None:
            return {}

        try:
            cleanup_result = ray.get(
                self.wm_trainer.cleanup_stale_rollout_data.remote(
                    keep_from_global_steps=int(keep_from_global_steps)
                )
            )
        except Exception as exc:
            print(f"[rollout gc] Failed to cleanup stale rollout shards: {exc}")
            return {"wm/cleanup/error": str(exc)}

        if not isinstance(cleanup_result, dict):
            return {}

        metrics = {
            "wm/cleanup/removed_dirs": float(cleanup_result.get("removed_dirs", 0)),
            "wm/cleanup/failed_dirs": float(cleanup_result.get("failed_dirs", 0)),
            "wm/cleanup/scanned_dirs": float(cleanup_result.get("scanned_dirs", 0)),
            "wm/cleanup/freed_mb": float(cleanup_result.get("freed_bytes", 0))
            / (1024.0 * 1024.0),
        }
        if cleanup_result.get("cleaned", False):
            print(
                "[rollout gc] Removed stale rollout shard dirs:",
                cleanup_result.get("removed", []),
            )
        return metrics

    def _get_resume_cfg(self):
        resume_cfg = self.config.trainer.get("resume", None)

        def _default_resume_cfg(enable: bool, auto_enabled: bool):
            # Keep backward compatibility with existing fields consumed downstream.
            # auto_enabled marks "implicit default-on" so logs can explain behavior.
            return SimpleNamespace(
                enable=enable,
                resume_dir=self.config.trainer.default_local_dir,
                state_path=None,
                resume_log_path=None,
                resume_epoch=None,
                resume_global_step=None,
                auto_enabled=auto_enabled,
            )

        if resume_cfg is None:
            # Default behavior: auto-attempt resume from default_local_dir.
            # Users can still disable via +trainer.resume.enable=false.
            return _default_resume_cfg(enable=True, auto_enabled=True), True

        # Also support simple boolean style: trainer.resume=true/false.
        if isinstance(resume_cfg, bool):
            return _default_resume_cfg(enable=resume_cfg, auto_enabled=False), bool(
                resume_cfg
            )

        enabled = bool(getattr(resume_cfg, "enable", True))
        return resume_cfg, enabled

    def _find_checkpoint_dir(self, base_dir: str, component: str, target_step=None):
        root = os.path.join(base_dir, component)
        if not os.path.isdir(root):
            return None

        candidates = []
        for name in os.listdir(root):
            full_path = os.path.join(root, name)
            if not os.path.isdir(full_path):
                continue
            step = self._extract_global_step_from_dir(full_path)
            if step is None:
                continue
            checkpoint_format, _, invalid_reason = self._inspect_checkpoint_dir(
                full_path, component=component
            )
            if checkpoint_format is None:
                print(
                    f"[resume] Skip invalid {component} checkpoint candidate {full_path}: {invalid_reason}"
                )
                continue
            candidates.append((step, full_path))

        if len(candidates) == 0:
            return None

        candidates.sort(key=lambda x: x[0])
        if target_step is None:
            return candidates[-1][1]

        valid = [item for item in candidates if item[0] <= target_step]
        if len(valid) > 0:
            return valid[-1][1]
        return candidates[0][1]

    def _cleanup_old_component_checkpoints(
        self,
        component: str,
        keep: int,
        protected_paths: list[str] | None = None,
    ):
        keep = max(0, int(keep))
        root = os.path.join(self.config.trainer.default_local_dir, component)
        if not os.path.isdir(root):
            return []

        protected = {
            os.path.abspath(path)
            for path in (protected_paths or [])
            if path and os.path.isdir(path)
        }

        candidates = []
        invalid_entries = []
        for name in os.listdir(root):
            full_path = os.path.join(root, name)
            if not os.path.isdir(full_path):
                continue
            step = self._extract_global_step_from_dir(full_path)
            if step is None:
                continue
            checkpoint_format, _, invalid_reason = self._inspect_checkpoint_dir(
                full_path, component=component
            )
            if checkpoint_format is None:
                invalid_entries.append((step, full_path, invalid_reason))
                continue
            candidates.append((step, full_path))

        removed_paths = []
        for _, path, invalid_reason in invalid_entries:
            if os.path.abspath(path) in protected:
                continue
            try:
                shutil.rmtree(path)
                removed_paths.append(path)
                print(
                    f"[checkpoint] Removed invalid {component} checkpoint {path}: {invalid_reason}"
                )
            except Exception as e:
                print(
                    f"[checkpoint] Failed to remove invalid {component} checkpoint {path}: {e}"
                )

        if len(candidates) <= keep:
            return removed_paths

        candidates.sort(key=lambda item: item[0])
        protected_entries = [
            (step, path)
            for step, path in candidates
            if os.path.abspath(path) in protected
        ]
        unprotected_entries = [
            (step, path)
            for step, path in candidates
            if os.path.abspath(path) not in protected
        ]

        keep_budget = max(0, keep - len(protected_entries))
        retained_unprotected = {
            os.path.abspath(path)
            for _, path in sorted(
                unprotected_entries, key=lambda item: item[0], reverse=True
            )[:keep_budget]
        }

        for _, path in unprotected_entries:
            if os.path.abspath(path) in retained_unprotected:
                continue
            try:
                shutil.rmtree(path)
                removed_paths.append(path)
            except Exception as e:
                print(
                    f"[checkpoint] Failed to remove old {component} checkpoint {path}: {e}"
                )
        return removed_paths

    def _find_latest_log_file(self, resume_dir: str):
        pattern = os.path.join(resume_dir, "run_*.log")
        candidates = sorted(glob.glob(pattern))
        if len(candidates) == 0:
            return None
        return candidates[-1]

    def _load_resume_state_from_log(self, log_path: str):
        # Fallback path when no structured resume_state json exists.
        # We recover step/epoch and WM scheduling statistics from the latest valid log row.
        if (log_path is None) or (not os.path.isfile(log_path)):
            return None

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[resume] Failed to read log file {log_path}: {e}")
            return None

        for raw_line in reversed(lines):
            line = raw_line.strip()
            if (not line) or line.startswith("#"):
                continue

            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue

            step = self._safe_int(parts[1], default=None)
            if step is None:
                continue

            try:
                payload = json.loads(parts[2])
            except Exception:
                continue

            epoch_from_log = payload.get("train/epoch", payload.get("epoch", None))
            if epoch_from_log is None:
                estimated_epoch = step // max(1, len(self.train_dataloader))
            else:
                estimated_epoch = self._safe_int(epoch_from_log, 0)

            return {
                "epoch": max(0, estimated_epoch),
                "global_step": max(0, step),
                "wm_loss_ema": self._safe_float(payload.get("wm/loss_ema", None)),
                "wm_ratio_signal_ema": self._safe_float(
                    payload.get("wm/ratio_signal_ema", None)
                ),
                "last_wm_ratio_signal_ema": self._safe_float(
                    payload.get(
                        "wm/ratio_signal_prev", payload.get("wm/ratio_signal_ema", None)
                    )
                ),
                "r_wm": self._safe_float(
                    payload.get("wm/ratio_wm", payload.get("wm/r_prev", 0.0)), 0.0
                ),
                "wm_weak_update_active": bool(
                    self._safe_float(payload.get("wm/weak_update_active", 0.0), 0.0)
                    >= 0.5
                ),
                "wm_weak_update_last_calibration_step": self._safe_int(
                    payload.get("wm/weak_update_last_calibration_step", None), None
                ),
                "resume_log_path": log_path,
                "resume_from": "log",
            }

        return None

    def _load_resume_state_from_json(self, resume_dir: str, state_path: str = None):
        # Priority order: explicit state path -> standard resume subdir -> legacy root file.
        candidates = []
        if state_path:
            candidates.append(state_path)
        candidates.append(
            os.path.join(resume_dir, "resume", "resume_state_latest.json")
        )
        candidates.append(os.path.join(resume_dir, "resume_state_latest.json"))

        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                state["resume_state_path"] = path
                state["resume_from"] = "state_json"
                return state
            except Exception as e:
                print(f"[resume] Failed to load state file {path}: {e}")
        return None

    def _resolve_resume_state(self):
        resume_cfg, resume_enabled = self._get_resume_cfg()
        if not resume_enabled:
            return None

        resume_dir = getattr(
            resume_cfg, "resume_dir", self.config.trainer.default_local_dir
        )
        resume_state_path = getattr(resume_cfg, "state_path", None)
        resume_log_path = getattr(resume_cfg, "resume_log_path", None)
        resume_epoch_override = getattr(resume_cfg, "resume_epoch", None)
        resume_step_override = getattr(resume_cfg, "resume_global_step", None)

        # Prefer structured runtime state; fallback to metrics log parsing.
        state = self._load_resume_state_from_json(
            resume_dir=resume_dir, state_path=resume_state_path
        )
        if state is None:
            if resume_log_path is None:
                resume_log_path = self._find_latest_log_file(resume_dir)
            state = self._load_resume_state_from_log(resume_log_path)

        if state is None:
            if bool(getattr(resume_cfg, "auto_enabled", False)):
                print(
                    f"[resume] Auto-resume is enabled by default, but no state/log found under: {resume_dir}. Start fresh."
                )
            else:
                print(
                    f"[resume] Enabled but no resume state/log found under: {resume_dir}. Start fresh."
                )
            return None

        epoch_overridden = False
        if resume_epoch_override is not None:
            state["epoch"] = self._safe_int(resume_epoch_override, 0)
            epoch_overridden = True
        if resume_step_override is not None:
            state["global_step"] = self._safe_int(resume_step_override, 0)

        state["epoch"] = max(0, self._safe_int(state.get("epoch", 0), 0))
        state["global_step"] = max(0, self._safe_int(state.get("global_step", 0), 0))

        actor_ckpt_dir = state.get("actor_ckpt_dir", None)
        actor_ckpt_format, _, actor_invalid_reason = self._inspect_checkpoint_dir(
            actor_ckpt_dir, component="actor"
        )
        if actor_ckpt_format is None:
            if actor_ckpt_dir:
                print(
                    f"[resume] Ignore invalid actor checkpoint from state {actor_ckpt_dir}: {actor_invalid_reason}"
                )
            actor_ckpt_dir = self._find_checkpoint_dir(
                resume_dir, "actor", target_step=state["global_step"]
            )
        state["actor_ckpt_dir"] = actor_ckpt_dir
        if state["global_step"] > 0 and actor_ckpt_dir is None:
            raise RuntimeError(
                "[resume] Resume requested, but no valid actor checkpoint was found under "
                f"{resume_dir} for step<={state['global_step']}. "
                "Incomplete checkpoint directories are skipped automatically. "
                "Use a valid RESUME_DIR or restart with RESUME_ENABLE=false."
            )

        critic_ckpt_dir = state.get("critic_ckpt_dir", None)
        critic_ckpt_format, _, critic_invalid_reason = self._inspect_checkpoint_dir(
            critic_ckpt_dir, component="critic"
        )
        if critic_ckpt_format is None:
            if critic_ckpt_dir:
                print(
                    f"[resume] Ignore invalid critic checkpoint from state {critic_ckpt_dir}: {critic_invalid_reason}"
                )
            critic_ckpt_dir = self._find_checkpoint_dir(
                resume_dir, "critic", target_step=state["global_step"]
            )
        state["critic_ckpt_dir"] = critic_ckpt_dir

        wm_ckpt_dir = state.get("world_model_ckpt_dir", None)
        wm_ckpt_format, _, wm_invalid_reason = self._inspect_checkpoint_dir(
            wm_ckpt_dir, component="world_model"
        )
        if wm_ckpt_format is None:
            if wm_ckpt_dir:
                print(
                    f"[resume] Ignore invalid world model checkpoint from state {wm_ckpt_dir}: {wm_invalid_reason}"
                )
            wm_ckpt_dir = self._find_checkpoint_dir(
                resume_dir, "world_model", target_step=state["global_step"] + 1
            )
        state["world_model_ckpt_dir"] = wm_ckpt_dir

        actor_step = (
            self._extract_global_step_from_dir(actor_ckpt_dir)
            if actor_ckpt_dir is not None
            else None
        )
        # If the configured/logged step and available actor checkpoint disagree,
        # trust checkpoint directory naming to avoid "state says X, weights are Y".
        if (actor_step is not None) and (actor_step != state["global_step"]):
            print(
                f"[resume] Align global_step from {state['global_step']} to checkpoint step {actor_step}."
            )
            state["global_step"] = actor_step
            if not epoch_overridden:
                state["epoch"] = max(
                    0, actor_step // max(1, len(self.train_dataloader))
                )

        state["resume_dir"] = resume_dir
        return state

    def _sync_vla_runtime_files_for_resume(
        self, source_model_dir: str, target_ckpt_dir: str
    ) -> None:
        """Ensure VLA custom runtime python files exist in resumed actor checkpoint.

        AutoProcessor/AutoModel may require these files via `auto_map` + trust_remote_code.
        Fine-tuned VLA checkpoints also need dataset_statistics.json for action un-normalization.
        """
        if not target_ckpt_dir or not os.path.isdir(target_ckpt_dir):
            return

        vla_name = str(getattr(self.config.actor_rollout_ref.model, "vla", "")).lower()
        if vla_name not in {"openvla", "openvla-oft"}:
            return

        variant_dir = "openvla_oft" if vla_name == "openvla-oft" else "openvla"
        repo_vla_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "utils",
                "vla_utils",
                variant_dir,
            )
        )

        candidate_roots = []
        if source_model_dir and os.path.isdir(source_model_dir):
            candidate_roots.append(source_model_dir)
        candidate_roots.append(repo_vla_dir)

        # OpenVLA-OFT relies on constants/train_utils in modeling_prismatic imports.
        required_files = [
            "configuration_prismatic.py",
            "modeling_prismatic.py",
            "processing_prismatic.py",
        ]
        optional_files = ["constants.py", "train_utils.py", "dataset_statistics.json"]
        all_files = required_files + optional_files

        os.makedirs(target_ckpt_dir, exist_ok=True)
        copied = []
        missing_required = []

        for filename in all_files:
            src = None
            for root in candidate_roots:
                candidate = os.path.join(root, filename)
                if os.path.isfile(candidate):
                    src = candidate
                    break

            if src is None:
                if filename in required_files:
                    missing_required.append(filename)
                continue

            dst = os.path.join(target_ckpt_dir, filename)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
                copied.append(filename)

        if len(copied) > 0:
            print(f"[resume] Synced VLA runtime files into actor checkpoint: {copied}")
        if len(missing_required) > 0:
            print(
                "[resume] Missing required VLA runtime files for resumed actor checkpoint: "
                f"{missing_required}"
            )

    def _prepare_resume_for_worker_init(self):
        # Worker model init reads config.model.path directly.
        # We must rewrite paths here (before init_workers spawns workers).
        state = self._resolve_resume_state()
        if state is None:
            return

        with open_dict(self.config):
            actor_ckpt_dir = state.get("actor_ckpt_dir", None)
            if actor_ckpt_dir:
                original_actor_model_path = self.config.actor_rollout_ref.model.path
                actor_ckpt_format, _ = self._detect_checkpoint_format(actor_ckpt_dir)
                if actor_ckpt_format is None:
                    raise RuntimeError(
                        f"[resume] Cannot determine actor checkpoint format under: {actor_ckpt_dir}"
                    )
                state["actor_ckpt_format"] = actor_ckpt_format
                state["actor_base_model_path"] = original_actor_model_path
                # Keep tokenizer/processor source stable when resumed actor dir is partial.
                self.config.actor_rollout_ref.model.tokenizer_path = (
                    original_actor_model_path
                )
                self.config.actor_rollout_ref.model.resume = True

                if actor_ckpt_format == FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
                    print(
                        "[resume] Actor checkpoint uses lightweight sharded FSDP shards; keep base model path for init and restore weights after worker init."
                    )
                else:
                    # Make resumed actor dir self-contained for trust_remote_code loading.
                    self._sync_vla_runtime_files_for_resume(
                        source_model_dir=original_actor_model_path,
                        target_ckpt_dir=actor_ckpt_dir,
                    )
                    self.config.actor_rollout_ref.model.path = actor_ckpt_dir

            critic_ckpt_dir = state.get("critic_ckpt_dir", None)
            if getattr(self, "use_critic", False) and critic_ckpt_dir:
                critic_ckpt_format, _ = self._detect_checkpoint_format(critic_ckpt_dir)
                if critic_ckpt_format is None:
                    raise RuntimeError(
                        f"[resume] Cannot determine critic checkpoint format under: {critic_ckpt_dir}"
                    )
                state["critic_ckpt_format"] = critic_ckpt_format
                state["critic_base_model_path"] = self.config.critic.model.path
                if critic_ckpt_format != FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
                    self.config.critic.model.path = critic_ckpt_dir

        self._resume_state = state
        self._resume_state_source = state.get("resume_from", "unknown")
        print(
            "[resume] Prepared model init from state:",
            {
                "epoch": state.get("epoch"),
                "global_step": state.get("global_step"),
                "actor_ckpt_dir": state.get("actor_ckpt_dir"),
                "actor_ckpt_format": state.get("actor_ckpt_format"),
                "critic_ckpt_dir": state.get("critic_ckpt_dir"),
                "critic_ckpt_format": state.get("critic_ckpt_format"),
                "world_model_ckpt_dir": state.get("world_model_ckpt_dir"),
                "source": self._resume_state_source,
            },
        )

    def _restore_policy_from_resume(self):
        state = self._resume_state if isinstance(self._resume_state, dict) else None
        if not state:
            return

        actor_ckpt_dir = state.get("actor_ckpt_dir", None)
        actor_ckpt_format = str(state.get("actor_ckpt_format", "") or "").lower()
        if (
            actor_ckpt_dir
            and actor_ckpt_format == FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT
            and hasattr(self, "actor_rollout_wg")
            and self.actor_rollout_wg is not None
        ):
            try:
                self.actor_rollout_wg.load_checkpoint(actor_ckpt_dir)
                print(
                    f"[resume] Restored actor weights from sharded FSDP checkpoint: {actor_ckpt_dir}"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"[resume] Failed to restore actor sharded-state checkpoint from {actor_ckpt_dir}: {exc}"
                ) from exc

        critic_ckpt_dir = state.get("critic_ckpt_dir", None)
        critic_ckpt_format = str(state.get("critic_ckpt_format", "") or "").lower()
        if (
            getattr(self, "use_critic", False)
            and critic_ckpt_dir
            and critic_ckpt_format == FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT
            and hasattr(self, "critic_wg")
            and self.critic_wg is not None
        ):
            try:
                self.critic_wg.load_checkpoint(critic_ckpt_dir)
                print(
                    f"[resume] Restored critic weights from sharded FSDP checkpoint: {critic_ckpt_dir}"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"[resume] Failed to restore critic sharded-state checkpoint from {critic_ckpt_dir}: {exc}"
                ) from exc

    def _sync_world_model_from_resume(self, load_into_wm_trainer: bool = False):
        # Rollout workers cache a world model mapping; sync it explicitly after resume.
        # In MERL mode we may also load the dedicated wm_trainer copy.
        state = self._resume_state if isinstance(self._resume_state, dict) else None
        if not state:
            return

        wm_ckpt_dir = state.get("world_model_ckpt_dir", None)
        if (wm_ckpt_dir is None) or (not os.path.isdir(wm_ckpt_dir)):
            return

        wm_ckpt_file = os.path.join(wm_ckpt_dir, "world_model.pth")
        if not os.path.isfile(wm_ckpt_file):
            return

        try:
            if (
                load_into_wm_trainer
                and hasattr(self, "wm_trainer")
                and self.wm_trainer is not None
            ):
                ray.get(self.wm_trainer.load_checkpoint.remote(wm_ckpt_file))
            self.actor_rollout_wg.load_world_model_mapping(wm_ckpt_dir)
            print(
                f"[resume] Loaded world model from {wm_ckpt_file} (load_into_wm_trainer={load_into_wm_trainer})."
            )
        except Exception as e:
            print(f"[resume] Failed to sync world model checkpoint: {e}")

    def _ensure_wm_trainer_initialized(self) -> bool:
        # Initialize WM trainer lazily so MBRL/MERL paths can safely use wm/eval and wm update.
        if getattr(self, "wm_trainer", None) is not None:
            return True

        wm_cfg = getattr(self.config.actor_rollout_ref, "world_model", None)
        if wm_cfg is None or (not bool(getattr(wm_cfg, "enable", False))):
            return False

        print("[!!!] Start initialize WM trainer worker...")
        wm_cfg_path = wm_cfg.config_path
        wm_overrides = OmegaConf.to_container(wm_cfg, resolve=True)
        rollout_base_dir = (
            self.actor_rollout_wg.rollout_base_dir
            if hasattr(self.actor_rollout_wg, "rollout_base_dir")
            else self.config.actor_rollout_ref.rollout_base_dir
        )
        rollout_base_dir = os.path.abspath(rollout_base_dir)
        with open_dict(self.config):
            self.config.actor_rollout_ref.rollout_base_dir = rollout_base_dir
        trainer_rank = self.config.actor_rollout_ref.wm_gpu_idx
        wm_options = {
            "name": "world_model_trainer",
            "num_gpus": 1,
        }
        self.wm_trainer = get_or_create_wm_trainer(
            wm_cfg_path,
            rollout_base_dir,
            wm_options,
            trainer_rank,
            wm_overrides,
        )
        init_res = ray.get(self.wm_trainer.init_model.remote())
        assert init_res.get("inited", False), "wm trainer has not been initialized."
        self._sync_world_model_from_resume(load_into_wm_trainer=True)
        return True

    def _write_resume_state(
        self,
        *,
        epoch: int,
        global_step: int,
        actor_ckpt_dir: str = None,
        critic_ckpt_dir: str = None,
        world_model_ckpt_dir: str = None,
        save_snapshot: bool = True,
    ):
        # Persist a minimal but sufficient restart snapshot:
        # checkpoint locations + trainer progress + WM scheduler state.
        resume_root = os.path.join(self.config.trainer.default_local_dir, "resume")
        os.makedirs(resume_root, exist_ok=True)

        state = {
            "version": 1,
            "epoch": max(0, self._safe_int(epoch, 0)),
            "global_step": max(0, self._safe_int(global_step, 0)),
            "actor_ckpt_dir": actor_ckpt_dir,
            "critic_ckpt_dir": critic_ckpt_dir,
            "world_model_ckpt_dir": world_model_ckpt_dir,
            "wm_loss_ema": self._safe_float(getattr(self, "_wm_loss_ema", None)),
            "wm_ratio_signal_ema": self._safe_float(
                getattr(self, "_wm_ratio_signal_ema", None)
            ),
            "last_wm_ratio_signal_ema": self._safe_float(
                getattr(self, "_last_wm_ratio_signal_ema", None)
            ),
            "r_wm": self._safe_float(getattr(self, "_r_wm", None), 0.0),
            "wm_weak_update_active": bool(
                getattr(self, "_wm_weak_update_active", False)
            ),
            "wm_weak_update_last_calibration_step": self._safe_int(
                getattr(self, "_wm_weak_update_last_calibration_step", None), None
            ),
            "replay_pool_snapshot": getattr(
                self, "_last_replay_pool_snapshot_path", None
            ),
            "train_mode": str(getattr(self.config.trainer, "train_mode", "MERL")),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }

        latest_path = os.path.join(resume_root, "resume_state_latest.json")
        try:
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, allow_nan=False)

            if save_snapshot:
                snapshot_name = f"resume_state_step_{state['global_step']}.json"
                snapshot_path = os.path.join(resume_root, snapshot_name)
                with open(snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2, allow_nan=False)

            self._resume_state = state
            self._resume_state_source = "runtime"
        except Exception as e:
            print(f"[resume] Failed to write resume state: {e}")

    def _replay_pool_persist_enabled(self) -> bool:
        resume_cfg = self.config.trainer.get("resume", None)
        if resume_cfg is None or isinstance(resume_cfg, bool):
            return False
        return bool(getattr(resume_cfg, "persist_replay_pool", False))

    def _replay_pool_snapshot_root(self, resume_dir: str = None) -> str:
        root_dir = resume_dir or self.config.trainer.default_local_dir
        return os.path.join(root_dir, "resume", "replay_pool")

    def _replay_pool_keep_last(self) -> int:
        resume_cfg = self.config.trainer.get("resume", None)
        if resume_cfg is None or isinstance(resume_cfg, bool):
            return 1
        return max(1, int(getattr(resume_cfg, "replay_pool_keep_last", 1)))

    def _replay_pool_max_bytes(self) -> int:
        resume_cfg = self.config.trainer.get("resume", None)
        if resume_cfg is None or isinstance(resume_cfg, bool):
            return int(15 * 1024**3)
        max_gb = float(getattr(resume_cfg, "replay_pool_max_gb", 15.0))
        return max(1, int(max_gb * 1024**3))

    def _replay_pool_state_dict(self, pool: PrioritizedPool) -> Dict[str, Any]:
        return {
            "capacity": int(pool.capacity),
            "buffer": list(pool.buffer),
            "priorities": list(pool.priorities),
            "next_idx": int(getattr(pool, "next_idx", 0)),
        }

    def _load_replay_pool_state_dict(
        self, pool: PrioritizedPool, state: Dict[str, Any]
    ) -> None:
        if not isinstance(state, dict):
            return
        capacity = max(1, int(getattr(pool, "capacity", state.get("capacity", 1))))
        buffer = list(state.get("buffer", []))
        priorities = list(state.get("priorities", []))
        if len(priorities) != len(buffer):
            priorities = [1.0 for _ in buffer]
        if len(buffer) > capacity:
            buffer = buffer[-capacity:]
            priorities = priorities[-capacity:]
        pool.buffer = buffer
        pool.priorities = [max(float(p), 1e-12) for p in priorities]
        next_idx = int(state.get("next_idx", len(pool.buffer) % capacity))
        pool.next_idx = int(next_idx % capacity)

    def _path_size_bytes(self, path: str) -> int:
        if not path or not os.path.exists(path):
            return 0
        if os.path.isfile(path):
            return int(os.path.getsize(path))
        total = 0
        for root, _, files in os.walk(path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.exists(file_path):
                    total += int(os.path.getsize(file_path))
        return total

    @staticmethod
    def _replay_snapshot_step(path: str) -> int:
        base = os.path.basename(path)
        try:
            return int(base.replace("replay_pool_step_", "").replace(".pt", ""))
        except Exception:
            return -1

    def _latest_replay_pool_snapshot_path(self, resume_dir: str = None) -> str:
        root = self._replay_pool_snapshot_root(resume_dir=resume_dir)
        candidates = glob.glob(os.path.join(root, "replay_pool_step_*.pt"))
        if not candidates:
            return None
        candidates.sort(
            key=lambda p: (self._replay_snapshot_step(p), os.path.getmtime(p))
        )
        return candidates[-1]

    def _cleanup_replay_pool_snapshots(self, protected_path: str = None) -> None:
        root = self._replay_pool_snapshot_root()
        if not os.path.isdir(root):
            return
        root_abs = os.path.abspath(root)
        snapshots = glob.glob(os.path.join(root, "replay_pool_step_*.pt"))
        snapshots.sort(
            key=lambda p: (self._replay_snapshot_step(p), os.path.getmtime(p))
        )
        keep_last = self._replay_pool_keep_last()
        protected_abs = os.path.abspath(protected_path) if protected_path else None

        keep = set(os.path.abspath(p) for p in snapshots[-keep_last:])
        if protected_abs:
            keep.add(protected_abs)

        def _safe_remove(path: str):
            path_abs = os.path.abspath(path)
            if not path_abs.startswith(root_abs + os.sep):
                return
            try:
                os.remove(path_abs)
            except FileNotFoundError:
                pass
            except Exception as exc:
                print(f"[ReplayPool] Failed to remove snapshot {path_abs}: {exc}")

        for path in snapshots:
            if os.path.abspath(path) not in keep:
                _safe_remove(path)

        snapshots = glob.glob(os.path.join(root, "replay_pool_step_*.pt"))
        snapshots.sort(
            key=lambda p: (self._replay_snapshot_step(p), os.path.getmtime(p))
        )
        max_bytes = self._replay_pool_max_bytes()
        total_bytes = sum(self._path_size_bytes(path) for path in snapshots)
        for path in snapshots:
            if total_bytes <= max_bytes:
                break
            path_abs = os.path.abspath(path)
            if path_abs in keep:
                continue
            path_bytes = self._path_size_bytes(path_abs)
            _safe_remove(path_abs)
            total_bytes -= path_bytes

        if total_bytes > max_bytes:
            print(
                "[ReplayPool] Latest replay snapshot exceeds configured cap; "
                f"keep_last={keep_last}, total_gb={total_bytes / 1024**3:.2f}, "
                f"cap_gb={max_bytes / 1024**3:.2f}",
                flush=True,
            )

    def _save_replay_pool_state(self, global_step: int) -> str:
        if not self._replay_pool_persist_enabled():
            return None
        if not hasattr(self, "real_prioritized_pool") or not hasattr(
            self, "wm_prioritized_pool"
        ):
            return None

        root = self._replay_pool_snapshot_root()
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f"replay_pool_step_{int(global_step)}.pt")
        tmp_path = path + ".tmp"
        payload = {
            "version": 1,
            "global_step": int(global_step),
            "real": self._replay_pool_state_dict(self.real_prioritized_pool),
            "wm": self._replay_pool_state_dict(self.wm_prioritized_pool),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            self._last_replay_pool_snapshot_path = path
            self._cleanup_replay_pool_snapshots(protected_path=path)
            print(
                f"[ReplayPool] Saved snapshot: {path} "
                f"(real={len(self.real_prioritized_pool)}, wm={len(self.wm_prioritized_pool)})",
                flush=True,
            )
            return path
        except Exception as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            print(f"[ReplayPool] Failed to save replay snapshot: {exc}", flush=True)
            return None

    def _restore_replay_pool_state(self, resume_state: Dict[str, Any] = None) -> bool:
        if not self._replay_pool_persist_enabled():
            return False
        if not hasattr(self, "real_prioritized_pool") or not hasattr(
            self, "wm_prioritized_pool"
        ):
            return False

        resume_cfg, _ = self._get_resume_cfg()
        resume_dir = getattr(
            resume_cfg, "resume_dir", self.config.trainer.default_local_dir
        )
        snapshot_path = None
        if isinstance(resume_state, dict):
            candidate = resume_state.get("replay_pool_snapshot", None)
            if candidate and os.path.exists(candidate):
                snapshot_path = candidate
        if snapshot_path is None:
            snapshot_path = self._latest_replay_pool_snapshot_path(
                resume_dir=resume_dir
            )
        if snapshot_path is None or not os.path.exists(snapshot_path):
            print("[ReplayPool] No replay snapshot found for resume.", flush=True)
            return False

        try:
            try:
                payload = torch.load(
                    snapshot_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                payload = torch.load(snapshot_path, map_location="cpu")
            self._load_replay_pool_state_dict(
                self.real_prioritized_pool, payload.get("real", {})
            )
            self._load_replay_pool_state_dict(
                self.wm_prioritized_pool, payload.get("wm", {})
            )
            self._last_replay_pool_snapshot_path = snapshot_path
            print(
                f"[ReplayPool] Restored snapshot: {snapshot_path} "
                f"(real={len(self.real_prioritized_pool)}, wm={len(self.wm_prioritized_pool)})",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"[ReplayPool] Failed to restore replay snapshot: {exc}", flush=True)
            return False

    #! key implementation of WMPO: for rollout mode
    def _save_rollouts(self, global_steps=0, rollout_epoch=1, use_wm=False):
        # batch_size = self.config.data.rollout_batch_size
        reward_tensor_lst = []
        rollout_train_split = str(
            self.config.trainer.get("rollout_train_split", "train")
        )
        rollout_eval_split = str(self.config.trainer.get("rollout_eval_split", "eval"))
        rollout_save_eval = bool(self.config.trainer.get("rollout_save_eval", False))
        rollout_save_to_hdfs = bool(
            self.config.trainer.get("rollout_save_to_hdfs", True)
        )
        rollout_do_sample = bool(self.config.trainer.get("rollout_do_sample", True))

        for epoch_idx in range(rollout_epoch):
            for test_data in self.rollout_dataloader:
                test_batch = DataProto.from_single_dict(test_data)
                test_batch.meta_info = {
                    "do_sample": rollout_do_sample,
                    "recompute_log_prob": False,
                    "rollout_base_dir": self.config.actor_rollout_ref.rollout_base_dir,
                    "save_to_hdfs": rollout_save_to_hdfs,
                    "global_steps": global_steps,
                    "return_rollouts": bool(rollout_save_to_hdfs or rollout_save_eval),
                    "strip_rollout_media": True,
                    "save_eval": rollout_save_eval,
                    "train_split": rollout_train_split,
                    "eval_split": rollout_eval_split,
                    "use_wm": use_wm,
                }
                gen_test_batch = self.actor_rollout_wg.generate_sequences(test_batch)
                test_batch = union_prompt_and_rollout_output(
                    test_batch,
                    gen_test_batch,
                    context="_save_rollouts",
                )
                (
                    verifier_score,
                    reward_metrics,
                    format_metrics,
                    reward_format_metrics,
                ) = self.val_reward_fn.verify(
                    test_batch
                )  # call val_reward_fn to verify and score the batch
                reward_tensor = torch.tensor(
                    verifier_score, dtype=torch.float32
                ).unsqueeze(-1)
                reward_tensor_lst.append(reward_tensor)
            print(f"Epoch {epoch_idx}")
        metric_dict = {}
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        metric_dict[f"rollout_reward/{self.config.data.task_suite_name}"] = (
            reward_tensor.mean().item()
        )
        return metric_dict

    def _validate(self, global_steps=0):
        reward_tensor_lst = []
        success_tensor_lst = []
        data_source_lst = []
        metric_dict = {}
        invalid_rollout_count = 0
        total_rollout_count = 0
        rollout_cfg = getattr(self.config.actor_rollout_ref, "rollout", None) or {}
        eval_rollout_max_steps = _get_positive_int_attr(
            rollout_cfg, "eval_max_steps", 0
        )
        for idx, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["evaluation_keep"] = np.full(len(test_batch), True, dtype=object)

            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "global_steps": global_steps,
            }
            if eval_rollout_max_steps > 0:
                test_batch.meta_info["max_steps"] = eval_rollout_max_steps

            # Preserve every requested trial, including an incomplete final batch.
            original_count = len(test_batch)
            world_size = int(self.actor_rollout_wg.world_size)
            padding = (-original_count) % world_size
            dispatch_batch = test_batch
            if padding:
                duplicates = test_batch.slice(torch.arange(padding) % original_count)
                duplicates.non_tensor_batch["evaluation_keep"] = np.full(padding, False, dtype=object)
                dispatch_batch = DataProto.concat([test_batch, duplicates])
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(dispatch_batch)
            if test_output_gen_batch is None or len(test_output_gen_batch) != len(dispatch_batch):
                raise RuntimeError("Evaluation must return one result per dispatched trial")
            if padding:
                test_output_gen_batch = test_output_gen_batch.slice(slice(0, original_count))
            metric_dict["validation/padding_rollouts"] = (
                metric_dict.get("validation/padding_rollouts", 0) + padding
            )
            strict_validate_rollout = bool(
                getattr(self.config.trainer, "strict_validate_rollout", False)
            )
            if (
                test_output_gen_batch is not None
                and getattr(test_output_gen_batch, "batch", None) is not None
                and len(test_output_gen_batch) > 0
            ):
                batch = test_output_gen_batch.batch
                device = next(iter(batch.values())).device
                valid_mask = torch.ones(
                    len(test_output_gen_batch), dtype=torch.bool, device=device
                )
                if "is_dummy" in batch:
                    valid_mask &= DataProtoFilter._per_sample_has_real_step(
                        batch["is_dummy"], len(test_output_gen_batch), device
                    )
                valid_tokens = DataProtoFilter._per_sample_valid_response_tokens(
                    batch=batch,
                    B=len(test_output_gen_batch),
                    device=device,
                    action_token_len=int(
                        getattr(
                            self.config.actor_rollout_ref.model, "action_token_len", 1
                        )
                    ),
                )
                valid_mask &= valid_tokens > 0
                valid_mask_cpu = valid_mask.detach().cpu()
                total_rollout_count += int(valid_mask_cpu.numel())
                invalid_rollout_count += int((~valid_mask_cpu).sum().item())
                if not bool(valid_mask.all().detach().item()):
                    invalid_idx = (
                        (~valid_mask)
                        .nonzero(as_tuple=False)
                        .view(-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    reasons = []
                    if hasattr(test_output_gen_batch, "non_tensor_batch"):
                        raw_reasons = test_output_gen_batch.non_tensor_batch.get(
                            "placeholder_reason", None
                        )
                        if raw_reasons is not None:
                            raw_reasons = np.asarray(raw_reasons, dtype=object)
                            reasons = [
                                str(raw_reasons[i])
                                for i in invalid_idx[:8]
                                if i < len(raw_reasons)
                            ]
                    message = (
                        "[validation rollout] produced samples with no real valid "
                        "response tokens. "
                        f"invalid_count={len(invalid_idx)}/{len(test_output_gen_batch)}, "
                        f"invalid_indices={invalid_idx[:8]}, reasons={reasons}."
                    )
                    if strict_validate_rollout:
                        raise RuntimeError(
                            "[MERL strict] "
                            + message
                            + " Check LIBERO rendering/env init before trusting test_score."
                        )
                    print(
                        message + " Dropping invalid validation samples and continuing."
                    )
                    if int(valid_mask_cpu.sum().item()) == 0:
                        print(
                            "[validation rollout] all samples in this validation batch are invalid; skip this batch.",
                            flush=True,
                        )
                        continue
                    test_output_gen_batch = test_output_gen_batch.slice(valid_mask_cpu)
                    test_batch = test_batch.slice(valid_mask_cpu)
            print(
                f"[ray_trainer eval]: Finshed one batch generation {idx+1}/{len(self.val_dataloader)}."
            )
            print("validation generation end")

            test_batch = union_prompt_and_rollout_output(
                test_batch,
                test_output_gen_batch,
                context="_validate",
            )
            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            verifier_score, reward_metrics, format_metrics, reward_format_metrics = (
                self.val_reward_fn.verify(test_batch)
            )
            reward_tensor = torch.tensor(verifier_score, dtype=torch.float32).unsqueeze(
                -1
            )
            if "env_complete" in test_batch.batch:
                success_tensor = (
                    _first_sample_column(
                        test_batch.batch["env_complete"], len(test_batch)
                    )
                    .detach()
                    .cpu()
                    .to(dtype=torch.float32)
                )
            elif "complete" in test_batch.batch:
                success_tensor = (
                    _first_sample_column(test_batch.batch["complete"], len(test_batch))
                    .detach()
                    .cpu()
                    .to(dtype=torch.float32)
                )
            else:
                # Backward-compatible fallback for non-robotic reward managers.
                success_tensor = (reward_tensor.view(-1).detach().cpu() > 0.5).to(
                    dtype=torch.float32
                )
                metric_dict["validation/success_from_score_fallback"] = 1.0

            for k, v in reward_metrics.items():
                metric_dict["test_reward/" + k] = v

            for k, v in format_metrics.items():
                metric_dict["format_acc/" + k] = v

            for k, v in reward_format_metrics.items():
                metric_dict["acc_wformat/" + k] = v
            reward_tensor_lst.append(reward_tensor)
            success_tensor_lst.append(success_tensor)
            # data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            # data_source_lst.append( [self.config.data.task_suite_name] * reward_tensor.shape[0])
            data_sources = test_batch.non_tensor_batch.get("data_source", None)
            if data_sources is None:
                data_sources = test_batch.non_tensor_batch.get(
                    "task_suite_name",
                    [self.config.data.task_suite_name] * reward_tensor.shape[0],
                )
            data_sources = np.asarray(data_sources, dtype=object).reshape(-1)
            if data_sources.shape[0] != int(reward_tensor.shape[0]):
                metric_dict["validation/data_source_mismatch_count"] = float(
                    abs(data_sources.shape[0] - int(reward_tensor.shape[0]))
                )
                data_sources = np.asarray(
                    [self.config.data.task_suite_name] * int(reward_tensor.shape[0]),
                    dtype=object,
                )
            data_source_lst.append(data_sources)

        if len(reward_tensor_lst) == 0:
            task_suite_name = str(getattr(self.config.data, "task_suite_name", "all"))
            return {
                f"test_score/{task_suite_name}": 0.0,
                "test_score/all": 0.0,
                f"success_rate/{task_suite_name}": 0.0,
                "success_rate/all": 0.0,
                f"success_count/{task_suite_name}": 0.0,
                "success_count/all": 0.0,
                f"num_trials/{task_suite_name}": 0.0,
                "num_trials/all": 0.0,
                "validation/invalid_rollout_count": float(invalid_rollout_count),
                "validation/total_rollout_count": float(total_rollout_count),
                "validation/invalid_rollout_ratio": (
                    1.0 if total_rollout_count > 0 else 0.0
                ),
                "validation/skipped_all_batches": 1.0,
            }

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        success_tensor = torch.cat(success_tensor_lst, dim=0).view(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_success = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_success[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_success[data_source].append(success_tensor[i].item())

        metric_dict = dict(metric_dict)
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"test_score/{data_source}"] = np.mean(rewards)
            successes = data_source_success[data_source]
            metric_dict[f"success_rate/{data_source}"] = float(np.mean(successes))
            metric_dict[f"success_count/{data_source}"] = float(np.sum(successes))
            metric_dict[f"num_trials/{data_source}"] = float(len(successes))

        metric_dict[f"test_score/all"] = reward_tensor.mean().item()
        metric_dict["success_rate/all"] = float(success_tensor.mean().item())
        metric_dict["success_count/all"] = float(success_tensor.sum().item())
        metric_dict["num_trials/all"] = float(success_tensor.numel())
        metric_dict["validation/invalid_rollout_count"] = float(invalid_rollout_count)
        metric_dict["validation/total_rollout_count"] = float(total_rollout_count)
        metric_dict["validation/invalid_rollout_ratio"] = (
            float(invalid_rollout_count) / float(total_rollout_count)
            if total_rollout_count > 0
            else 0.0
        )

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        # Apply resume path overrides before any worker loads model weights.
        self._prepare_resume_for_worker_init()
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )  # * defined in main_ppo.py, i.e., RobActorRolloutRefWorker
            self.resource_pool_to_cls[resource_pool][
                "actor_rollout"
            ] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == "gae":
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in ["rloo"]:
            self.use_critic = False
        elif self.config.algorithm.adv_estimator in ["grpo"]:
            self.use_critic = False
        elif self.config.algorithm.adv_estimator in ["reinforce_plus_plus"]:
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        self._restore_policy_from_resume()
        if self.config.actor_rollout_ref.world_model.enable:
            # Keep rollout-side WM consistent with resumed checkpoint, even when no wm_trainer is used.
            self._sync_world_model_from_resume(load_into_wm_trainer=False)

        #!!! create wm trainer worker !!!
        train_mode = str(getattr(self.config.trainer, "train_mode", "MERL")).upper()
        if self.config.trainer.get("val_only", False) or train_mode == "MFRL":
            return

        if bool(getattr(self.config.actor_rollout_ref.world_model, "enable", False)):
            self._ensure_wm_trainer_initialized()

    def fit(self):
        """
        The training loop of VLA-RL.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            local_dir=self.config.trainer.default_local_dir,
            wandb_mode=self.config.trainer.wandb_mode,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # Runtime resume state (if any) is prepared in init_workers.
        # For MFRL we recover progress/checkpoint pointers, and continue from
        # epoch boundary (coarse-grained recovery by design).
        resume_state = (
            self._resume_state if isinstance(self._resume_state, dict) else {}
        )
        start_epoch = 0
        global_steps = 0
        if len(resume_state) > 0:
            start_epoch = max(0, self._safe_int(resume_state.get("epoch", 0), 0))
            global_steps = max(0, self._safe_int(resume_state.get("global_step", 0), 0))
            print(
                "[resume] Continue fit (MFRL) from state:",
                {"epoch": start_epoch, "global_step": global_steps},
            )

        dp_size = (
            self.actor_rollout_wg.world_size
            // self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        )
        batch_size = self.config.data.train_batch_size
        n_samples = self.config.data.n_samples
        latest_actor_ckpt_dir = (
            resume_state.get("actor_ckpt_dir", None) if len(resume_state) > 0 else None
        )
        latest_critic_ckpt_dir = (
            resume_state.get("critic_ckpt_dir", None) if len(resume_state) > 0 else None
        )

        # peform rollout before training
        if self.config.trainer.get("rollout_before_train", False):
            print("Mode: Rollout before Train...")
            self._save_rollouts(
                global_steps=global_steps,
                rollout_epoch=self.config.trainer.get("sim_rollout_epoch", 1000),
                use_wm=False,
            )
            return

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", False
        ):
            print("Mode: Validate before Train...")
            val_metrics = self._validate(global_steps=global_steps)
            print("[ray_trainer eval]: Get all val_metrics!")
            val_metrics = {f"val/{key}": val for key, val in val_metrics.items()}
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=global_steps)
            if self.config.trainer.get("val_only", False):
                return

        print("################### Start Training Now ###################")
        self._training_started = time.monotonic()
        # update_wm = self.config.actor_rollout_ref.world_model.get("fine_tune", False)
        # skip_first_rollout = False
        # skip_first_wm_update = False
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            if self._training_step_limit_reached(global_steps):
                break
            print(f"[Epoch] Start epoch: {epoch} / {self.config.trainer.total_epochs}")

            self.train_dataloader.start_new_epoch()
            while not self._training_step_limit_reached(global_steps):
                valid_batch = []
                buffer_batch = []
                if self.train_dataloader.buffer_size() > 0:
                    buffer_batch = self.train_dataloader.get_from_buffer(
                        batch_size, self.actor_rollout_wg.world_size
                    )

                metrics = defaultdict(list)
                metrics["timing/gen"] = 0
                metrics["timing/verify"] = 0
                metrics["timing/acc&trunc_filter"] = 0
                metrics["timing/filter_format_error"] = 0
                metrics["timing/compute_all_entropy"] = 0

                print("1. Genrating batches")
                print(f"batch size: {batch_size}, sample nums: {n_samples}")
                while len(valid_batch) < batch_size * n_samples:
                    print(
                        f"[Batch] len of valid_batch: {len(valid_batch)}, expect: {batch_size * n_samples}"
                    )
                    try:
                        batch_dict = self.train_dataloader.get_next_batch()
                    except StopIteration:
                        break

                    ## Generate rollout batch
                    # generate a batch
                    with Timer(
                        name="gen", text="{name}: {seconds:.1f} seconds"
                    ) as timer:
                        newbatch: DataProto = DataProto.from_single_dict(batch_dict)

                        if len(buffer_batch) > 0:
                            newbatch = DataProto.concat([buffer_batch, newbatch])
                            buffer_batch = []

                        if "robotwin" in self.config.data.task_suite_name:
                            gen_batch = newbatch.select(
                                batch_keys=["task_id", "trial_id", "trial_seed"],
                                non_tensor_batch_keys={"task_suite_name"},
                                meta_info_keys={},
                            )
                        else:
                            gen_batch = newbatch.select(
                                batch_keys=["task_id", "trial_id"],
                                non_tensor_batch_keys={"task_suite_name"},
                                meta_info_keys={},
                            )
                        newbatch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(newbatch.batch))],
                            dtype=object,
                        )

                        batch_lst = sum(
                            [
                                [newbatch[i : i + 1] for _ in range(n_samples)]
                                for i in range(len(newbatch))
                            ],
                            [],
                        )
                        gen_batch.meta_info = {
                            "eos_token_id": self.tokenizer.eos_token_id,
                            "n_samples": n_samples,
                            "pad_token_id": self.tokenizer.pad_token_id,
                            "global_steps": global_steps,
                        }
                        train_max_steps = _get_positive_int_attr(
                            self.config.actor_rollout_ref.rollout, "train_max_steps", 0
                        )
                        if train_max_steps > 0:
                            gen_batch.meta_info["max_steps"] = train_max_steps

                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            prompts=gen_batch
                        )
                        try:
                            roll_batch = DataProto.concat(batch_lst)
                            roll_batch = union_prompt_and_rollout_output(
                                roll_batch,
                                gen_batch_output,
                                context="fit.real_rollout",
                            )
                        finally:
                            del gen_batch_output
                            # torch.cuda.empty_cache()
                            gc.collect()

                    metrics["timing/gen"] += timer.last
                    print("[Batch] Finshed one batch generation! To validate it.")

                    ## Verify and score the rollout results
                    with Timer(
                        name="verify", text="{name}: {seconds:.1f} seconds"
                    ) as timer:
                        (
                            scores_tensor,
                            reward_metrics,
                            format_metrics,
                            reward_format_metrics,
                        ) = self.reward_fn.verify(roll_batch)
                        for k, v in reward_metrics.items():
                            metrics["train_verify_score/" + k].append(v)
                        for k, v in format_metrics.items():
                            metrics["format_score/" + k].append(v)
                        for k, v in reward_format_metrics.items():
                            metrics["train_verify_score_wo_format/" + k].append(v)
                    metrics["timing/verify"] += timer.last

                    ## Filter by accuracy and truncation
                    # do accuracy filtering and score logging
                    with Timer(
                        name="acc&trunc_filter", text="{name}: {seconds:.1f} seconds"
                    ) as timer:
                        if (
                            self.config.data.filter_accuracy
                            or self.config.data.filter_truncated
                        ):
                            print(f"[Batch] before filtering: {len(roll_batch)}")
                            filtered_roll_batch = self.filter(
                                roll_batch.batch["acc"].unsqueeze(1),
                                roll_batch,
                                n_samples,
                            )
                            print(
                                f"[Batch] after filtering: {len(filtered_roll_batch)}"
                            )
                            roll_batch_to_add = (
                                filtered_roll_batch
                                if len(filtered_roll_batch) > 0
                                else roll_batch
                            )
                        else:
                            roll_batch_to_add = roll_batch
                    metrics["timing/acc&trunc_filter"] += timer.last

                    if len(valid_batch) == 0:
                        valid_batch = roll_batch_to_add
                    else:
                        valid_batch = DataProto.concat([valid_batch, roll_batch_to_add])
                    print(
                        f"[Batch] Collected {len(valid_batch)} / {batch_size * n_samples} rollouts and each prompt has {n_samples} samples."
                    )

                ## Check if collected enough data for a full batch
                if len(valid_batch) < batch_size * n_samples:
                    print(
                        f"[Epoch] len of valid_batch: {len(valid_batch)}, expect: {batch_size * n_samples}, discrad these batches and close this epoch."
                    )
                    break
                elif len(valid_batch) > batch_size * n_samples:
                    valid_batch = self.add_to_buffer(valid_batch, batch_size, n_samples)
                    print(
                        f"[Epoch] len of valid_batch: {len(valid_batch)}, expect: {batch_size * n_samples}, reorg these batches."
                    )
                else:
                    print(
                        f"[Epoch] len of valid_batch: {len(valid_batch)}, expect: {batch_size * n_samples}, start training these batches."
                    )

                ## Calculate average metrics
                print("2. Calculating Average Metrics")
                for k, v in reward_metrics.items():
                    metrics["train_verify_score/" + k] = np.mean(
                        metrics["train_verify_score/" + k]
                    )
                for k, v in format_metrics.items():
                    metrics["format_score/" + k] = np.mean(metrics["format_score/" + k])
                for k, v in reward_format_metrics.items():
                    metrics["train_verify_score_wo_format/" + k] = np.mean(
                        metrics["train_verify_score_wo_format/" + k]
                    )

                batch = valid_batch
                print(f"rollout batch size: {len(batch)}")

                ## Compute reference policy log probabilities
                # compute reference log_prob
                if self.use_reference_policy:
                    with Timer(
                        name="ref", text="{name}: {seconds:.1f} seconds"
                    ) as timer:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                    metrics["timing/ref"] = timer.last

                ## Compute reward scores
                with Timer(
                    name="reward", text="{name}: {seconds:.1f} seconds"
                ) as timer:
                    if self.use_rm:
                        print("Not implement yet")
                        raise ValueError
                        # batch.meta_info['n_samples'] = n_samples
                        # reward_model_tensor= self.rm_wg.compute_rm_score(batch)
                        # if 'metrics' in reward_model_tensor.meta_info:
                        #     reward_model_metrics = reduce_metrics(reward_model_tensor.meta_info.pop('metrics'))
                        #     metrics.update(reward_model_metrics)
                        # batch = batch.union(reward_model_tensor)
                metrics["timing/reward_model"] = timer.last

                ## Compute GAE advantages
                print("3. Calculating GAE")
                with Timer(name="adv", text="{name}: {seconds:.1f} seconds") as timer:
                    # directly reuse previously computed rewards; but with reward shaping
                    reward_tensor_dict, reward_metrics = self.reward_fn(batch)
                    batch.batch["token_level_scores"] = reward_tensor_dict["all"]
                    for k, v in reward_metrics.items():
                        metrics["train_reward/" + k] = v
                    # decomposed rewards:
                    for k, v in reward_tensor_dict.items():
                        batch.batch[k] = v

                    # compute rewards. apply_kl_penalty if available
                    batch, kl_metrics = apply_kl_penalty(
                        batch,
                        kl_ctrl=self.kl_ctrl,
                        kl_penalty=self.config.algorithm.kl_penalty,
                        action_token_len=self.config.actor_rollout_ref.model.action_token_len,
                        action_chunks_len=self.config.actor_rollout_ref.model.action_chunks_len,
                        config=self.config,
                    )
                    metrics.update(kl_metrics)

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        self.config.algorithm.gamma,
                        self.config.algorithm.lam,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        config=self.config,
                    )
                    actor_contract_stats = inject_actor_token_contract(
                        batch,
                        action_token_len=self.config.actor_rollout_ref.model.action_token_len,
                        strict=False,
                        context="fit.actor_update",
                    )
                    metrics["actor_input/real_token_count"] = float(
                        actor_contract_stats.get("real_token_count", 0.0)
                    )
                    metrics["actor_input/imag_token_count"] = float(
                        actor_contract_stats.get("imag_token_count", 0.0)
                    )
                    metrics["actor_input/imag_weight_mean"] = float(
                        actor_contract_stats.get("imag_weight_mean", 0.0)
                    )
                metrics["timing/adv"] = timer.last

                # critic is disabled

                ## Update actor model
                print("4. Updating Acotr Model")
                # implement critic warmup
                if self.config.trainer.critic_warmup <= global_steps:
                    # update actor
                    with Timer(
                        name="update_actor", text="{name}: {seconds:.1f} seconds"
                    ) as timer:
                        batch.meta_info["is_filtered"] = True
                        batch.meta_info["train_mode"] = False
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        entropy_output = self.actor_rollout_wg.compute_entropy(
                            data=batch
                        )
                    metrics["timing/update_actor"] = timer.last
                    actor_output_metrics = reduce_metrics(
                        actor_output.meta_info["metrics"]
                    )
                    entropy_output_metrics = reduce_metrics(
                        entropy_output.meta_info["metrics"]
                    )
                    metrics.update(actor_output_metrics)
                    metrics.update(entropy_output_metrics)

                ## Validate model performance
                print("5. Validating Model")
                # validate
                if (
                    self.val_reward_fn is not None
                    and (global_steps + 1) % self.config.trainer.test_freq == 0
                ):
                    with Timer(
                        name="testing", text="{name}: {seconds:.1f} seconds"
                    ) as timer:
                        val_metrics: dict = self._validate(
                            global_steps=global_steps + 1
                        )
                        val_metrics = {
                            f"val/{key}": val for key, val in val_metrics.items()
                        }
                    metrics["timing/testing"] = timer.last
                    metrics.update(val_metrics)

                ## Collect and log metrics
                print("6. Collecting Metrics")
                # collect metrics
                with Timer(
                    name="logging1", text="{name}: {seconds:.1f} seconds"
                ) as timer:
                    data_metrics = compute_data_metrics(batch=batch, config=self.config)
                with Timer(
                    name="logging2", text="{name}: {seconds:.1f} seconds"
                ) as timer:
                    metrics.update(data_metrics)
                with Timer(
                    name="logging3", text="{name}: {seconds:.1f} seconds"
                ) as timer:
                    # Explicit progress fields make log-based resume robust.
                    metrics["train/epoch"] = int(epoch)
                    metrics["train/global_step"] = int(global_steps)
                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=global_steps)

                checkpoint_saved = False
                if (
                    self.config.trainer.save_freq > 0
                    and (global_steps + 1) % self.config.trainer.save_freq == 0
                ):
                    actor_ckpt_this_step = os.path.join(
                        self.config.trainer.default_local_dir,
                        "actor",
                        f"global_step_{global_steps}",
                    )
                    actor_remote_path = None  # if self.config.trainer.default_hdfs_dir is None else os.path.join(
                    # self.config.trainer.default_hdfs_dir, 'actor')
                    self.actor_rollout_wg.save_checkpoint(
                        actor_ckpt_this_step, actor_remote_path
                    )
                    print(f"Saved ckpt step_{global_steps} into {actor_ckpt_this_step}")
                    latest_actor_ckpt_dir = actor_ckpt_this_step
                    max_saved_actor_checkpoints = max(
                        1,
                        int(
                            getattr(
                                self.config.trainer,
                                "max_saved_actor_checkpoints",
                                2,
                            )
                        ),
                    )
                    removed_actor_ckpts = self._cleanup_old_component_checkpoints(
                        component="actor",
                        keep=max_saved_actor_checkpoints,
                        protected_paths=[latest_actor_ckpt_dir],
                    )
                    if len(removed_actor_ckpts) > 0:
                        print(
                            f"[checkpoint] Removed old actor checkpoints: {removed_actor_ckpts}"
                        )
                    checkpoint_saved = True

                    if self.use_critic:
                        critic_ckpt_this_step = os.path.join(
                            self.config.trainer.default_local_dir,
                            "critic",
                            f"global_step_{global_steps}",
                        )
                        critic_remote_path = None  # if self.config.trainer.default_hdfs_dir is None else os.path.join(
                        # self.config.trainer.default_hdfs_dir, 'critic')
                        self.critic_wg.save_checkpoint(
                            critic_ckpt_this_step, critic_remote_path
                        )
                        latest_critic_ckpt_dir = critic_ckpt_this_step
                        max_saved_critic_checkpoints = max(
                            1,
                            int(
                                getattr(
                                    self.config.trainer,
                                    "max_saved_critic_checkpoints",
                                    1,
                                )
                            ),
                        )
                        removed_critic_ckpts = self._cleanup_old_component_checkpoints(
                            component="critic",
                            keep=max_saved_critic_checkpoints,
                            protected_paths=[latest_critic_ckpt_dir],
                        )
                        if len(removed_critic_ckpts) > 0:
                            print(
                                f"[checkpoint] Removed old critic checkpoints: {removed_critic_ckpts}"
                            )
                    if self.use_rm:
                        prm_local_path = os.path.join(
                            self.config.trainer.default_local_dir,
                            "prm",
                            f"global_step_{global_steps}",
                        )
                        prm_remote_path = None  # if self.config.trainer.default_hdfs_dir is None else os.path.join(
                        # self.config.trainer.default_hdfs_dir, 'critic')
                        self.rm_wg.save_checkpoint(prm_local_path, prm_remote_path)

                if checkpoint_saved:
                    # MFRL has no dedicated WM checkpoint.
                    self._write_resume_state(
                        epoch=epoch,
                        global_step=global_steps,
                        actor_ckpt_dir=latest_actor_ckpt_dir,
                        critic_ckpt_dir=latest_critic_ckpt_dir,
                        world_model_ckpt_dir=None,
                        save_snapshot=True,
                    )

                global_steps += 1

        # perform validation after training
        if self.val_reward_fn is not None and self.config.trainer.get("final_val_after_train", True):
            val_metrics = self._validate(global_steps=global_steps)
            val_metrics = {f"val/{key}": val for key, val in val_metrics.items()}
            pprint(f"Final validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=global_steps)

        # Final refresh even if last step is not on save boundary.
        # MFRL does not maintain world_model.pth.
        self._write_resume_state(
            epoch=self.config.trainer.total_epochs,
            global_step=global_steps,
            actor_ckpt_dir=latest_actor_ckpt_dir,
            critic_ckpt_dir=latest_critic_ckpt_dir,
            world_model_ckpt_dir=None,
            save_snapshot=False,
        )

    def fit_wm_v5(self):
        """
        WM-aware training loop (version 5) with:
        - train_mode switch: "MBRL"/"MFRL"/"MERL"
        - uid ensure patch: ensure every DataProto used downstream has non_tensor_batch["uid"]
        - is_dummy filtering: completely filter out dummy data before PPO training
        - robust error handling: never crash due to empty/invalid batch

        New ratio scheduler:
        - use WM training total loss only for logging: wm/loss, wm/loss_ema
        - use WM visual prediction error proxy (wm_ratio_signal = loss_noise) for ratio scheduling
        - monotone target mapping:
            r_star = (1 - min_real_ratio) / (1 + (e_ema / tau)^gamma)
        - conservative asymmetric tracking:
            upward: slow, only accelerated by positive improvement
            downward: faster rollback for safety
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        import uuid
        import numpy as np

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            local_dir=self.config.trainer.default_local_dir,
            wandb_mode=self.config.trainer.wandb_mode,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # WM config
        wm_cfg = getattr(self.config.actor_rollout_ref, "world_model", None) or {}
        rollout_cfg = getattr(self.config.actor_rollout_ref, "rollout", None) or {}
        train_rollout_max_steps = _get_positive_int_attr(
            rollout_cfg, "train_max_steps", 0
        )
        eval_rollout_max_steps = _get_positive_int_attr(
            rollout_cfg, "eval_max_steps", 0
        )

        # ---- training / logging related ----
        wm_loss_smooth_alpha = float(
            getattr(wm_cfg, "wm_loss_smooth_alpha", 0.50)
        )  # total WM loss EMA (for logging only)
        min_real_ratio = float(getattr(wm_cfg, "min_real_ratio", 0.01))
        max_wm_ratio = float(getattr(wm_cfg, "max_wm_ratio", 0.25))
        wm_warmup_steps = int(getattr(wm_cfg, "wm_warmup_steps", 0))
        wm_training_steps_per_epoch = int(
            getattr(wm_cfg, "training_steps_per_epoch", 100)
        )
        wm_inner_steps = int(
            getattr(
                self.config.actor_rollout_ref.world_model,
                "wm_inner_steps",
                min(100, wm_training_steps_per_epoch),
            )
        )
        wm_inner_steps = max(1, min(wm_inner_steps, wm_training_steps_per_epoch))
        save_wm_on_update = getattr(wm_cfg, "save_on_update", True)
        save_freq_wm_inner = int(getattr(wm_cfg, "save_freq_wm_inner", 1))
        save_freq_wm_outer = int(getattr(wm_cfg, "save_freq_wm_outer", 100))
        wm_eval_interval = int(getattr(wm_cfg, "wm_eval_interval", 10))
        fixed_eval_root = str(getattr(wm_cfg, "fixed_eval_root", "") or "").strip()
        fixed_eval_enabled = bool(
            getattr(wm_cfg, "fixed_eval_enabled", False)
        ) and bool(fixed_eval_root)
        persist_imag_rollout_shards = bool(
            getattr(wm_cfg, "persist_imag_rollout_shards", False)
        )
        min_real_ratio = min(max(min_real_ratio, 0.0), 1.0)
        max_wm_ratio = min(max(max_wm_ratio, 0.0), 1.0 - min_real_ratio)

        # ---- current-batch confidence driven MERL scheduling ----
        imag_ratio_min = float(getattr(wm_cfg, "imag_ratio_min", 0.10))
        imag_ratio_max = float(getattr(wm_cfg, "imag_ratio_max", max_wm_ratio))
        imag_ratio_gamma = float(getattr(wm_cfg, "imag_ratio_gamma", 1.5))
        imag_confidence_ema_alpha = float(
            getattr(wm_cfg, "imag_confidence_ema_alpha", 0.80)
        )
        imag_horizon_min = int(getattr(wm_cfg, "imag_horizon_min", 128))
        imag_horizon_max = int(getattr(wm_cfg, "imag_horizon_max", 192))
        imag_obs_error_scale = float(getattr(wm_cfg, "imag_obs_error_scale", 8.0))
        imag_done_error_scale = float(getattr(wm_cfg, "imag_done_error_scale", 2.0))
        imag_weight_min = float(getattr(wm_cfg, "imag_weight_min", 0.0))
        imag_weight_eta = float(getattr(wm_cfg, "imag_weight_eta", 1.5))
        wm_sample_weight_max = float(getattr(wm_cfg, "wm_sample_weight_max", 0.25))
        imag_priority_eps = float(getattr(wm_cfg, "imag_priority_eps", 1e-3))
        imag_priority_beta = float(getattr(wm_cfg, "imag_priority_beta", 1.0))
        wm_sample_weight_max = min(max(wm_sample_weight_max, imag_weight_min), 1.0)

        imag_ratio_max = min(max(imag_ratio_max, 0.0), max_wm_ratio)
        imag_ratio_min = min(max(imag_ratio_min, 0.0), imag_ratio_max)
        imag_horizon_min = max(1, imag_horizon_min)
        imag_horizon_max = max(imag_horizon_min, imag_horizon_max)
        print(
            "[rollout horizon] "
            f"train_max_steps={train_rollout_max_steps or 'default'}, "
            f"eval_max_steps={eval_rollout_max_steps or 'default'}, "
            f"imag_horizon=[{imag_horizon_min}, {imag_horizon_max}]",
            flush=True,
        )

        # ---- new ratio scheduler related ----
        # tau: reference error level, where target ratio is about half of max usable imagined ratio
        wm_ratio_ref_loss = float(
            getattr(
                wm_cfg, "wm_ratio_ref_loss", getattr(wm_cfg, "wm_target_loss", 0.25)
            )
        )
        # gamma: curvature of monotone inverse-power mapping
        wm_ratio_gamma = float(
            getattr(wm_cfg, "wm_ratio_gamma", getattr(wm_cfg, "wm_mapping_gamma", 2.0))
        )
        wm_ratio_rounding = str(getattr(wm_cfg, "wm_ratio_rounding", "carry"))
        if wm_ratio_rounding not in ("stochastic", "floor", "ceil", "carry"):
            print(
                f"[WM RATIO] unknown wm_ratio_rounding={wm_ratio_rounding}; "
                "fallback to carry.",
                flush=True,
            )
            wm_ratio_rounding = "carry"
        # upward tracking rate: slow and conservative
        wm_ratio_up_min = float(getattr(wm_cfg, "wm_ratio_up_min", 0.005))
        wm_ratio_up_max = float(getattr(wm_cfg, "wm_ratio_up_max", 0.02))
        # downward rollback rate: faster for safety
        wm_ratio_down_max = float(getattr(wm_cfg, "wm_ratio_down_max", 0.05))
        # positive-improvement clipping range
        wm_ratio_improve_clip = float(getattr(wm_cfg, "wm_ratio_improve_clip", 0.10))
        # EMA for ratio signal itself
        wm_ratio_signal_smooth_alpha = float(
            getattr(wm_cfg, "wm_ratio_signal_smooth_alpha", wm_loss_smooth_alpha)
        )
        wm_ratio_anchor_coverage_threshold = float(
            getattr(wm_cfg, "wm_ratio_anchor_coverage_threshold", 0.60)
        )
        wm_ratio_health_min_slope = float(
            getattr(wm_cfg, "wm_ratio_health_min_slope", 0.15)
        )
        wm_ratio_cooldown_steps = max(
            0, int(getattr(wm_cfg, "wm_ratio_cooldown_steps", 3))
        )
        wm_ratio_realized_min = float(
            getattr(wm_cfg, "wm_ratio_realized_min", 0.50)
        )
        wm_ratio_critic_kl_limit = float(
            getattr(wm_cfg, "wm_ratio_critic_kl_limit", 0.40)
        )
        wm_ratio_critic_reward_min = float(
            getattr(wm_cfg, "wm_ratio_critic_reward_min", 0.0)
        )
        wm_horizon_growth_window = max(
            1, int(getattr(wm_cfg, "wm_horizon_growth_window", 10))
        )
        wm_ratio_anchor_coverage_threshold = max(
            wm_ratio_anchor_coverage_threshold, 1e-6
        )
        wm_ratio_health_min_slope = float(
            np.clip(wm_ratio_health_min_slope, 0.0, 1.0)
        )
        wm_ratio_realized_min = float(np.clip(wm_ratio_realized_min, 0.0, 1.0))

        if not hasattr(self, "_wm_loss_ema"):
            self._wm_loss_ema = None
        if not hasattr(self, "_wm_ratio_signal_ema"):
            self._wm_ratio_signal_ema = None
        if not hasattr(self, "_last_wm_ratio_signal_ema"):
            self._last_wm_ratio_signal_ema = None
        if not hasattr(self, "_r_wm"):
            self._r_wm = 0.0
        if not hasattr(self, "_imag_confidence_ema"):
            self._imag_confidence_ema = None
        if not hasattr(self, "_imagined_horizon"):
            self._imagined_horizon = imag_horizon_min
        if not hasattr(self, "_wm_ratio_carry"):
            self._wm_ratio_carry = float(
                max(0.0, getattr(wm_cfg, "wm_ratio_carry_init", 0.0))
            )
        if not hasattr(self, "_wm_weak_update_active"):
            self._wm_weak_update_active = False
        if not hasattr(self, "_wm_weak_update_last_calibration_step"):
            self._wm_weak_update_last_calibration_step = None
        if not hasattr(self, "_last_wm_actor_anchor_coverage"):
            self._last_wm_actor_anchor_coverage = 1.0
        if not hasattr(self, "_last_actor_ppo_kl_hard_skip_ratio"):
            self._last_actor_ppo_kl_hard_skip_ratio = 0.0
        if not hasattr(self, "_last_actor_ppo_kl_hard_max"):
            self._last_actor_ppo_kl_hard_max = 0.0
        if not hasattr(self, "_actor_lr_health_scale"):
            self._actor_lr_health_scale = 1.0
        if not hasattr(self, "_wm_ratio_health_metrics"):
            self._wm_ratio_health_metrics = {}
        if not hasattr(self, "_wm_horizon_health_metrics"):
            self._wm_horizon_health_metrics = {}
        if not hasattr(self, "_wm_ratio_cooldown_remaining"):
            self._wm_ratio_cooldown_remaining = 0
        if not hasattr(self, "_last_wm_ratio_realized_vs_target"):
            self._last_wm_ratio_realized_vs_target = 1.0
        if not hasattr(self, "_last_wm_target_sample_count"):
            self._last_wm_target_sample_count = 0.0
        if not hasattr(self, "_last_critic_kl"):
            self._last_critic_kl = 0.0
        if not hasattr(self, "_last_critic_reward_mean"):
            self._last_critic_reward_mean = 0.0
        if not hasattr(self, "_merl_policy_health_history"):
            self._merl_policy_health_history = []

        # training mode: MBRL / MFRL / MERL
        train_mode = str(getattr(self.config.trainer, "train_mode", "MERL")).upper()
        if train_mode not in ("MBRL", "MFRL", "MERL", "ONLINE_MBRL"):
            print(
                f"[fit_wm_v5] Unknown train_mode='{train_mode}', falling back to 'MERL'"
            )
            train_mode = "MERL"
        print(f"train_mode: {train_mode}")
        if train_mode == "ONLINE_MBRL":
            from merl.modes import validate_online_mbrl
            validate_online_mbrl(wm_cfg)
            self._imagined_horizon = imag_horizon_min
            self._actor_lr_health_scale = 1.0
        strict_mode_assert = bool(
            getattr(self.config.trainer, "strict_mode_assert", train_mode == "MERL")
        )
        merl_imagined_reward_hard_constraint = bool(
            getattr(
                wm_cfg,
                "merl_imagined_reward_hard_constraint",
                train_mode == "MERL",
            )
        )
        merl_imagined_reward_confidence = float(
            getattr(wm_cfg, "merl_imagined_reward_confidence", 0.10)
        )
        merl_imagined_reward_weight_cap = float(
            getattr(
                wm_cfg,
                "merl_imagined_reward_weight_cap",
                merl_imagined_reward_confidence,
            )
        )
        merl_imagined_reward_confidence = min(
            max(merl_imagined_reward_confidence, 0.0), 1.0
        )
        merl_imagined_reward_weight_cap = min(
            max(merl_imagined_reward_weight_cap, 0.0), 1.0
        )
        imag_advantage_abs_clip = max(
            0.0, float(getattr(wm_cfg, "imag_advantage_abs_clip", 0.0))
        )
        wm_actor_anchor_reward_min = max(
            0.0, float(getattr(wm_cfg, "wm_actor_anchor_reward_min", 1.0))
        )
        use_wm_reward_proxy = bool(
            getattr(wm_cfg, "use_wm_reward_proxy", train_mode != "MERL")
        )
        wm_real_anchor_reward = bool(
            getattr(
                wm_cfg,
                "wm_real_anchor_reward",
                train_mode == "MERL" and (not use_wm_reward_proxy),
            )
        )
        require_wm_anchor_reward = bool(
            getattr(
                wm_cfg,
                "require_wm_anchor_reward",
                train_mode == "MERL" and wm_real_anchor_reward,
            )
        )
        zero_unanchored_wm_weight = bool(
            getattr(
                wm_cfg,
                "zero_unanchored_wm_weight",
                train_mode == "MERL" and wm_real_anchor_reward,
            )
        )
        wm_grpo_uid_mode = str(
            getattr(
                wm_cfg,
                "wm_grpo_uid_mode",
                "anchor" if train_mode == "MERL" and wm_real_anchor_reward else "source",
            )
            or "source"
        ).strip().lower()

        update_wm = self.config.actor_rollout_ref.world_model.get("fine_tune", False)
        update_wm_effective = bool(update_wm) and train_mode in ("MERL", "ONLINE_MBRL")
        wm_enabled = bool(getattr(wm_cfg, "enable", False))
        if wm_enabled and train_mode != "MFRL":
            self._ensure_wm_trainer_initialized()

        # Runtime resume state (if any) is prepared in init_workers.
        # Here we restore loop progress and WM ratio scheduler internals.
        resume_state = (
            self._resume_state if isinstance(self._resume_state, dict) else {}
        )
        start_epoch = 0
        global_steps = 0
        if len(resume_state) > 0:
            start_epoch = max(0, self._safe_int(resume_state.get("epoch", 0), 0))
            global_steps = max(0, self._safe_int(resume_state.get("global_step", 0), 0))

            restored_wm_loss_ema = self._safe_float(
                resume_state.get("wm_loss_ema", None), None
            )
            restored_wm_ratio_signal_ema = self._safe_float(
                resume_state.get("wm_ratio_signal_ema", None), None
            )
            restored_last_wm_ratio_signal_ema = self._safe_float(
                resume_state.get("last_wm_ratio_signal_ema", None), None
            )
            restored_r_wm = self._safe_float(resume_state.get("r_wm", None), None)
            restored_wm_weak_update_active = bool(
                resume_state.get("wm_weak_update_active", False)
            )
            restored_wm_weak_update_last_calibration_step = self._safe_int(
                resume_state.get("wm_weak_update_last_calibration_step", None),
                None,
            )

            if restored_wm_loss_ema is not None:
                self._wm_loss_ema = restored_wm_loss_ema
            if restored_wm_ratio_signal_ema is not None:
                self._wm_ratio_signal_ema = restored_wm_ratio_signal_ema
            if restored_last_wm_ratio_signal_ema is not None:
                self._last_wm_ratio_signal_ema = restored_last_wm_ratio_signal_ema
            if restored_r_wm is not None:
                self._r_wm = restored_r_wm
            if restored_wm_weak_update_active:
                self._wm_weak_update_active = True
            if restored_wm_weak_update_last_calibration_step is not None:
                self._wm_weak_update_last_calibration_step = (
                    restored_wm_weak_update_last_calibration_step
                )

            print(
                "[resume] Continue fit_wm_v5 from state:",
                {
                    "epoch": start_epoch,
                    "global_step": global_steps,
                    "wm_loss_ema": self._wm_loss_ema,
                    "wm_ratio_signal_ema": self._wm_ratio_signal_ema,
                    "last_wm_ratio_signal_ema": self._last_wm_ratio_signal_ema,
                    "r_wm": self._r_wm,
                    "wm_weak_update_active": self._wm_weak_update_active,
                    "wm_weak_update_last_calibration_step": self._wm_weak_update_last_calibration_step,
                },
            )

        total_epochs = int(getattr(self.config.trainer, "total_epochs", 0))
        if start_epoch > 0 and global_steps <= 0:
            print(
                "[resume] Ignore stale nonzero-epoch resume state with global_step=0; "
                "restart training loop from epoch 0 while keeping recoverable checkpoints/replay state.",
                flush=True,
            )
            start_epoch = 0
            if isinstance(resume_state, dict):
                resume_state["epoch"] = 0

        initial_global_steps = int(global_steps)
        batch_size = self.config.data.train_batch_size
        n_samples = self.config.data.n_samples
        actor_dp_world_size = max(
            1, int(getattr(self.actor_rollout_wg, "world_size", 1) or 1)
        )
        wm_rollout_n_samples_max = int(
            getattr(wm_cfg, "wm_rollout_n_samples_max", max(n_samples, 8))
        )
        wm_rollout_n_samples_max = max(1, wm_rollout_n_samples_max)
        real_prompt_min_for_wm = int(getattr(wm_cfg, "real_prompt_min_for_wm", 1))
        real_prompt_min_for_wm = max(0, min(real_prompt_min_for_wm, batch_size))
        wm_weak_update_enable = bool(getattr(wm_cfg, "weak_update_enable", True))
        wm_weak_update_ratio_threshold = float(
            getattr(wm_cfg, "weak_update_ratio_threshold", 0.95)
        )
        wm_weak_update_ratio_threshold = float(
            np.clip(wm_weak_update_ratio_threshold, 0.0, 1.0)
        )
        wm_weak_update_exit_ratio_threshold = float(
            getattr(
                wm_cfg,
                "weak_update_exit_ratio_threshold",
                max(0.0, wm_weak_update_ratio_threshold - 0.05),
            )
        )
        wm_weak_update_exit_ratio_threshold = float(
            np.clip(
                wm_weak_update_exit_ratio_threshold,
                0.0,
                wm_weak_update_ratio_threshold,
            )
        )
        wm_sparse_real_interval = max(
            1, int(getattr(wm_cfg, "wm_sparse_real_interval", 5))
        )
        wm_sparse_real_prompts = int(
            getattr(
                wm_cfg,
                "wm_sparse_real_prompts",
                max(1, real_prompt_min_for_wm),
            )
        )
        wm_sparse_real_prompts = max(0, min(wm_sparse_real_prompts, batch_size))
        if not wm_weak_update_enable:
            self._wm_weak_update_active = False
        elif (
            self._safe_float(getattr(self, "_r_wm", None), 0.0)
            >= wm_weak_update_ratio_threshold
        ):
            self._wm_weak_update_active = True
        latest_actor_ckpt_dir = (
            resume_state.get("actor_ckpt_dir", None) if len(resume_state) > 0 else None
        )
        latest_critic_ckpt_dir = (
            resume_state.get("critic_ckpt_dir", None) if len(resume_state) > 0 else None
        )
        latest_world_model_ckpt_dir = (
            resume_state.get("world_model_ckpt_dir", None)
            if len(resume_state) > 0
            else None
        )
        if train_mode in ("MBRL", "MERL", "ONLINE_MBRL"):
            self._restore_replay_pool_state(resume_state=resume_state)

        # helper utilities
        def _sample_n_from_pool(pool: DataProto, n: int):
            L = len(pool)
            if n <= 0 or L == 0:
                return DataProto.empty_like(pool)
            if n >= L:
                return pool
            idx = np.random.choice(L, n, replace=False)
            mask = np.zeros(L, dtype=bool)
            mask[idx] = True
            mask_t = torch.from_numpy(mask)
            return pool.slice(mask_t)

        def _pool_to_list(pool: DataProto):
            return [pool[i : i + 1] for i in range(len(pool))]

        def debug_dummy(x, name=""):
            try:
                if x is None:
                    print(f"[debug] {name}: None")
                    return
                if not hasattr(x, "batch"):
                    print(f"[debug] {name}: not a DataProto (type={type(x)})")
                    return
                if "is_dummy" in x.batch:
                    dummy_ratio = x.batch["is_dummy"].float().mean().item()
                    print(
                        f"[debug] {name}: len={len(x)}, dummy_ratio={dummy_ratio:.4f}"
                    )
                else:
                    print(f"[debug] {name}: len={len(x)}, no is_dummy field")
            except Exception as e:
                print(f"[debug] {name}: error ({e})")

        def _ensure_uid(dp):
            try:
                if dp is None:
                    return
                if not hasattr(dp, "non_tensor_batch"):
                    return
                if "uid" not in dp.non_tensor_batch:
                    L = len(dp)
                    dp.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(L)], dtype=object
                    )
            except Exception as e:
                print(f"[_ensure_uid] failed to set uid: {e}")

        def _strip_uid_namespace(uid_value) -> str:
            uid_key = str(uid_value)
            for marker in (
                "::wm_unanchored",
                "::wm",
                "::real",
                "::dp_pad",
            ):
                if marker in uid_key:
                    uid_key = uid_key.split(marker, 1)[0]
            return uid_key

        def _ensure_group_uid(dp):
            """Keep a prompt-level uid for GRPO and WM reward anchors.

            ``uid`` is allowed to be sample-level and unique.  GRPO needs a
            stable prompt/task group id shared by all n_samples rollouts and by
            anchored WM observations.  Store it separately so replay/mixing can
            preserve both identities.
            """
            try:
                if dp is None or not hasattr(dp, "non_tensor_batch"):
                    return
                _ensure_uid(dp)
                B = len(dp)
                if B <= 0:
                    return
                uids = np.asarray(dp.non_tensor_batch.get("uid", []), dtype=object)
                if len(uids) != B:
                    uids = np.array([str(uuid.uuid4()) for _ in range(B)], dtype=object)
                    dp.non_tensor_batch["uid"] = uids
                if (
                    "sample_uid" not in dp.non_tensor_batch
                    or len(np.asarray(dp.non_tensor_batch["sample_uid"], dtype=object))
                    != B
                ):
                    dp.non_tensor_batch["sample_uid"] = uids.copy()

                existing = np.asarray(
                    dp.non_tensor_batch.get("group_uid", []), dtype=object
                )
                if len(existing) == B:
                    dp.non_tensor_batch["group_uid"] = np.array(
                        [_strip_uid_namespace(uid) for uid in existing],
                        dtype=object,
                    )
                    return

                if "prompt_uid" in dp.non_tensor_batch and len(
                    np.asarray(dp.non_tensor_batch["prompt_uid"], dtype=object)
                ) == B:
                    prompt_uids = np.asarray(
                        dp.non_tensor_batch["prompt_uid"], dtype=object
                    )
                    group_uids = np.array(
                        [_strip_uid_namespace(uid) for uid in prompt_uids],
                        dtype=object,
                    )
                elif int(n_samples) > 1 and B % int(n_samples) == 0:
                    stride = int(n_samples)
                    group_uids = np.array(
                        [
                            _strip_uid_namespace(uids[(idx // stride) * stride])
                            for idx in range(B)
                        ],
                        dtype=object,
                    )
                else:
                    group_uids = np.array(
                        [_strip_uid_namespace(uid) for uid in uids],
                        dtype=object,
                    )
                dp.non_tensor_batch["group_uid"] = group_uids
            except Exception as e:
                print(f"[_ensure_group_uid] failed: {e}", flush=True)

        def _filter_dataproto_by_sample_mask(
            dp: DataProto, mask: torch.Tensor, *, context: str
        ) -> DataProto:
            if not _is_valid_dataproto(dp):
                return dp
            B = len(dp)
            mask = mask.detach().to(dtype=torch.bool).reshape(-1)
            if mask.numel() != B:
                print(
                    f"[{context}] sample mask length mismatch: "
                    f"{mask.numel()} vs batch={B}; keep original batch.",
                    flush=True,
                )
                return dp
            keep_count = int(mask.sum().item())
            if keep_count <= 0:
                return DataProto.empty_like(dp)
            if keep_count >= B:
                return dp
            return dp.slice(mask)

        def _sample_field_or_default(
            dp: DataProto,
            key: str,
            *,
            default: float,
            dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            B = len(dp)
            device = next(iter(dp.batch.values())).device
            if key not in dp.batch:
                return torch.full((B,), default, device=device, dtype=dtype)
            value = dp.batch[key]
            if value.dim() > 1:
                value = value.reshape(B, -1)[:, 0]
            return value.to(device=device, dtype=dtype).reshape(B)

        def _attach_real_reward_anchors(dp: DataProto, is_wm: bool):
            """Carry real-env outcome as the reward anchor for WM observations."""
            if not _is_valid_dataproto(dp):
                return dp
            try:
                _ensure_uid(dp)
                _ensure_group_uid(dp)
                B = len(dp)
                device = next(iter(dp.batch.values())).device
                if "anchor_reward" in dp.batch:
                    anchor_reward = _sample_field_or_default(
                        dp, "anchor_reward", default=0.0, dtype=torch.float32
                    )
                else:
                    anchor_reward = torch.zeros(
                        (B,), dtype=torch.float32, device=device
                    )

                if "has_anchor_reward" in dp.batch:
                    has_anchor_reward = (
                        _sample_field_or_default(
                            dp, "has_anchor_reward", default=0.0, dtype=torch.float32
                        )
                        > 0.5
                    )
                else:
                    has_anchor_reward = torch.zeros(
                        (B,), dtype=torch.bool, device=device
                    )

                if "env_complete" in dp.batch:
                    env_reward = _sample_field_or_default(
                        dp, "env_complete", default=0.0, dtype=torch.float32
                    )
                    anchor_reward = env_reward
                    has_anchor_reward = torch.ones(
                        (B,), dtype=torch.bool, device=device
                    )
                elif (not is_wm) and "complete" in dp.batch:
                    env_reward = _sample_field_or_default(
                        dp, "complete", default=0.0, dtype=torch.float32
                    )
                    anchor_reward = env_reward
                    has_anchor_reward = torch.ones(
                        (B,), dtype=torch.bool, device=device
                    )

                dp.batch["anchor_reward"] = anchor_reward.to(
                    dtype=torch.float32
                ).contiguous()
                dp.batch["has_anchor_reward"] = has_anchor_reward.to(
                    dtype=torch.float32
                ).contiguous()

                group_uids = np.asarray(
                    dp.non_tensor_batch.get("group_uid", []), dtype=object
                )
                if len(group_uids) == B:
                    anchor_uids = np.asarray(
                        dp.non_tensor_batch.get("anchor_uid", []), dtype=object
                    )
                    if (not is_wm) or len(anchor_uids) != B:
                        dp.non_tensor_batch["anchor_uid"] = group_uids.copy()
                    else:
                        dp.non_tensor_batch["anchor_uid"] = np.array(
                            [_strip_uid_namespace(uid) for uid in anchor_uids],
                            dtype=object,
                        )
                elif "anchor_uid" not in dp.non_tensor_batch:
                    dp.non_tensor_batch["anchor_uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(B)], dtype=object
                    )
            except Exception as e:
                print(f"[_attach_real_reward_anchors] failed: {e}", flush=True)
            return dp

        def _build_anchor_reward_lookup(dp: DataProto, *, real_only: bool = False):
            lookup = {}
            if not _is_valid_dataproto(dp):
                return lookup
            try:
                _ensure_uid(dp)
                _ensure_group_uid(dp)
                dp = _attach_real_reward_anchors(dp, is_wm=False)
                B = len(dp)
                uids = np.asarray(
                    dp.non_tensor_batch.get(
                        "group_uid", dp.non_tensor_batch.get("uid", [])
                    ),
                    dtype=object,
                )
                anchor_uids = np.asarray(
                    dp.non_tensor_batch.get("anchor_uid", uids), dtype=object
                )
                if len(anchor_uids) != B:
                    return lookup
                has_anchor = _sample_field_or_default(
                    dp, "has_anchor_reward", default=0.0, dtype=torch.float32
                ) > 0.5
                anchor_reward = _sample_field_or_default(
                    dp, "anchor_reward", default=0.0, dtype=torch.float32
                )
                if real_only and "is_wm" in dp.batch:
                    is_wm = _sample_field_or_default(
                        dp, "is_wm", default=0.0, dtype=torch.float32
                    ) > 0.5
                    has_anchor = has_anchor & (~is_wm)
                has_anchor_np = has_anchor.detach().cpu().numpy()
                rewards_np = anchor_reward.detach().cpu().numpy()
                for idx, key in enumerate(anchor_uids):
                    if not has_anchor_np[idx]:
                        continue
                    key = _strip_uid_namespace(key)
                    reward_value = float(rewards_np[idx])
                    # Binary LIBERO success is an outcome reward; if multiple real
                    # samples share the same prompt uid, any success is a valid
                    # positive anchor for the generated observation branch.
                    lookup[key] = max(float(lookup.get(key, 0.0)), reward_value)
            except Exception as e:
                print(f"[_build_anchor_reward_lookup] failed: {e}", flush=True)
            return lookup

        def _apply_anchor_reward_lookup(
            dp: DataProto,
            anchor_lookup: Dict[str, float],
            *,
            wm_only: bool = True,
        ):
            if not _is_valid_dataproto(dp):
                return dp
            try:
                _ensure_uid(dp)
                _ensure_group_uid(dp)
                dp = _attach_real_reward_anchors(dp, is_wm=wm_only)
                B = len(dp)
                device = next(iter(dp.batch.values())).device
                uids = np.asarray(
                    dp.non_tensor_batch.get(
                        "group_uid", dp.non_tensor_batch.get("uid", [])
                    ),
                    dtype=object,
                )
                anchor_uids = np.asarray(
                    dp.non_tensor_batch.get("anchor_uid", uids), dtype=object
                )
                if len(anchor_uids) != B:
                    return dp

                anchor_reward = _sample_field_or_default(
                    dp, "anchor_reward", default=0.0, dtype=torch.float32
                ).clone()
                has_anchor = (
                    _sample_field_or_default(
                        dp, "has_anchor_reward", default=0.0, dtype=torch.float32
                    )
                    > 0.5
                )
                if wm_only and "is_wm" in dp.batch:
                    target_mask = _sample_field_or_default(
                        dp, "is_wm", default=0.0, dtype=torch.float32
                    ) > 0.5
                elif wm_only:
                    target_mask = torch.ones((B,), dtype=torch.bool, device=device)
                else:
                    target_mask = torch.ones((B,), dtype=torch.bool, device=device)

                matched_anchor = torch.zeros((B,), dtype=torch.bool, device=device)
                for idx, key in enumerate(anchor_uids):
                    if not bool(target_mask[idx].item()):
                        continue
                    key = _strip_uid_namespace(key)
                    if key not in anchor_lookup:
                        continue
                    anchor_reward[idx] = float(anchor_lookup[key])
                    has_anchor[idx] = True
                    matched_anchor[idx] = True

                if wm_only and require_wm_anchor_reward:
                    has_anchor = torch.where(
                        target_mask & (~matched_anchor),
                        torch.zeros_like(has_anchor),
                        has_anchor,
                    )

                dp.batch["anchor_reward"] = anchor_reward.to(
                    dtype=torch.float32
                ).contiguous()
                dp.batch["has_anchor_reward"] = has_anchor.to(
                    dtype=torch.float32
                ).contiguous()
            except Exception as e:
                print(f"[_apply_anchor_reward_lookup] failed: {e}", flush=True)
            return dp

        def _split_grpo_uid_namespace_by_source(dp):
            stats = {
                "applied": 0.0,
                "cross_source_shared_count": 0.0,
                "real_group_count": 0.0,
                "imag_group_count": 0.0,
                "grpo_uid_singleton_group_count": 0.0,
                "effective_grpo_group_count": 0.0,
            }
            try:
                if not _is_valid_dataproto(dp):
                    return stats
                if not hasattr(dp, "non_tensor_batch"):
                    return stats
                if "is_wm" not in dp.batch:
                    return stats

                _ensure_uid(dp)
                _ensure_group_uid(dp)
                uids = np.asarray(
                    dp.non_tensor_batch.get(
                        "group_uid", dp.non_tensor_batch["uid"]
                    ),
                    dtype=object,
                )
                is_wm_np = (
                    dp.batch["is_wm"].reshape(-1).detach().cpu().numpy() > 0.5
                )
                if len(uids) != len(is_wm_np):
                    return stats

                real_uid_set = {_strip_uid_namespace(uid) for uid in uids[~is_wm_np]}
                imag_uid_set = {_strip_uid_namespace(uid) for uid in uids[is_wm_np]}
                shared_uid_set = real_uid_set & imag_uid_set
                stats["cross_source_shared_count"] = float(len(shared_uid_set))
                stats["real_group_count"] = float(len(real_uid_set))
                stats["imag_group_count"] = float(len(imag_uid_set))

                if (not real_uid_set) or (not imag_uid_set):
                    return stats

                new_uids = uids.copy()
                for idx, uid in enumerate(uids):
                    suffix = "::wm" if is_wm_np[idx] else "::real"
                    uid_key = _strip_uid_namespace(uid)
                    if not uid_key.endswith(suffix):
                        new_uids[idx] = f"{uid_key}{suffix}"

                dp.non_tensor_batch["grpo_uid"] = new_uids
                unique_uids, uid_counts = np.unique(
                    np.asarray(list(map(str, new_uids)), dtype=object),
                    return_counts=True,
                )
                stats["grpo_uid_singleton_group_count"] = float(
                    int((uid_counts <= 1).sum())
                )
                stats["effective_grpo_group_count"] = float(
                    int((uid_counts > 1).sum())
                )
                stats["applied"] = 1.0
            except Exception as e:
                print(f"[_split_grpo_uid_namespace_by_source] failed: {e}")
            return stats

        def _update_grpo_group_size_stats(stats: Dict[str, float], uids) -> None:
            try:
                if uids is None:
                    return
                uid_arr = np.asarray(list(map(str, uids)), dtype=object)
                if uid_arr.size <= 0:
                    return
                _, uid_counts = np.unique(uid_arr, return_counts=True)
                stats["grpo_uid_singleton_group_count"] = float(
                    int((uid_counts <= 1).sum())
                )
                stats["effective_grpo_group_count"] = float(
                    int((uid_counts > 1).sum())
                )
                stats["grpo_uid_mean_group_size"] = float(np.mean(uid_counts))
                stats["grpo_uid_min_group_size"] = float(np.min(uid_counts))
                stats["grpo_uid_max_group_size"] = float(np.max(uid_counts))
            except Exception as e:
                print(f"[_update_grpo_group_size_stats] failed: {e}", flush=True)

        def _apply_grpo_uid_grouping(dp):
            stats = {
                "mode_anchor": 0.0,
                "mode_source": 0.0,
                "mode_none": 0.0,
                "applied": 0.0,
                "cross_source_shared_count": 0.0,
                "real_group_count": 0.0,
                "imag_group_count": 0.0,
                "anchor_group_count": 0.0,
                "unanchored_wm_count": 0.0,
                "grpo_uid_singleton_group_count": 0.0,
                "effective_grpo_group_count": 0.0,
                "grpo_uid_mean_group_size": 0.0,
                "grpo_uid_min_group_size": 0.0,
                "grpo_uid_max_group_size": 0.0,
            }
            mode = wm_grpo_uid_mode
            if mode in ("off", "none", "shared"):
                stats["mode_none"] = 1.0
                try:
                    if _is_valid_dataproto(dp) and hasattr(dp, "non_tensor_batch"):
                        _ensure_uid(dp)
                        _ensure_group_uid(dp)
                        group_uids = np.asarray(
                            dp.non_tensor_batch.get(
                                "group_uid", dp.non_tensor_batch["uid"]
                            ),
                            dtype=object,
                        )
                        if len(group_uids) == len(dp):
                            dp.non_tensor_batch["grpo_uid"] = group_uids.copy()
                            _update_grpo_group_size_stats(stats, group_uids)
                except Exception as e:
                    print(f"[_apply_grpo_uid_grouping:none] failed: {e}", flush=True)
                return stats
            if mode in ("source", "split"):
                stats["mode_source"] = 1.0
                split_stats = _split_grpo_uid_namespace_by_source(dp)
                stats.update({k: float(v) for k, v in split_stats.items()})
                return stats
            if mode not in ("anchor", "reward_anchor", "reward-anchor"):
                mode = "anchor"

            stats["mode_anchor"] = 1.0
            try:
                if not _is_valid_dataproto(dp):
                    return stats
                if not hasattr(dp, "non_tensor_batch"):
                    return stats
                if "is_wm" not in dp.batch:
                    return stats

                _ensure_uid(dp)
                _ensure_group_uid(dp)
                uids = np.asarray(dp.non_tensor_batch["uid"], dtype=object)
                group_uids = np.asarray(
                    dp.non_tensor_batch.get("group_uid", uids), dtype=object
                )
                anchor_uids = np.asarray(
                    dp.non_tensor_batch.get("anchor_uid", group_uids), dtype=object
                )
                is_wm_np = (
                    dp.batch["is_wm"].reshape(-1).detach().cpu().numpy() > 0.5
                )
                has_anchor_np = (
                    _sample_field_or_default(
                        dp, "has_anchor_reward", default=0.0, dtype=torch.float32
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    > 0.5
                )
                if (
                    len(uids) != len(is_wm_np)
                    or len(group_uids) != len(is_wm_np)
                    or len(anchor_uids) != len(is_wm_np)
                    or len(has_anchor_np) != len(is_wm_np)
                ):
                    return stats

                new_uids = group_uids.copy()
                for idx, uid in enumerate(uids):
                    uid_key = _strip_uid_namespace(uid)
                    if is_wm_np[idx]:
                        if has_anchor_np[idx]:
                            new_uids[idx] = _strip_uid_namespace(anchor_uids[idx])
                        else:
                            new_uids[idx] = f"{uid_key}::wm_unanchored"
                    else:
                        new_uids[idx] = _strip_uid_namespace(anchor_uids[idx])

                real_uid_set = {str(uid) for uid in new_uids[~is_wm_np]}
                imag_uid_set = {str(uid) for uid in new_uids[is_wm_np]}
                shared_uid_set = real_uid_set & imag_uid_set
                stats["cross_source_shared_count"] = float(len(shared_uid_set))
                stats["real_group_count"] = float(len(real_uid_set))
                stats["imag_group_count"] = float(len(imag_uid_set))
                stats["anchor_group_count"] = float(len(set(map(str, new_uids))))
                stats["unanchored_wm_count"] = float(
                    int((is_wm_np & (~has_anchor_np)).sum())
                )

                dp.non_tensor_batch["grpo_uid"] = new_uids
                _update_grpo_group_size_stats(stats, new_uids)
                stats["applied"] = 1.0
            except Exception as e:
                print(f"[_apply_grpo_uid_grouping] failed: {e}", flush=True)
            return stats

        def _enforce_wm_anchor_weight_contract(dp: DataProto):
            stats = {
                "enabled": 0.0,
                "wm_anchor_coverage": 0.0,
                "wm_unanchored_weight_zeroed": 0.0,
            }
            if not zero_unanchored_wm_weight or not _is_valid_dataproto(dp):
                return stats
            if "is_wm" not in dp.batch or "is_weight" not in dp.batch:
                return stats
            try:
                stats["enabled"] = 1.0
                is_wm = _sample_field_or_default(
                    dp, "is_wm", default=0.0, dtype=torch.float32
                ) > 0.5
                has_anchor = _sample_field_or_default(
                    dp, "has_anchor_reward", default=0.0, dtype=torch.float32
                ) > 0.5
                if bool(is_wm.any().item()):
                    stats["wm_anchor_coverage"] = float(
                        has_anchor[is_wm].float().mean().item()
                    )
                no_anchor_wm = is_wm & (~has_anchor)
                if bool(no_anchor_wm.any().item()):
                    dp.batch["is_weight"] = dp.batch["is_weight"].clone()
                    dp.batch["is_weight"][no_anchor_wm] = 0.0
                    stats["wm_unanchored_weight_zeroed"] = float(
                        no_anchor_wm.float().sum().item()
                    )
            except Exception as e:
                print(f"[_enforce_wm_anchor_weight_contract] failed: {e}", flush=True)
            return stats

        def _set_sample_source_fields(
            dp: DataProto,
            *,
            is_wm: bool,
            is_weight: float = 1.0,
        ) -> DataProto:
            if not _is_valid_dataproto(dp):
                return dp
            try:
                device = next(iter(dp.batch.values())).device
                B = len(dp)
                dp.batch["is_wm"] = torch.full(
                    (B,), float(bool(is_wm)), dtype=torch.float32, device=device
                )
                dp.batch["is_weight"] = torch.full(
                    (B,), float(is_weight), dtype=torch.float32, device=device
                )
            except Exception as e:
                print(f"[_set_sample_source_fields] failed: {e}", flush=True)
            return dp

        def _compute_target_wm_count(total_needed: int, r_wm_value: float):
            metrics = {}
            clipped_r_wm = float(np.clip(r_wm_value, 0.0, 1.0))
            raw_wm_to_take = float(total_needed) * clipped_r_wm
            carry_in = (
                float(max(0.0, getattr(self, "_wm_ratio_carry", 0.0)))
                if wm_ratio_rounding == "carry"
                else 0.0
            )
            if clipped_r_wm <= 0.0:
                n_wm_to_take = 0
            elif clipped_r_wm >= 1.0:
                n_wm_to_take = total_needed
            elif wm_ratio_rounding == "stochastic":
                base = int(np.floor(raw_wm_to_take))
                frac = float(raw_wm_to_take - base)
                n_wm_to_take = base + int(np.random.random() < frac)
            elif wm_ratio_rounding == "ceil":
                n_wm_to_take = int(np.ceil(raw_wm_to_take))
            elif wm_ratio_rounding == "floor":
                n_wm_to_take = int(np.floor(raw_wm_to_take))
            elif wm_ratio_rounding == "carry":
                n_wm_to_take = int(np.floor(raw_wm_to_take + carry_in))
            else:
                n_wm_to_take = int(np.floor(raw_wm_to_take))
            n_wm_to_take = int(max(0, min(total_needed, n_wm_to_take)))
            carry_out = 0.0
            if wm_ratio_rounding == "carry":
                carry_out = float(max(0.0, raw_wm_to_take + carry_in - n_wm_to_take))
            metrics["wm/target_wm_sample_float"] = float(raw_wm_to_take)
            metrics["wm/target_wm_sample_count"] = int(n_wm_to_take)
            metrics["wm/target_real_sample_count"] = int(total_needed - n_wm_to_take)
            metrics["wm/ratio_rounding_stochastic"] = (
                1.0 if wm_ratio_rounding == "stochastic" else 0.0
            )
            metrics["wm/ratio_rounding_carry"] = (
                1.0 if wm_ratio_rounding == "carry" else 0.0
            )
            metrics["wm/ratio_carry_in"] = float(carry_in)
            metrics["wm/ratio_carry_out"] = float(carry_out)
            return n_wm_to_take, metrics

        def _sample_keys_for_grouping(
            dp: DataProto, *, prefer_anchor: bool = False
        ) -> np.ndarray:
            if not _is_valid_dataproto(dp):
                return np.asarray([], dtype=object)
            _ensure_uid(dp)
            _ensure_group_uid(dp)
            fallback = dp.non_tensor_batch.get("group_uid", dp.non_tensor_batch["uid"])
            if prefer_anchor:
                values = dp.non_tensor_batch.get("anchor_uid", fallback)
            else:
                values = fallback
            values = np.asarray(values, dtype=object)
            if len(values) != len(dp):
                values = np.asarray(fallback, dtype=object)
            return np.asarray(
                [_strip_uid_namespace(value) for value in values], dtype=object
            )

        def _per_sample_valid_response_tokens(dp: DataProto) -> torch.Tensor:
            if not _is_valid_dataproto(dp):
                return torch.zeros((0,), dtype=torch.long)
            batch = dp.batch
            B = len(dp)
            device = (
                dp.device
                if hasattr(dp, "device")
                else next(iter(batch.values())).device
            )
            tokens = DataProtoFilter._per_sample_valid_response_tokens(
                batch=batch,
                B=B,
                device=device,
                action_token_len=self.config.actor_rollout_ref.model.action_token_len,
            )
            if "wm_pred_valid" in batch:
                wm_pred_valid = batch["wm_pred_valid"]
                if wm_pred_valid.dim() > 1:
                    wm_pred_valid = wm_pred_valid.view(B, -1)[:, 0]
                tokens = torch.where(
                    wm_pred_valid.to(device=device, dtype=torch.bool),
                    tokens,
                    torch.zeros_like(tokens),
                )
            return tokens.to(device=device).reshape(B)

        def _wm_actor_admission_mask(
            dp: DataProto,
            *,
            real_anchor_keys: Optional[set] = None,
        ):
            stats = {
                "actor_anchor_reward_min": float(wm_actor_anchor_reward_min),
                "actor_admission_candidate_count": 0.0,
                "actor_admission_has_anchor_count": 0.0,
                "actor_admission_positive_anchor_count": 0.0,
                "actor_admission_valid_token_count": 0.0,
                "actor_admission_pairable_count": 0.0,
                "actor_admission_count": 0.0,
                "actor_admission_rejected_count": 0.0,
            }
            if not _is_valid_dataproto(dp):
                return torch.zeros((0,), dtype=torch.bool), stats
            B = len(dp)
            device = next(iter(dp.batch.values())).device
            stats["actor_admission_candidate_count"] = float(B)
            has_anchor = (
                _sample_field_or_default(
                    dp, "has_anchor_reward", default=0.0, dtype=torch.float32
                )
                > 0.5
            )
            anchor_reward = _sample_field_or_default(
                dp, "anchor_reward", default=0.0, dtype=torch.float32
            )
            valid_tokens = _per_sample_valid_response_tokens(dp)
            positive_anchor = (
                has_anchor
                & (anchor_reward >= float(wm_actor_anchor_reward_min))
                & (valid_tokens.to(device=device) > 0)
            )
            stats["actor_admission_has_anchor_count"] = float(
                has_anchor.detach().cpu().sum().item()
            )
            stats["actor_admission_positive_anchor_count"] = float(
                positive_anchor.detach().cpu().sum().item()
            )
            stats["actor_admission_valid_token_count"] = float(
                (valid_tokens.detach().cpu() > 0).sum().item()
            )

            if real_anchor_keys is None:
                pairable = torch.ones((B,), dtype=torch.bool, device=device)
            else:
                group_uids = np.asarray(
                    dp.non_tensor_batch.get(
                        "group_uid", dp.non_tensor_batch.get("uid", [])
                    ),
                    dtype=object,
                )
                anchor_uids = np.asarray(
                    dp.non_tensor_batch.get("anchor_uid", group_uids), dtype=object
                )
                if len(anchor_uids) != B:
                    pairable_np = np.zeros((B,), dtype=bool)
                else:
                    pairable_np = np.asarray(
                        [
                            _strip_uid_namespace(uid) in real_anchor_keys
                            for uid in anchor_uids
                        ],
                        dtype=bool,
                    )
                pairable = torch.from_numpy(pairable_np).to(
                    device=device, dtype=torch.bool
                )
            stats["actor_admission_pairable_count"] = float(
                pairable.detach().cpu().sum().item()
            )
            admit_mask = positive_anchor & pairable
            admitted = int(admit_mask.detach().cpu().sum().item())
            stats["actor_admission_count"] = float(admitted)
            stats["actor_admission_rejected_count"] = float(max(0, B - admitted))
            return admit_mask, stats

        def _slice_dataproto_indices(
            dp: DataProto, indices, *, context: str
        ) -> Optional[DataProto]:
            if not _is_valid_dataproto(dp):
                return None
            try:
                clean_indices = sorted(
                    {
                        int(idx)
                        for idx in list(indices)
                        if 0 <= int(idx) < len(dp)
                    }
                )
                if len(clean_indices) == 0:
                    return None
                mask = torch.zeros(
                    (len(dp),),
                    dtype=torch.bool,
                    device=next(iter(dp.batch.values())).device,
                )
                mask[torch.as_tensor(clean_indices, dtype=torch.long, device=mask.device)] = True
                return clone_dataproto_for_replay(
                    _filter_dataproto_by_sample_mask(
                        dp, mask, context=f"{context} index slice"
                    )
                )
            except Exception as e:
                print(f"[_slice_dataproto_indices:{context}] failed: {e}", flush=True)
                return None

        def _build_merl_group_aware_actor_mix(
            *,
            current_real_batch: DataProto,
            real_pool: DataProto,
            wm_pool: DataProto,
            wm_sample_weights: torch.Tensor,
            wm_priorities: torch.Tensor,
            total_needed: int,
            target_wm_count: int,
        ):
            stats = {
                "wm/group_aware_enabled": 1.0,
                "wm/group_aware_target_wm_sample_count": float(target_wm_count),
                "wm/group_aware_current_anchor_candidate_count": 0.0,
                "wm/group_aware_selected_wm_sample_count": 0.0,
                "wm/group_aware_real_replaced_count": 0.0,
                "wm/group_aware_real_group_count": 0.0,
                "wm/group_aware_fallback_real_batch": 0.0,
            }
            fallback = clone_dataproto_for_replay(current_real_batch)
            fallback = _attach_real_reward_anchors(fallback, is_wm=False)
            fallback = _set_sample_source_fields(fallback, is_wm=False, is_weight=1.0)
            if target_wm_count <= 0 or not _is_valid_dataproto(real_pool):
                stats["wm/group_aware_fallback_real_batch"] = 1.0
                return fallback, None, stats

            real_pool = _attach_real_reward_anchors(real_pool, is_wm=False)
            wm_pool = _attach_real_reward_anchors(wm_pool, is_wm=True)
            wm_pool = _apply_anchor_reward_lookup(
                wm_pool, real_anchor_reward_lookup, wm_only=True
            )
            real_keys = _sample_keys_for_grouping(real_pool, prefer_anchor=True)
            stats["wm/group_aware_real_group_count"] = float(
                len(set(map(str, real_keys)))
            )
            if not _is_valid_dataproto(wm_pool) or len(real_keys) != len(real_pool):
                stats["wm/group_aware_fallback_real_batch"] = 1.0
                return fallback, None, stats

            wm_keys = _sample_keys_for_grouping(wm_pool, prefer_anchor=True)
            has_anchor = (
                _sample_field_or_default(
                    wm_pool, "has_anchor_reward", default=0.0, dtype=torch.float32
                )
                > 0.5
            )
            real_key_set = set(map(str, real_keys))
            admission_mask, admission_stats = _wm_actor_admission_mask(
                wm_pool, real_anchor_keys=real_key_set
            )
            for stat_key, stat_value in admission_stats.items():
                stats[f"wm/group_aware_{stat_key}"] = float(stat_value)
            has_anchor_np = has_anchor.detach().cpu().numpy()
            admission_np = admission_mask.detach().cpu().numpy()
            candidate_indices = [
                idx
                for idx, key in enumerate(wm_keys)
                if bool(has_anchor_np[idx])
                and bool(admission_np[idx])
                and str(key) in real_key_set
            ]
            stats["wm/group_aware_current_anchor_candidate_count"] = float(
                len(candidate_indices)
            )
            stats["wm/group_aware_positive_anchor_candidate_count"] = float(
                len(candidate_indices)
            )
            if len(candidate_indices) <= 0:
                stats["wm/group_aware_fallback_real_batch"] = 1.0
                return fallback, None, stats

            priority_arr = np.ones((len(wm_pool),), dtype=np.float64)
            try:
                if wm_priorities is not None and int(wm_priorities.numel()) == len(wm_pool):
                    priority_arr = (
                        wm_priorities.detach().cpu().float().numpy().astype(np.float64)
                    )
            except Exception:
                priority_arr = np.ones((len(wm_pool),), dtype=np.float64)
            candidate_indices = sorted(
                candidate_indices,
                key=lambda idx: float(priority_arr[idx]),
                reverse=True,
            )

            real_keep_indices = list(range(min(len(real_pool), total_needed)))
            if len(real_keep_indices) <= 0:
                stats["wm/group_aware_fallback_real_batch"] = 1.0
                return fallback, None, stats

            selected_wm_indices = []
            removed_real_indices = set()
            max_wm = min(target_wm_count, len(candidate_indices), len(real_keep_indices))
            for wm_idx in candidate_indices:
                if len(selected_wm_indices) >= max_wm:
                    break
                anchor_key = str(wm_keys[wm_idx])
                same_group_real = [
                    real_idx
                    for real_idx in real_keep_indices
                    if real_idx not in removed_real_indices
                    and str(real_keys[real_idx]) == anchor_key
                ]
                if len(same_group_real) <= 1:
                    continue
                removed_real_indices.add(int(same_group_real[-1]))
                selected_wm_indices.append(int(wm_idx))

            if len(selected_wm_indices) <= 0:
                stats["wm/group_aware_fallback_real_batch"] = 1.0
                return fallback, None, stats

            final_real_indices = [
                idx for idx in real_keep_indices if idx not in removed_real_indices
            ]
            mixed_parts = []
            mixed_samples = []
            real_chunk = _slice_dataproto_indices(
                real_pool, final_real_indices, context="MERL group-aware real"
            )
            if _is_valid_dataproto(real_chunk):
                real_chunk = _set_sample_source_fields(
                    real_chunk, is_wm=False, is_weight=1.0
                )
                mixed_parts.append(real_chunk)
                mixed_samples.append(
                    {"dataproto": real_chunk, "is_wm": False, "is_weight": 1.0}
                )

            for wm_idx in selected_wm_indices:
                wm_chunk = clone_dataproto_for_replay(
                    wm_pool.slice(slice(int(wm_idx), int(wm_idx) + 1))
                )
                weight = 1.0
                try:
                    if (
                        wm_sample_weights is not None
                        and int(wm_sample_weights.numel()) == len(wm_pool)
                    ):
                        weight = float(wm_sample_weights[int(wm_idx)].item())
                except Exception:
                    weight = 1.0
                wm_chunk = _set_sample_source_fields(
                    wm_chunk, is_wm=True, is_weight=weight
                )
                mixed_parts.append(wm_chunk)
                mixed_samples.append(
                    {"dataproto": wm_chunk, "is_wm": True, "is_weight": weight}
                )

            if not mixed_parts:
                stats["wm/group_aware_fallback_real_batch"] = 1.0
                return fallback, None, stats
            mixed_parts = align_dataproto_list_for_concat(mixed_parts)
            mixed_batch = DataProto.concat(mixed_parts)
            if len(mixed_batch) < total_needed:
                backfill = _sample_real_backfill(
                    total_needed - len(mixed_batch), real_backfill_pool=real_pool
                )
                if _is_valid_dataproto(backfill):
                    mixed_parts = align_dataproto_list_for_concat([mixed_batch, backfill])
                    mixed_batch = DataProto.concat(mixed_parts)
                    mixed_samples.append(
                        {"dataproto": backfill, "is_wm": False, "is_weight": 1.0}
                    )
            if len(mixed_batch) > total_needed:
                mixed_batch = mixed_batch.slice(slice(0, total_needed))
            stats["wm/group_aware_selected_wm_sample_count"] = float(
                len(selected_wm_indices)
            )
            stats["wm/group_aware_real_replaced_count"] = float(
                len(removed_real_indices)
            )
            return mixed_batch, mixed_samples, stats

        def _sample_real_backfill(
            n: int,
            *,
            real_backfill_pool: Optional[DataProto] = None,
        ) -> Optional[DataProto]:
            if n <= 0:
                return None
            chunks = []
            try:
                if _is_valid_dataproto(real_backfill_pool):
                    pool_len = len(real_backfill_pool)
                    replace = n > pool_len
                    idxes = np.random.choice(pool_len, size=n, replace=replace)
                    chunks = [
                        clone_dataproto_for_replay(
                            real_backfill_pool.slice(slice(int(idx), int(idx) + 1))
                        )
                        for idx in idxes
                    ]
                elif len(self.real_prioritized_pool) > 0:
                    samples, _, _ = self.real_prioritized_pool.sample(
                        n, alpha=0.6, beta=0.4, replace=True
                    )
                    chunks = [
                        clone_dataproto_for_replay(sample["dataproto"])
                        for sample in samples
                        if isinstance(sample, dict) and "dataproto" in sample
                    ]
                if not chunks:
                    return None
                chunks = align_dataproto_list_for_concat(chunks)
                backfill = DataProto.concat(chunks)
                backfill = _attach_real_reward_anchors(backfill, is_wm=False)
                backfill = _set_sample_source_fields(
                    backfill, is_wm=False, is_weight=1.0
                )
                return backfill
            except Exception as e:
                print(f"[_sample_real_backfill] failed: {e}", flush=True)
                return None

        def _keep_only_wm_with_current_real_anchor(
            dp: DataProto,
            *,
            real_backfill_pool: Optional[DataProto],
            context: str,
        ):
            stats = {
                "enabled": 0.0,
                "wm_candidate_count": 0.0,
                "wm_shared_anchor_count": 0.0,
                "wm_positive_actor_anchor_count": 0.0,
                "wm_invalid_actor_anchor_count": 0.0,
                "wm_dropped_count": 0.0,
                "real_backfill_count": 0.0,
                "shared_anchor_coverage": 0.0,
            }
            if (
                train_mode != "MERL"
                or not require_wm_anchor_reward
                or not wm_real_anchor_reward
                or not _is_valid_dataproto(dp)
                or "is_wm" not in dp.batch
            ):
                return dp, stats
            try:
                stats["enabled"] = 1.0
                _ensure_uid(dp)
                _ensure_group_uid(dp)
                dp = _apply_anchor_reward_lookup(
                    dp, real_anchor_reward_lookup, wm_only=True
                )
                B = len(dp)
                is_wm = _sample_field_or_default(
                    dp, "is_wm", default=0.0, dtype=torch.float32
                ) > 0.5
                if not bool(is_wm.any().item()):
                    return dp, stats

                has_anchor = _sample_field_or_default(
                    dp, "has_anchor_reward", default=0.0, dtype=torch.float32
                ) > 0.5
                group_uids = np.asarray(
                    dp.non_tensor_batch.get(
                        "group_uid", dp.non_tensor_batch.get("uid", [])
                    ),
                    dtype=object,
                )
                anchor_uids = np.asarray(
                    dp.non_tensor_batch.get("anchor_uid", group_uids), dtype=object
                )
                if len(anchor_uids) != B or len(group_uids) != B:
                    return dp, stats

                is_wm_np = is_wm.detach().cpu().numpy()
                has_anchor_np = has_anchor.detach().cpu().numpy()
                real_anchor_keys = {
                    _strip_uid_namespace(anchor_uids[idx])
                    for idx in range(B)
                    if not bool(is_wm_np[idx])
                }
                admission_mask, admission_stats = _wm_actor_admission_mask(
                    dp, real_anchor_keys=real_anchor_keys
                )
                for stat_key, stat_value in admission_stats.items():
                    stats[stat_key] = float(stat_value)
                admission_np = admission_mask.detach().cpu().numpy()
                shared_wm_np = np.zeros((B,), dtype=bool)
                for idx in range(B):
                    if not bool(is_wm_np[idx]):
                        continue
                    anchor_key = _strip_uid_namespace(anchor_uids[idx])
                    shared_wm_np[idx] = (
                        bool(has_anchor_np[idx])
                        and bool(admission_np[idx])
                        and (anchor_key in real_anchor_keys)
                    )

                wm_count = int(is_wm_np.sum())
                shared_wm_count = int(shared_wm_np.sum())
                stats["wm_candidate_count"] = float(wm_count)
                stats["wm_shared_anchor_count"] = float(shared_wm_count)
                stats["wm_positive_actor_anchor_count"] = float(shared_wm_count)
                stats["wm_invalid_actor_anchor_count"] = float(
                    max(0, wm_count - shared_wm_count)
                )
                stats["shared_anchor_coverage"] = (
                    float(shared_wm_count) / float(wm_count) if wm_count > 0 else 0.0
                )

                keep_np = (~is_wm_np) | shared_wm_np
                drop_count = int(B - keep_np.sum())
                stats["wm_dropped_count"] = float(drop_count)
                if drop_count <= 0:
                    return dp, stats

                keep_mask = torch.from_numpy(keep_np).to(
                    device=next(iter(dp.batch.values())).device, dtype=torch.bool
                )
                filtered = _filter_dataproto_by_sample_mask(
                    dp, keep_mask, context=f"{context} shared-anchor filter"
                )
                backfill = _sample_real_backfill(
                    drop_count, real_backfill_pool=real_backfill_pool
                )
                if _is_valid_dataproto(backfill):
                    if _is_valid_dataproto(filtered):
                        concat_parts = align_dataproto_list_for_concat(
                            [filtered, backfill]
                        )
                        filtered = DataProto.concat(concat_parts)
                    else:
                        filtered = backfill
                    stats["real_backfill_count"] = float(len(backfill))
                filtered = _apply_anchor_reward_lookup(
                    filtered, real_anchor_reward_lookup, wm_only=True
                )
                return filtered, stats
            except Exception as e:
                print(
                    f"[_keep_only_wm_with_current_real_anchor:{context}] failed: {e}",
                    flush=True,
                )
                return dp, stats

        def _is_valid_dataproto(dp):
            """Check if dp is a valid non-empty DataProto"""
            if dp is None:
                return False
            if not hasattr(dp, "batch"):
                return False
            if len(dp) == 0:
                return False
            return True

        def _valid_response_token_stats(dp: DataProto, wm_only: bool = False):
            if not _is_valid_dataproto(dp):
                return {
                    "sample_count": 0,
                    "valid_sample_count": 0,
                    "token_count": 0,
                    "min_tokens": 0,
                    "mean_tokens": 0.0,
                }
            batch = dp.batch
            B = len(dp)
            device = (
                dp.device
                if hasattr(dp, "device")
                else next(iter(batch.values())).device
            )
            tokens = DataProtoFilter._per_sample_valid_response_tokens(
                batch=batch,
                B=B,
                device=device,
                action_token_len=self.config.actor_rollout_ref.model.action_token_len,
            )
            if "wm_pred_valid" in batch:
                wm_pred_valid = batch["wm_pred_valid"]
                if wm_pred_valid.dim() > 1:
                    wm_pred_valid = wm_pred_valid.view(B, -1)[:, 0]
                tokens = torch.where(
                    wm_pred_valid.to(device=device, dtype=torch.bool),
                    tokens,
                    torch.zeros_like(tokens),
                )
            if wm_only:
                if "is_wm" in batch:
                    is_wm = batch["is_wm"]
                    if is_wm.dim() > 1:
                        is_wm = is_wm.view(B, -1)[:, 0]
                    sample_mask = is_wm.to(device=device) > 0.5
                else:
                    sample_mask = torch.ones(B, dtype=torch.bool, device=device)
            else:
                sample_mask = torch.ones(B, dtype=torch.bool, device=device)

            selected_tokens = tokens[sample_mask].detach().cpu().long()
            sample_count = int(selected_tokens.numel())
            if sample_count == 0:
                return {
                    "sample_count": 0,
                    "valid_sample_count": 0,
                    "token_count": 0,
                    "min_tokens": 0,
                    "mean_tokens": 0.0,
                }
            valid_sample_count = int((selected_tokens > 0).sum().item())
            token_count = int(selected_tokens.sum().item())
            min_tokens = int(selected_tokens.min().item())
            mean_tokens = float(selected_tokens.float().mean().item())
            return {
                "sample_count": sample_count,
                "valid_sample_count": valid_sample_count,
                "token_count": token_count,
                "min_tokens": min_tokens,
                "mean_tokens": mean_tokens,
            }

        def _ceil_div(numerator: int, denominator: int) -> int:
            if denominator <= 0:
                return 0
            return max(0, (int(numerator) + int(denominator) - 1) // int(denominator))

        def _split_prompt_batch(dp: DataProto, keep_count: int):
            if not _is_valid_dataproto(dp):
                return dp, None
            if keep_count <= 0:
                return DataProto.empty_like(dp), dp
            if keep_count >= len(dp):
                return dp, None

            keep_mask = torch.zeros(len(dp), dtype=torch.bool)
            keep_mask[:keep_count] = True
            keep_batch = dp.slice(keep_mask)
            remain_batch = dp.slice(~keep_mask)
            return keep_batch, remain_batch

        def _dispatch_safe_prompt_count(
            requested_prompt_count: int,
            available_prompt_count: int,
            *,
            context: str,
        ) -> int:
            requested_prompt_count = max(0, int(requested_prompt_count))
            available_prompt_count = max(0, int(available_prompt_count))
            if requested_prompt_count <= 0 or available_prompt_count <= 0:
                return 0
            dispatch_count = min(
                available_prompt_count,
                max(requested_prompt_count, actor_dp_world_size),
            )
            if dispatch_count < actor_dp_world_size:
                print(
                    f"[Batch] {context}: only {dispatch_count} prompts available for "
                    f"actor DP world_size={actor_dp_world_size}; prompt batch will be padded before dispatch.",
                    flush=True,
                )
            elif dispatch_count > requested_prompt_count:
                print(
                    f"[Batch] {context}: expanding prompt dispatch from "
                    f"{requested_prompt_count} to {dispatch_count} to satisfy "
                    f"actor DP world_size={actor_dp_world_size}. Extra valid samples "
                    "will be dropped by the existing batch reorg path.",
                    flush=True,
                )
            return dispatch_count

        def _pad_prompt_batch_for_dispatch(dp: DataProto, *, context: str) -> DataProto:
            if not _is_valid_dataproto(dp) or len(dp) >= actor_dp_world_size:
                return dp
            original_len = len(dp)
            parts = [dp]
            for pad_idx in range(actor_dp_world_size - original_len):
                src_idx = pad_idx % original_len
                parts.append(dp[src_idx : src_idx + 1])
            padded = DataProto.concat(parts)
            if "uid" in padded.non_tensor_batch:
                uids = np.asarray(padded.non_tensor_batch["uid"], dtype=object).copy()
                for idx in range(original_len, len(uids)):
                    uids[idx] = f"{uids[idx]}::dp_pad::{uuid.uuid4()}"
                padded.non_tensor_batch["uid"] = uids
            print(
                f"[Batch] {context}: padded prompt batch from {original_len} to "
                f"{len(padded)} for actor DP world_size={actor_dp_world_size}.",
                flush=True,
            )
            return padded

        def _select_prompt_representatives(dp: DataProto) -> DataProto:
            if not _is_valid_dataproto(dp):
                return dp
            if "uid" not in dp.non_tensor_batch:
                stride = max(int(n_samples), 1)
                mask = np.zeros(len(dp), dtype=bool)
                mask[::stride] = True
                return dp.slice(torch.from_numpy(mask))
            uids = np.asarray(dp.non_tensor_batch["uid"], dtype=object)
            seen = set()
            mask = np.zeros(len(dp), dtype=bool)
            for idx, uid in enumerate(uids):
                uid_key = str(uid)
                if uid_key in seen:
                    continue
                seen.add(uid_key)
                mask[idx] = True
            return dp.slice(torch.from_numpy(mask))

        def _clip_wm_ratio(ratio_value: float) -> float:
            return float(np.clip(float(ratio_value), 0.0, max_wm_ratio))

        def _compute_ratio_signal_target():
            signal_ema = getattr(self, "_wm_ratio_signal_ema", None)
            last_signal_ema = getattr(self, "_last_wm_ratio_signal_ema", None)

            metrics_out = {
                "wm/ratio_scheduler_signal_active": 0.0,
                "wm/ratio_signal_rel_improve": 0.0,
            }
            if signal_ema is None:
                return None, 0.0, metrics_out

            tau = max(float(wm_ratio_ref_loss), 1e-6)
            gamma = max(float(wm_ratio_gamma), 1e-6)
            raw_target = max_wm_ratio / (1.0 + (float(signal_ema) / tau) ** gamma)
            signal_target = _clip_wm_ratio(raw_target)
            rel_improve = 0.0
            if last_signal_ema is not None:
                denom = max(abs(float(last_signal_ema)), 1e-6)
                rel_improve = float(
                    np.clip(
                        (float(last_signal_ema) - float(signal_ema)) / denom,
                        -wm_ratio_improve_clip,
                        wm_ratio_improve_clip,
                    )
                )

            metrics_out.update(
                {
                    "wm/ratio_scheduler_signal_active": 1.0,
                    "wm/ratio_signal_target": float(signal_target),
                    "wm/ratio_signal_ema_for_scheduler": float(signal_ema),
                    "wm/ratio_signal_rel_improve": float(rel_improve),
                }
            )
            return float(signal_target), float(rel_improve), metrics_out

        def _track_wm_ratio(target_ratio: float, rel_improve: float) -> float:
            target_ratio = _clip_wm_ratio(target_ratio)
            r_prev = _clip_wm_ratio(getattr(self, "_r_wm", 0.0) or 0.0)
            cooldown_remaining = int(
                max(0, getattr(self, "_wm_ratio_cooldown_remaining", 0) or 0)
            )
            if cooldown_remaining > 0:
                self._wm_ratio_cooldown_remaining = max(0, cooldown_remaining - 1)
                self._wm_ratio_carry = 0.0
                self._r_wm = 0.0
                self._wm_ratio_health_metrics = {
                    "wm/ratio_cooldown_active": 1.0,
                    "wm/ratio_cooldown_remaining": float(cooldown_remaining),
                    "wm/ratio_cooldown_remaining_after_step": float(
                        self._wm_ratio_cooldown_remaining
                    ),
                    "wm/ratio_health_anchor_coverage_prev": float(
                        np.clip(
                            getattr(self, "_last_wm_actor_anchor_coverage", 1.0),
                            0.0,
                            1.0,
                        )
                    ),
                    "wm/ratio_health_realized_vs_target_prev": float(
                        np.clip(
                            getattr(self, "_last_wm_ratio_realized_vs_target", 1.0),
                            0.0,
                            1.0,
                        )
                    ),
                    "wm/ratio_health_step_multiplier": 0.0,
                    "wm/ratio_health_step_cap": 0.0,
                }
                return 0.0

            if target_ratio >= r_prev:
                improve_scale = 0.0
                if wm_ratio_improve_clip > 0:
                    improve_scale = max(0.0, rel_improve) / wm_ratio_improve_clip
                improve_scale = float(np.clip(improve_scale, 0.0, 1.0))
                step_cap = (
                    wm_ratio_up_min
                    + (wm_ratio_up_max - wm_ratio_up_min) * improve_scale
                )
                anchor_cov = float(
                    np.clip(
                        getattr(self, "_last_wm_actor_anchor_coverage", 1.0),
                        0.0,
                        1.0,
                    )
                )
                anchor_scale = min(
                    1.0, anchor_cov / wm_ratio_anchor_coverage_threshold
                )
                realized_ratio = float(
                    np.clip(
                        getattr(self, "_last_wm_ratio_realized_vs_target", 1.0),
                        0.0,
                        1.0,
                    )
                )
                anchor_multiplier = wm_ratio_health_min_slope + (
                    1.0 - wm_ratio_health_min_slope
                ) * anchor_scale
                hard_skip_ratio = float(
                    np.clip(
                        getattr(self, "_last_actor_ppo_kl_hard_skip_ratio", 0.0),
                        0.0,
                        1.0,
                    )
                )
                hard_multiplier = max(
                    wm_ratio_health_min_slope, 1.0 - hard_skip_ratio
                )
                hard_max = float(
                    max(0.0, getattr(self, "_last_actor_ppo_kl_hard_max", 0.0))
                )
                hard_limit = max(
                    0.0,
                    float(
                        getattr(
                            self.config.actor_rollout_ref.actor,
                            "ppo_kl_hard_limit",
                            0.0,
                        )
                        or 0.0
                    ),
                )
                if hard_limit > 0.0 and hard_max > hard_limit:
                    kl_multiplier = max(
                        wm_ratio_health_min_slope,
                        hard_limit / max(hard_max, 1e-6),
                    )
                else:
                    kl_multiplier = 1.0
                health_multiplier = float(
                    np.clip(
                        anchor_multiplier * hard_multiplier * kl_multiplier,
                        wm_ratio_health_min_slope,
                        1.0,
                    )
                )
                step_cap = step_cap * health_multiplier
                self._wm_ratio_health_metrics = {
                    "wm/ratio_cooldown_active": 0.0,
                    "wm/ratio_health_anchor_coverage_prev": float(anchor_cov),
                    "wm/ratio_health_realized_vs_target_prev": float(realized_ratio),
                    "wm/ratio_health_anchor_multiplier": float(anchor_multiplier),
                    "wm/ratio_health_hard_skip_ratio_prev": float(hard_skip_ratio),
                    "wm/ratio_health_hard_multiplier": float(hard_multiplier),
                    "wm/ratio_health_kl_max_prev": float(hard_max),
                    "wm/ratio_health_kl_multiplier": float(kl_multiplier),
                    "wm/ratio_health_step_multiplier": float(health_multiplier),
                    "wm/ratio_health_step_cap": float(step_cap),
                }
                return float(min(target_ratio, r_prev + step_cap))

            down_step = min(
                wm_ratio_down_max, max(r_prev - target_ratio, wm_ratio_up_min)
            )
            self._wm_ratio_health_metrics = {
                "wm/ratio_cooldown_active": 0.0,
                "wm/ratio_health_anchor_coverage_prev": float(
                    np.clip(
                        getattr(self, "_last_wm_actor_anchor_coverage", 1.0),
                        0.0,
                        1.0,
                    )
                ),
                "wm/ratio_health_hard_skip_ratio_prev": float(
                    np.clip(
                        getattr(self, "_last_actor_ppo_kl_hard_skip_ratio", 0.0),
                        0.0,
                        1.0,
                    )
                ),
                "wm/ratio_health_realized_vs_target_prev": float(
                    np.clip(
                        getattr(self, "_last_wm_ratio_realized_vs_target", 1.0),
                        0.0,
                        1.0,
                    )
                ),
                "wm/ratio_health_step_multiplier": 1.0,
                "wm/ratio_health_step_cap": float(down_step),
            }
            return float(max(target_ratio, r_prev - down_step))

        def _compute_alpha_from_confidence(confidence_value: float) -> float:
            confidence_value = float(np.clip(confidence_value, 0.0, 1.0))
            alpha = imag_ratio_min + (imag_ratio_max - imag_ratio_min) * (
                confidence_value**imag_ratio_gamma
            )
            return float(np.clip(alpha, imag_ratio_min, imag_ratio_max))

        def _compute_horizon_from_confidence(confidence_value: float) -> int:
            confidence_value = float(np.clip(confidence_value, 0.0, 1.0))
            desired_horizon = imag_horizon_min + round(
                (imag_horizon_max - imag_horizon_min) * confidence_value
            )
            desired_horizon = int(
                np.clip(desired_horizon, imag_horizon_min, imag_horizon_max)
            )
            current_horizon = int(
                np.clip(
                    getattr(self, "_imagined_horizon", imag_horizon_min),
                    imag_horizon_min,
                    imag_horizon_max,
                )
            )
            growth_allowed = True
            health_window_count = 0
            if desired_horizon > current_horizon:
                history = list(getattr(self, "_merl_policy_health_history", []))
                recent = history[-wm_horizon_growth_window:]
                health_window_count = len(recent)
                growth_allowed = (
                    len(recent) >= wm_horizon_growth_window
                    and all(bool(item.get("healthy_for_horizon", False)) for item in recent)
                )
                if growth_allowed:
                    chunk_len = max(
                        1,
                        int(
                            getattr(
                                self.config.actor_rollout_ref.model,
                                "action_chunks_len",
                                1,
                            )
                        ),
                    )
                    next_horizon = min(desired_horizon, current_horizon + chunk_len)
                else:
                    next_horizon = current_horizon
            else:
                next_horizon = desired_horizon
            self._wm_horizon_health_metrics = {
                "wm/horizon_desired_from_confidence": float(desired_horizon),
                "wm/horizon_growth_allowed": 1.0 if growth_allowed else 0.0,
                "wm/horizon_health_window_count": float(health_window_count),
                "wm/horizon_growth_window": float(wm_horizon_growth_window),
            }
            return int(np.clip(next_horizon, imag_horizon_min, imag_horizon_max))

        def _metric_float(metrics_dict, keys, default: float = 0.0) -> float:
            if isinstance(keys, str):
                keys = (keys,)
            for key in keys:
                if key not in metrics_dict:
                    continue
                value = metrics_dict.get(key)
                try:
                    if isinstance(value, (list, tuple)):
                        if len(value) <= 0:
                            continue
                        value = float(np.mean([float(v) for v in value]))
                    elif isinstance(value, torch.Tensor):
                        value = float(value.detach().float().mean().item())
                    else:
                        value = float(value)
                    if np.isfinite(value):
                        return value
                except Exception:
                    continue
            return float(default)

        def _record_merl_policy_health(metrics_dict) -> None:
            if train_mode != "MERL":
                return
            hard_limit = float(
                getattr(
                    self.config.actor_rollout_ref.actor,
                    "ppo_kl_hard_limit",
                    0.12,
                )
                or 0.12
            )
            actor_kl = _metric_float(
                metrics_dict,
                ("actor/ppo_kl", "actor/kl"),
                default=0.0,
            )
            hard_skip = max(
                _metric_float(
                    metrics_dict, "actor/ppo_kl_hard_skip_ratio", default=0.0
                ),
                _metric_float(
                    metrics_dict, "actor/optimizer_step_skipped_by_kl", default=0.0
                ),
            )
            critic_kl = _metric_float(metrics_dict, "critic/kl", default=0.0)
            critic_reward = _metric_float(
                metrics_dict,
                ("critic/rewards/mean", "critic/score/mean"),
                default=0.0,
            )
            target_wm_count = _metric_float(
                metrics_dict, "wm/target_wm_sample_count", default=0.0
            )
            realized_ratio = _metric_float(
                metrics_dict,
                "wm/ratio_realized_vs_target",
                default=(1.0 if target_wm_count <= 0.0 else 0.0),
            )
            real_success = _metric_float(
                metrics_dict,
                (
                    "rollout/real_success_rate",
                    "rollout/success_rate",
                    "paper_metrics/rollout_real_success_rate_sr",
                ),
                default=float("nan"),
            )
            eval_success = _metric_float(
                metrics_dict,
                (
                    "val/success_rate/all",
                    "val/success_rate/libero_10",
                    "eval/success_rate/all->val/success_rate/all",
                    "eval/success_rate/libero_10->val/success_rate/libero_10",
                ),
                default=float("nan"),
            )

            reasons = {
                "actor_hard_skip": hard_skip > 0.0,
                "critic_kl": critic_kl > wm_ratio_critic_kl_limit,
                "critic_reward": critic_reward < wm_ratio_critic_reward_min,
                "low_realization": (
                    target_wm_count > 0.0
                    and realized_ratio < wm_ratio_realized_min
                ),
            }
            cooldown_triggered = any(reasons.values())
            if cooldown_triggered and wm_ratio_cooldown_steps > 0:
                self._wm_ratio_cooldown_remaining = max(
                    int(getattr(self, "_wm_ratio_cooldown_remaining", 0) or 0),
                    int(wm_ratio_cooldown_steps),
                )
                self._wm_ratio_carry = 0.0
                self._r_wm = 0.0

            history = list(getattr(self, "_merl_policy_health_history", []))
            prev = history[-1] if len(history) > 0 else {}
            real_non_decreasing = True
            if np.isfinite(real_success) and np.isfinite(
                float(prev.get("real_success", float("nan")))
            ):
                real_non_decreasing = real_success >= float(prev["real_success"]) - 1e-6
            eval_non_decreasing = True
            if np.isfinite(eval_success) and np.isfinite(
                float(prev.get("eval_success", float("nan")))
            ):
                eval_non_decreasing = eval_success >= float(prev["eval_success"]) - 1e-6
            healthy_for_horizon = (
                actor_kl <= hard_limit
                and hard_skip <= 0.0
                and critic_reward >= wm_ratio_critic_reward_min
                and real_non_decreasing
                and eval_non_decreasing
            )
            history.append(
                {
                    "actor_kl": float(actor_kl),
                    "hard_skip": float(hard_skip),
                    "critic_kl": float(critic_kl),
                    "critic_reward": float(critic_reward),
                    "realized_ratio": float(realized_ratio),
                    "real_success": float(real_success),
                    "eval_success": float(eval_success),
                    "healthy_for_horizon": bool(healthy_for_horizon),
                }
            )
            self._merl_policy_health_history = history[-64:]
            self._last_wm_ratio_realized_vs_target = float(
                np.clip(realized_ratio, 0.0, 1.0)
            )
            self._last_wm_target_sample_count = float(max(0.0, target_wm_count))
            self._last_critic_kl = float(max(0.0, critic_kl))
            self._last_critic_reward_mean = float(critic_reward)

            metrics_dict["wm/ratio_cooldown_triggered"] = (
                1.0 if cooldown_triggered else 0.0
            )
            metrics_dict["wm/ratio_cooldown_remaining_next"] = float(
                int(getattr(self, "_wm_ratio_cooldown_remaining", 0) or 0)
            )
            for reason, active in reasons.items():
                metrics_dict[f"wm/ratio_cooldown_reason_{reason}"] = (
                    1.0 if active else 0.0
                )
            metrics_dict["wm/policy_health_actor_kl"] = float(actor_kl)
            metrics_dict["wm/policy_health_critic_kl"] = float(critic_kl)
            metrics_dict["wm/policy_health_critic_reward_mean"] = float(
                critic_reward
            )
            metrics_dict["wm/policy_health_realized_vs_target"] = float(
                realized_ratio
            )
            metrics_dict["wm/policy_health_for_horizon"] = (
                1.0 if healthy_for_horizon else 0.0
            )

        def _compute_wm_rollout_confidence(dp: DataProto):
            if not _is_valid_dataproto(dp):
                return None

            num_samples = len(dp)
            confidence = torch.ones(num_samples, dtype=torch.float32)
            sample_weight = torch.ones(num_samples, dtype=torch.float32)
            uncertainty = torch.zeros(num_samples, dtype=torch.float32)
            priority = torch.full(
                (num_samples,), imag_priority_eps, dtype=torch.float32
            )
            obs_error = torch.zeros(num_samples, dtype=torch.float32)
            done_error = torch.zeros(num_samples, dtype=torch.float32)

            compact_obs_error = dp.batch.get("wm_obs_error", None)
            compact_done_error = dp.batch.get("wm_done_error", None)
            wm_pred_valid = dp.batch.get("wm_pred_valid", None)

            if wm_pred_valid is None:
                pred_valid_mask = torch.ones(num_samples, dtype=torch.bool)
            else:
                if wm_pred_valid.dim() > 1:
                    wm_pred_valid = wm_pred_valid.view(num_samples, -1)[:, 0]
                pred_valid_mask = wm_pred_valid.detach().cpu().to(torch.bool)

            if compact_obs_error is not None:
                obs_error = (
                    compact_obs_error.detach()
                    .cpu()
                    .float()
                    .view(num_samples, -1)[:, 0]
                    .clamp_min(0.0)
                )
                if compact_done_error is not None:
                    done_error = (
                        compact_done_error.detach()
                        .cpu()
                        .float()
                        .view(num_samples, -1)[:, 0]
                        .clamp_min(0.0)
                    )

                invalid_mask = ~pred_valid_mask
                if invalid_mask.any():
                    obs_error[invalid_mask] = 1.0
                    done_error[invalid_mask] = 1.0

                total_error = (
                    imag_obs_error_scale * obs_error
                    + imag_done_error_scale * done_error
                )
                confidence = torch.exp(-total_error).clamp(0.05, 1.0)
                uncertainty = (1.0 - confidence).clamp(0.0, 1.0)
                sample_weight = (
                    imag_weight_min
                    + (1.0 - imag_weight_min) * torch.pow(confidence, imag_weight_eta)
                ).clamp(imag_weight_min, wm_sample_weight_max)
                priority = (imag_priority_eps + imag_priority_beta * uncertainty).clamp(
                    min=imag_priority_eps
                )

                return {
                    "confidence": confidence,
                    "sample_weight": sample_weight,
                    "uncertainty": uncertainty,
                    "priority": priority,
                    "obs_error": obs_error,
                    "done_error": done_error,
                    "pred_valid": pred_valid_mask.to(torch.float32),
                    "valid_prediction_ratio": float(
                        pred_valid_mask.to(torch.float32).mean().item()
                    ),
                }

            pred_video = dp.batch.get("video", None)
            env_video = dp.batch.get("env_video", None)
            env_dones = dp.batch.get("env_dones", None)
            finish_step = dp.batch.get("finish_step", None)

            finish_step_cpu = None
            if finish_step is not None:
                finish_step_cpu = finish_step.detach().cpu().float()

            if (
                pred_video is not None
                and env_video is not None
                and pred_video.shape == env_video.shape
            ):
                pred_video_cpu = pred_video.detach().cpu().float() / 255.0
                env_video_cpu = env_video.detach().cpu().float() / 255.0
                max_frames = pred_video_cpu.shape[1]
                for idx in range(num_samples):
                    if not bool(pred_valid_mask[idx].item()):
                        continue
                    valid_frames = max_frames
                    if finish_step_cpu is not None:
                        valid_frames = int(
                            np.clip(finish_step_cpu[idx].item() + 1, 1, max_frames)
                        )
                    diff = (
                        pred_video_cpu[idx, :valid_frames]
                        - env_video_cpu[idx, :valid_frames]
                    )
                    obs_error[idx] = diff.pow(2).mean()

            if env_dones is not None and finish_step_cpu is not None:
                env_dones_cpu = env_dones.detach().cpu().float()
                max_frames = env_dones_cpu.shape[1]
                for idx in range(num_samples):
                    done_positions = torch.nonzero(
                        env_dones_cpu[idx] > 0.5, as_tuple=False
                    )
                    actual_finish = (
                        float(done_positions[0].item() + 1)
                        if done_positions.numel() > 0
                        else float(max_frames)
                    )
                    done_error[idx] = abs(
                        actual_finish - float(finish_step_cpu[idx].item())
                    ) / max(float(max_frames), 1.0)

            invalid_mask = ~pred_valid_mask
            if invalid_mask.any():
                obs_error[invalid_mask] = 1.0
                done_error[invalid_mask] = 1.0

            total_error = (
                imag_obs_error_scale * obs_error + imag_done_error_scale * done_error
            )
            confidence = torch.exp(-total_error).clamp(0.05, 1.0)
            uncertainty = (1.0 - confidence).clamp(0.0, 1.0)
            sample_weight = (
                imag_weight_min
                + (1.0 - imag_weight_min) * torch.pow(confidence, imag_weight_eta)
            ).clamp(imag_weight_min, wm_sample_weight_max)
            priority = (imag_priority_eps + imag_priority_beta * uncertainty).clamp(
                min=imag_priority_eps
            )

            return {
                "confidence": confidence,
                "sample_weight": sample_weight,
                "uncertainty": uncertainty,
                "priority": priority,
                "obs_error": obs_error,
                "done_error": done_error,
                "pred_valid": pred_valid_mask.to(torch.float32),
                "valid_prediction_ratio": float(
                    pred_valid_mask.to(torch.float32).mean().item()
                ),
            }

        def _add_pool_samples(
            pool,
            dataproto: DataProto,
            is_wm: bool,
            sample_weights=None,
            priorities=None,
        ):
            if not _is_valid_dataproto(dataproto):
                return
            dataproto = _attach_real_reward_anchors(dataproto, is_wm=is_wm)
            samples = []
            prios = []
            for idx in range(len(dataproto)):
                replay_dp = clone_dataproto_for_replay(
                    dataproto.slice(slice(idx, idx + 1))
                )
                sample = {
                    "dataproto": replay_dp,
                    "is_wm": bool(is_wm),
                    "sample_weight": float(
                        sample_weights[idx].item()
                        if sample_weights is not None
                        else 1.0
                    ),
                }
                priority_value = float(
                    priorities[idx].item() if priorities is not None else 1.0
                )
                samples.append(sample)
                prios.append(max(priority_value, 1e-6))
            if len(samples) > 0:
                pool.add(samples, priorities=prios)

        def _collect_wm_eval_metrics(eval_global_steps: int):
            wm_eval_metrics = {}
            wm_eval_max_batches = getattr(self, "wm_eval_max_batches", 10)

            if self.wm_trainer is None:
                wm_eval_metrics["wm/eval/skipped_no_wm_trainer"] = 1.0
                return wm_eval_metrics

            if fixed_eval_enabled:
                wm_eval_metrics["wm/eval/fixed_shared"] = 1.0

            try:
                eval_result = ray.get(
                    self.wm_trainer.evaluate_world_model.remote(
                        global_steps=eval_global_steps,
                        max_batches=wm_eval_max_batches,
                    )
                )
            except Exception as exc:
                print(f"[WM Eval] evaluate_world_model failed: {exc}")
                wm_eval_metrics["wm/eval/error"] = str(exc)
                return wm_eval_metrics

            if isinstance(eval_result, dict):
                for key, value in eval_result.items():
                    wm_eval_metrics[f"wm/eval/{key}"] = value

            return wm_eval_metrics

        # pre-train validation / rollouts if configured
        if self.config.trainer.get("rollout_before_train", False):
            print("Mode: Rollout before Train")
            self._save_rollouts(
                global_steps=global_steps,
                rollout_epoch=self.config.trainer.get("sim_rollout_epoch", 1000),
                use_wm=(train_mode == "MBRL"),
            )
            return

        if fixed_eval_enabled and self.config.trainer.get("val_before_train", False):
            print("Mode: WM fixed evaluation before Train...")
            initial_wm_eval_metrics = _collect_wm_eval_metrics(
                eval_global_steps=global_steps
            )
            if len(initial_wm_eval_metrics) > 0:
                pprint(f"Initial WM eval metrics: {initial_wm_eval_metrics}")
                logger.log(data=initial_wm_eval_metrics, step=global_steps)

        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", False
        ):
            print("Mode: Validation before Train...")
            val_metrics = self._validate(global_steps=global_steps)
            print("[ray_trainer eval]: Get all val_metrics!")
            val_metrics = {f"val/{key}": val for key, val in val_metrics.items()}
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=global_steps)
            if self.config.trainer.get("val_only", False):
                print("Only evaluation")
                return

        print("###### Start Training Now (fit_wm_v4) ######")
        self._training_started = time.monotonic()

        # main loop
        # Resume from the recorded epoch boundary (coarse-grained recovery by design).
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            if self._training_step_limit_reached(global_steps):
                break
            try:
                self.train_dataloader.start_new_epoch()
            except Exception:
                print(
                    f"[Epoch] Warning: start_new_epoch() raised an exception at epoch {epoch}"
                )
                self._dataloader_restarted = False
                print(
                    f"-------------------- [Epoch] Start epoch: {epoch} / {self.config.trainer.total_epochs} --------------------"
                )

            while not self._training_step_limit_reached(global_steps):
                valid_batch = None
                calibration_real_batch = None
                buffer_batch = []
                total_needed = batch_size * n_samples

                policy_mode = "MBRL" if train_mode == "ONLINE_MBRL" else train_mode
                planned_r_wm_prev = 0.0
                real_collection_prompt_target = batch_size
                real_collection_sample_target = total_needed
                valid_batch_target_size = total_needed
                weak_update_active = False
                weak_update_calibration_step = False
                last_weak_update_calibration_step = getattr(
                    self, "_wm_weak_update_last_calibration_step", None
                )
                if train_mode in ("MBRL", "ONLINE_MBRL"):
                    planned_r_wm_prev = 1.0
                    real_collection_prompt_target = 0
                    real_collection_sample_target = 0
                    valid_batch_target_size = total_needed
                    if train_mode == "ONLINE_MBRL":
                        real_collection_prompt_target = batch_size
                        real_collection_sample_target = total_needed
                elif (
                    train_mode == "MERL"
                    and update_wm_effective
                    and global_steps >= wm_warmup_steps
                ):
                    planned_r_wm_prev = _clip_wm_ratio(
                        getattr(self, "_r_wm", 0.0) or 0.0
                    )
                    weak_update_active = bool(
                        getattr(self, "_wm_weak_update_active", False)
                    )
                    if (
                        wm_weak_update_enable
                        and planned_r_wm_prev >= wm_weak_update_ratio_threshold
                    ):
                        weak_update_active = True
                    if weak_update_active:
                        policy_mode = "MBRL"
                        self._wm_weak_update_active = True
                        weak_update_calibration_step = bool(
                            wm_sparse_real_prompts > 0
                            and (
                                last_weak_update_calibration_step is None
                                or (
                                    global_steps
                                    - int(last_weak_update_calibration_step)
                                )
                                >= wm_sparse_real_interval
                            )
                        )
                        real_collection_prompt_target = (
                            wm_sparse_real_prompts
                            if weak_update_calibration_step
                            else 0
                        )
                        real_collection_sample_target = (
                            real_collection_prompt_target * n_samples
                        )
                        valid_batch_target_size = total_needed
                    else:
                        self._wm_weak_update_active = False
                        planned_real_samples = int(
                            np.ceil(total_needed * (1.0 - planned_r_wm_prev))
                        )
                        real_collection_prompt_target = _ceil_div(
                            planned_real_samples, n_samples
                        )
                        if wm_inner_steps > 0:
                            real_collection_prompt_target = max(
                                real_collection_prompt_target,
                                real_prompt_min_for_wm,
                            )
                        real_collection_prompt_target = int(
                            np.clip(real_collection_prompt_target, 0, batch_size)
                        )
                        real_collection_sample_target = max(
                            real_collection_prompt_target * n_samples,
                            0,
                        )
                        valid_batch_target_size = real_collection_sample_target

                if self.train_dataloader.buffer_size() > 0:
                    buffer_batch = self.train_dataloader.get_from_buffer(
                        batch_size, self.actor_rollout_wg.world_size
                    )

                metrics = defaultdict(list)
                metrics["timing/gen"] = 0
                metrics["timing/verify"] = 0
                metrics["timing/acc&trunc_filter"] = 0
                metrics["timing/filter_format_error"] = 0
                metrics["timing/compute_all_entropy"] = 0
                metrics["wm/warmup_steps_remaining"] = float(
                    max(0, wm_warmup_steps - global_steps)
                )
                metrics["wm/ratio_wm_prev"] = float(planned_r_wm_prev)
                metrics["wm/ratio_real_prev"] = float(1.0 - planned_r_wm_prev)
                metrics["env/real_prompt_target"] = float(real_collection_prompt_target)
                metrics["env/real_sample_target"] = float(real_collection_sample_target)
                metrics["env/policy_sample_target"] = float(valid_batch_target_size)
                metrics["env/total_sample_target"] = float(total_needed)
                metrics["wm/weak_update_active"] = 1.0 if weak_update_active else 0.0
                metrics["wm/weak_update_calibration_step"] = (
                    1.0 if weak_update_calibration_step else 0.0
                )
                metrics["wm/weak_update_ratio_threshold"] = float(
                    wm_weak_update_ratio_threshold
                )
                metrics["wm/weak_update_exit_ratio_threshold"] = float(
                    wm_weak_update_exit_ratio_threshold
                )
                metrics["wm/weak_update_sparse_interval"] = float(
                    wm_sparse_real_interval
                )
                metrics["wm/weak_update_last_calibration_step"] = float(
                    last_weak_update_calibration_step
                    if last_weak_update_calibration_step is not None
                    else -1
                )
                if train_mode == "MERL" and global_steps < wm_warmup_steps:
                    metrics["wm/update/skipped_warmup"] = 1.0
                    metrics["wm/ratio_target_confidence"] = 0.0
                    metrics["wm/ratio_target_combined"] = 0.0
                    metrics["wm/ratio_scheduler_signal_active"] = 0.0
                if weak_update_active and (not weak_update_calibration_step):
                    metrics["wm/update/skipped_sparse_weak"] = 1.0

                print("### 1. Generate batches by LIBERO. ###")
                print(f"batch size: {batch_size}, sample nums: {n_samples}")

                # For MBRL we may pre-generate WM rollouts
                pre_wm_roll_batch = None

                while True:
                    current_len = (
                        len(valid_batch) if _is_valid_dataproto(valid_batch) else 0
                    )
                    print(
                        f"[Batch] len of valid_batch: {current_len}, expect: {valid_batch_target_size}"
                    )

                    if current_len >= valid_batch_target_size:
                        print(
                            "[Batch] Sufficient data collected, breaking collection loop."
                        )
                        break

                    batch_dict = None
                    try:
                        batch_dict = self.train_dataloader.get_next_batch()
                    except StopIteration:
                        if not getattr(self, "_dataloader_restarted", False):
                            print(
                                f"[Batch] StopIteration from train_dataloader -- attempting to start a new epoch and retry once (epoch={epoch})"
                            )
                            try:
                                self.train_dataloader.start_new_epoch()
                                self._dataloader_restarted = True
                                batch_dict = self.train_dataloader.get_next_batch()
                            except StopIteration:
                                print(
                                    "[Batch] After restart, train_dataloader still exhausted. Breaking batch collection."
                                )
                                break
                            except Exception as e:
                                print(
                                    f"[Batch] Error after restarting train_dataloader: {e}. Breaking batch collection."
                                )
                                break
                        else:
                            print(
                                "[Batch] StopIteration and already restarted once this epoch. Breaking batch collection."
                            )
                            break
                    except Exception as e:
                        print(
                            f"[Batch] Error getting batch from train_dataloader: {e}. Breaking batch collection."
                        )
                        break

                    if batch_dict is None:
                        print(
                            "[Batch] batch_dict is None after exception handling, breaking."
                        )
                        break

                    print("[Batch] Start one batch generation!")
                    with Timer(
                        name="gen", text="{name}: {seconds:.1f} seconds"
                    ) as timer:
                        newbatch: DataProto = DataProto.from_single_dict(batch_dict)

                        if len(newbatch) == 0:
                            print("[Batch] newbatch is empty, skip this batch.")
                            continue

                        if len(buffer_batch) > 0:
                            newbatch = DataProto.concat([buffer_batch, newbatch])
                            buffer_batch = []

                        if policy_mode != "MBRL":
                            remaining_real_samples = max(
                                0, valid_batch_target_size - current_len
                            )
                            remaining_real_prompts = _ceil_div(
                                remaining_real_samples, n_samples
                            )
                            if remaining_real_prompts <= 0:
                                print(
                                    "[Batch] Real rollout target already satisfied, stop collecting more real prompts."
                                )
                                break
                            dispatch_real_prompts = _dispatch_safe_prompt_count(
                                remaining_real_prompts,
                                len(newbatch),
                                context="real rollout top-up",
                            )
                            if dispatch_real_prompts <= 0:
                                print(
                                    "[Batch] No dispatchable prompts for real rollout top-up; skip this batch."
                                )
                                continue
                            if dispatch_real_prompts < len(newbatch):
                                newbatch, buffer_remainder = _split_prompt_batch(
                                    newbatch, dispatch_real_prompts
                                )
                                if _is_valid_dataproto(buffer_remainder):
                                    self.train_dataloader.add_to_buffer(
                                        buffer_remainder
                                    )

                        # GRPO groups outcomes by uid, so generate a stable prompt-level
                        # uid before prompt replication and carry it through real/imagined rollouts.
                        newbatch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(newbatch.batch))],
                            dtype=object,
                        )
                        newbatch = _pad_prompt_batch_for_dispatch(
                            newbatch,
                            context="policy prompt dispatch",
                        )

                        if "robotwin" in self.config.data.task_suite_name:
                            gen_batch = newbatch.select(
                                batch_keys=["task_id", "trial_id", "trial_seed"],
                                non_tensor_batch_keys={"task_suite_name", "uid"},
                                meta_info_keys={},
                            )
                        else:
                            gen_batch = newbatch.select(
                                batch_keys=["task_id", "trial_id"],
                                non_tensor_batch_keys={"task_suite_name", "uid"},
                                meta_info_keys={},
                            )

                        batch_lst = sum(
                            [
                                [newbatch[i : i + 1] for _ in range(n_samples)]
                                for i in range(len(newbatch))
                            ],
                            [],
                        )

                        gen_batch.meta_info = {
                            "eos_token_id": self.tokenizer.eos_token_id,
                            "n_samples": n_samples,
                            "pad_token_id": self.tokenizer.pad_token_id,
                        }

                        has_wm_tmp_consumer = self.wm_trainer is not None
                        will_use_online_wm_step = bool(
                            has_wm_tmp_consumer
                            and update_wm_effective
                            and (global_steps >= wm_warmup_steps)
                            and (
                                (not weak_update_active) or weak_update_calibration_step
                            )
                        )
                        should_save_real_train_shards = bool(
                            will_use_online_wm_step and wm_inner_steps > 0
                        )
                        should_save_online_eval_shards = bool(
                            will_use_online_wm_step
                            and (not weak_update_active)
                            and (not fixed_eval_enabled)
                            and wm_eval_interval is not None
                            and int(wm_eval_interval) > 0
                            and (global_steps + 1) % int(wm_eval_interval) == 0
                        )
                        metrics["wm/io/fixed_eval_enabled"] = (
                            1.0 if fixed_eval_enabled else 0.0
                        )
                        metrics["wm/io/warmup_active"] = (
                            1.0 if global_steps < wm_warmup_steps else 0.0
                        )
                        metrics["wm/io/write_train_real"] = (
                            1.0 if should_save_real_train_shards else 0.0
                        )
                        metrics["wm/io/write_eval_real"] = (
                            1.0 if should_save_online_eval_shards else 0.0
                        )
                        metrics["wm/io/persist_imag_shards"] = (
                            1.0 if persist_imag_rollout_shards else 0.0
                        )

                        if (
                            (train_mode == "ONLINE_MBRL" or (weak_update_active and weak_update_calibration_step))
                            and real_collection_prompt_target > 0
                            and not _is_valid_dataproto(calibration_real_batch)
                        ):
                            calibration_dispatch_prompts = _dispatch_safe_prompt_count(
                                real_collection_prompt_target,
                                len(newbatch),
                                context="weak-update real calibration",
                            )
                            calibration_prompt_batch, _ = _split_prompt_batch(
                                newbatch, calibration_dispatch_prompts
                            )
                            if _is_valid_dataproto(calibration_prompt_batch):
                                if "robotwin" in self.config.data.task_suite_name:
                                    calibration_gen_batch = (
                                        calibration_prompt_batch.select(
                                            batch_keys=[
                                                "task_id",
                                                "trial_id",
                                                "trial_seed",
                                            ],
                                            non_tensor_batch_keys={
                                                "task_suite_name",
                                                "uid",
                                            },
                                            meta_info_keys={},
                                        )
                                    )
                                else:
                                    calibration_gen_batch = (
                                        calibration_prompt_batch.select(
                                            batch_keys=["task_id", "trial_id"],
                                            non_tensor_batch_keys={
                                                "task_suite_name",
                                                "uid",
                                            },
                                            meta_info_keys={},
                                        )
                                    )

                                calibration_batch_lst = sum(
                                    [
                                        [
                                            calibration_prompt_batch[i : i + 1]
                                            for _ in range(n_samples)
                                        ]
                                        for i in range(len(calibration_prompt_batch))
                                    ],
                                    [],
                                )
                                calibration_gen_batch.meta_info = {
                                    "eos_token_id": self.tokenizer.eos_token_id,
                                    "n_samples": n_samples,
                                    "pad_token_id": self.tokenizer.pad_token_id,
                                }
                                calibration_gen_batch.meta_info["use_wm"] = False
                                calibration_gen_batch.meta_info["save_to_hdfs"] = (
                                    should_save_real_train_shards
                                )
                                calibration_gen_batch.meta_info["return_rollouts"] = (
                                    bool(should_save_real_train_shards)
                                )
                                calibration_gen_batch.meta_info[
                                    "strip_rollout_media"
                                ] = True
                                calibration_gen_batch.meta_info["train_split"] = (
                                    "train_real"
                                )
                                calibration_gen_batch.meta_info["eval_split"] = (
                                    "eval_real"
                                )
                                calibration_gen_batch.meta_info["rollout_base_dir"] = (
                                    self.config.actor_rollout_ref.rollout_base_dir
                                )
                                calibration_gen_batch.meta_info["global_steps"] = (
                                    global_steps
                                )
                                calibration_gen_batch.meta_info["save_eval"] = False
                                if train_rollout_max_steps > 0:
                                    calibration_gen_batch.meta_info["max_steps"] = (
                                        train_rollout_max_steps
                                    )

                                calibration_real_output = (
                                    self.actor_rollout_wg.generate_sequences(
                                        prompts=calibration_gen_batch
                                    )
                                )
                                roll_batch_real_calibration = DataProto.concat(
                                    calibration_batch_lst
                                )
                                roll_batch_real_calibration = (
                                    union_prompt_and_rollout_output(
                                        roll_batch_real_calibration,
                                        calibration_real_output,
                                        context="fit_wm_v5.calibration_real_rollout",
                                    )
                                )
                                _ensure_uid(roll_batch_real_calibration)

                                debug_dummy(
                                    roll_batch_real_calibration,
                                    "real_rollout_calibration (before filter)",
                                )
                                roll_batch_real_calibration = (
                                    DataProtoFilter.filter_rollout_samples(
                                        roll_batch_real_calibration
                                    )
                                )
                                debug_dummy(
                                    roll_batch_real_calibration,
                                    "real_rollout_calibration (after filter)",
                                )

                                if _is_valid_dataproto(roll_batch_real_calibration):
                                    calibration_real_batch = roll_batch_real_calibration
                                    self._wm_weak_update_last_calibration_step = int(
                                        global_steps
                                    )
                                    metrics["wm/weak_update_last_calibration_step"] = (
                                        float(global_steps)
                                    )
                                    metrics[
                                        "wm/weak_update_calibration_real_samples"
                                    ] = float(len(calibration_real_batch))
                                    print(
                                        "[WM DATA] Collected real simulator-training batch.",
                                        flush=True,
                                    )

                        # Mode-sensitive real rollouts: skip real env generation for MBRL
                        if policy_mode != "MBRL":
                            gen_batch.meta_info["use_wm"] = False
                            gen_batch.meta_info["save_to_hdfs"] = (
                                should_save_real_train_shards
                            )
                            gen_batch.meta_info["return_rollouts"] = bool(
                                should_save_real_train_shards
                                or should_save_online_eval_shards
                            )
                            gen_batch.meta_info["strip_rollout_media"] = True
                            gen_batch.meta_info["train_split"] = "train_real"
                            gen_batch.meta_info["eval_split"] = "eval_real"
                            gen_batch.meta_info["rollout_base_dir"] = (
                                self.config.actor_rollout_ref.rollout_base_dir
                            )
                            gen_batch.meta_info["global_steps"] = global_steps
                            gen_batch.meta_info["save_eval"] = (
                                should_save_online_eval_shards
                            )
                            if train_rollout_max_steps > 0:
                                gen_batch.meta_info["max_steps"] = (
                                    train_rollout_max_steps
                                )

                            real_gen_output = self.actor_rollout_wg.generate_sequences(
                                prompts=gen_batch
                            )

                            roll_batch_real = DataProto.concat(batch_lst)
                            roll_batch_real = union_prompt_and_rollout_output(
                                roll_batch_real,
                                real_gen_output,
                                context="fit_wm_v5.real_rollout",
                            )

                            roll_batch_to_add_real = roll_batch_real
                            _ensure_uid(roll_batch_to_add_real)

                            debug_dummy(
                                roll_batch_to_add_real, "real_rollout (before filter)"
                            )
                            roll_batch_to_add_real = (
                                DataProtoFilter.filter_rollout_samples(
                                    roll_batch_to_add_real
                                )
                            )
                            debug_dummy(
                                roll_batch_to_add_real, "real_rollout (after filter)"
                            )

                            if not _is_valid_dataproto(valid_batch):
                                valid_batch = roll_batch_to_add_real
                            else:
                                valid_batch = DataProto.concat(
                                    [valid_batch, roll_batch_to_add_real]
                                )
                            _ensure_uid(valid_batch)
                            print("[Batch] Finished one REAL rollout generation!")

                        else:
                            # MBRL: pre-generate WM rollout for this prompt set and use as valid_batch
                            try:
                                prompts_for_wm = gen_batch
                                prompts_for_wm.meta_info["use_wm"] = True
                                prompts_for_wm.meta_info["n_samples"] = n_samples
                                prompts_for_wm.meta_info["save_to_hdfs"] = (
                                    persist_imag_rollout_shards and has_wm_tmp_consumer
                                )
                                prompts_for_wm.meta_info["return_rollouts"] = bool(
                                    prompts_for_wm.meta_info["save_to_hdfs"]
                                )
                                prompts_for_wm.meta_info["strip_rollout_media"] = True
                                prompts_for_wm.meta_info["train_split"] = "imag_train"
                                prompts_for_wm.meta_info["eval_split"] = "eval_real"
                                prompts_for_wm.meta_info["rollout_base_dir"] = (
                                    self.config.actor_rollout_ref.rollout_base_dir
                                )
                                prompts_for_wm.meta_info["global_steps"] = global_steps
                                prompts_for_wm.meta_info["save_eval"] = False
                                prompts_for_wm.meta_info["max_steps"] = int(
                                    np.clip(
                                        getattr(
                                            self,
                                            "_imagined_horizon",
                                            imag_horizon_max,
                                        ),
                                        imag_horizon_min,
                                        imag_horizon_max,
                                    )
                                )

                                pre_wm_roll_batch = (
                                    self.actor_rollout_wg.generate_sequences(
                                        prompts=prompts_for_wm
                                    )
                                )

                                if (
                                    pre_wm_roll_batch is None
                                    or len(pre_wm_roll_batch) == 0
                                ):
                                    print(
                                        "[MBRL] WM generation returned empty; skipping this batch."
                                    )
                                    continue

                                _ensure_uid(pre_wm_roll_batch)

                                debug_dummy(
                                    pre_wm_roll_batch, "wm_rollout_mbrl (before filter)"
                                )
                                pre_wm_roll_batch = (
                                    DataProtoFilter.filter_rollout_samples(
                                        pre_wm_roll_batch
                                    )
                                )
                                debug_dummy(
                                    pre_wm_roll_batch, "wm_rollout_mbrl (after filter)"
                                )

                                if not _is_valid_dataproto(valid_batch):
                                    valid_batch = pre_wm_roll_batch
                                else:
                                    valid_batch = DataProto.concat(
                                        [valid_batch, pre_wm_roll_batch]
                                    )
                                _ensure_uid(valid_batch)

                                print(
                                    "[Batch] Finished one WM-only rollout generation (MBRL mode)."
                                )
                            except Exception as e:
                                print(
                                    f"[MBRL] WM generation failed during pre-generation: {e}"
                                )
                                continue

                        metrics["timing/gen"] += timer.last

                    current_len = (
                        len(valid_batch) if _is_valid_dataproto(valid_batch) else 0
                    )
                    if current_len < valid_batch_target_size:
                        print(
                            f"[Epoch] len of valid_batch: {current_len}, expect: {valid_batch_target_size}, continue collecting."
                        )
                        continue

                    if current_len > valid_batch_target_size:
                        valid_batch = self.add_to_buffer(
                            valid_batch, batch_size, n_samples
                        )
                        _ensure_uid(valid_batch)
                        print(
                            f"[Epoch] len of valid_batch: {len(valid_batch)}, expect: {valid_batch_target_size}, reorg these batches."
                        )
                    else:
                        print(
                            f"[Epoch] len of valid_batch: {len(valid_batch)}, expect: {valid_batch_target_size}, start training these batches."
                        )

                    break

                if not _is_valid_dataproto(valid_batch):
                    print(
                        f"[Epoch] valid_batch is None or empty after collection, skip this epoch iteration."
                    )
                    if self.train_dataloader.buffer_size() > 0:
                        print(
                            "[Epoch] Trying to get more data from buffer for next iteration."
                        )
                        continue
                    else:
                        print("[Epoch] Buffer is also empty, breaking epoch loop.")
                        break

                valid_batch = clone_dataproto_for_replay(valid_batch)
                if train_mode == "ONLINE_MBRL" and not _is_valid_dataproto(calibration_real_batch):
                    raise RuntimeError("ONLINE_MBRL requires a valid real-data batch for every WM update")
                _ensure_uid(valid_batch)
                valid_batch = _attach_real_reward_anchors(
                    valid_batch, is_wm=(policy_mode == "MBRL")
                )
                real_anchor_reward_lookup = _build_anchor_reward_lookup(
                    valid_batch, real_only=False
                )
                metrics["wm/real_anchor_lookup_size"] = float(
                    len(real_anchor_reward_lookup)
                )
                if len(real_anchor_reward_lookup) > 0:
                    metrics["wm/real_anchor_lookup_reward_mean"] = float(
                        np.mean(list(real_anchor_reward_lookup.values()))
                    )
                else:
                    metrics["wm/real_anchor_lookup_reward_mean"] = 0.0
                if _is_valid_dataproto(calibration_real_batch):
                    calibration_real_batch = clone_dataproto_for_replay(
                        calibration_real_batch
                    )
                    _ensure_uid(calibration_real_batch)
                    calibration_real_batch = _attach_real_reward_anchors(
                        calibration_real_batch, is_wm=False
                    )
                    calibration_anchor_lookup = _build_anchor_reward_lookup(
                        calibration_real_batch, real_only=False
                    )
                    for key, value in calibration_anchor_lookup.items():
                        real_anchor_reward_lookup[key] = max(
                            float(real_anchor_reward_lookup.get(key, 0.0)),
                            float(value),
                        )
                    metrics["wm/real_anchor_lookup_size"] = float(
                        len(real_anchor_reward_lookup)
                    )

                print(
                    "### 2. Generate WM imagined rollouts and compute mixing weight (confidence-driven). ###"
                )

                current_imagined_horizon = int(
                    np.clip(
                        getattr(self, "_imagined_horizon", imag_horizon_min),
                        imag_horizon_min,
                        imag_horizon_max,
                    )
                )
                metrics["wm/current_imag_horizon"] = current_imagined_horizon

                # decide use_wm_now depending on mode
                if train_mode in ("MBRL", "ONLINE_MBRL"):
                    use_wm_now = True
                else:
                    use_wm_now = update_wm_effective and (
                        global_steps >= wm_warmup_steps
                    )

                r_wm = 0.0
                wm_roll_batch = None
                wm_confidence_stats = None
                wm_ratio_eval_batch = None

                if use_wm_now and policy_mode != "MFRL":
                    if policy_mode == "MBRL" and _is_valid_dataproto(valid_batch):
                        wm_roll_batch = valid_batch
                        print("[WM] Using prompt-only WM rollouts for policy batch.")
                        if (
                            weak_update_active
                            and weak_update_calibration_step
                            and _is_valid_dataproto(calibration_real_batch)
                        ):
                            prompts_for_ratio = _select_prompt_representatives(
                                calibration_real_batch
                            )
                            prompts_for_ratio = _pad_prompt_batch_for_dispatch(
                                prompts_for_ratio,
                                context="WM ratio calibration dispatch",
                            )

                            prompts_for_ratio.meta_info["use_wm"] = True
                            prompts_for_ratio.meta_info["n_samples"] = n_samples
                            prompts_for_ratio.meta_info["save_to_hdfs"] = False
                            prompts_for_ratio.meta_info["return_rollouts"] = False
                            prompts_for_ratio.meta_info["strip_rollout_media"] = True
                            prompts_for_ratio.meta_info["train_split"] = "imag_train"
                            prompts_for_ratio.meta_info["eval_split"] = "eval_real"
                            prompts_for_ratio.meta_info["rollout_base_dir"] = (
                                self.config.actor_rollout_ref.rollout_base_dir
                            )
                            prompts_for_ratio.meta_info["global_steps"] = global_steps
                            prompts_for_ratio.meta_info["save_eval"] = False
                            prompts_for_ratio.meta_info["max_steps"] = (
                                current_imagined_horizon
                            )

                            with Timer(
                                name="gen_wm_cal",
                                text="{name}: {seconds:.1f} seconds",
                            ) as timer_wm_cal:
                                wm_ratio_eval_batch = (
                                    self.actor_rollout_wg.generate_sequences(
                                        prompts=prompts_for_ratio
                                    )
                                )
                                metrics["timing/gen"] += timer_wm_cal.last

                            if _is_valid_dataproto(wm_ratio_eval_batch):
                                _ensure_uid(wm_ratio_eval_batch)
                                debug_dummy(
                                    wm_ratio_eval_batch,
                                    "wm_ratio_eval_batch (before filter)",
                                )
                                wm_ratio_eval_batch = (
                                    DataProtoFilter.filter_rollout_samples(
                                        wm_ratio_eval_batch
                                    )
                                )
                                debug_dummy(
                                    wm_ratio_eval_batch,
                                    "wm_ratio_eval_batch (after filter)",
                                )
                    else:
                        if not _is_valid_dataproto(valid_batch):
                            print("[WM] valid_batch is invalid, skip WM generation.")
                            use_wm_now = False
                        else:
                            prompts_for_wm = _select_prompt_representatives(valid_batch)
                            prompts_for_wm = _pad_prompt_batch_for_dispatch(
                                prompts_for_wm,
                                context="WM imagined rollout dispatch",
                            )

                            prompts_for_wm.meta_info["use_wm"] = True
                            prompt_count_for_wm = max(len(prompts_for_wm), 1)
                            planned_wm_sample_target = max(
                                1, total_needed - max(valid_batch_target_size, 0)
                            )
                            wm_rollout_n_samples = _ceil_div(
                                planned_wm_sample_target, prompt_count_for_wm
                            )
                            wm_rollout_n_samples = int(
                                np.clip(
                                    wm_rollout_n_samples,
                                    1,
                                    wm_rollout_n_samples_max,
                                )
                            )
                            prompts_for_wm.meta_info["n_samples"] = wm_rollout_n_samples
                            prompts_for_wm.meta_info["save_to_hdfs"] = (
                                persist_imag_rollout_shards and has_wm_tmp_consumer
                            )
                            prompts_for_wm.meta_info["return_rollouts"] = bool(
                                prompts_for_wm.meta_info["save_to_hdfs"]
                            )
                            prompts_for_wm.meta_info["strip_rollout_media"] = True
                            prompts_for_wm.meta_info["train_split"] = "imag_train"
                            prompts_for_wm.meta_info["eval_split"] = "eval_real"
                            prompts_for_wm.meta_info["rollout_base_dir"] = (
                                self.config.actor_rollout_ref.rollout_base_dir
                            )
                            prompts_for_wm.meta_info["global_steps"] = global_steps
                            prompts_for_wm.meta_info["save_eval"] = False
                            prompts_for_wm.meta_info["max_steps"] = (
                                current_imagined_horizon
                            )
                            metrics["wm/rollout_n_samples"] = float(
                                wm_rollout_n_samples
                            )

                            with Timer(
                                name="gen_wm", text="{name}: {seconds:.1f} seconds"
                            ) as timer_wm:
                                wm_gen_output = (
                                    self.actor_rollout_wg.generate_sequences(
                                        prompts=prompts_for_wm
                                    )
                                )
                                metrics["timing/gen"] += timer_wm.last
                                print("[Batch] Generated WM IMAGINED rollouts.")
                                wm_roll_batch = wm_gen_output

                            if _is_valid_dataproto(wm_roll_batch):
                                _ensure_uid(wm_roll_batch)
                                debug_dummy(wm_roll_batch, "wm_rollout (before filter)")
                                wm_roll_batch = DataProtoFilter.filter_rollout_samples(
                                    wm_roll_batch
                                )
                                debug_dummy(wm_roll_batch, "wm_rollout (after filter)")
                            else:
                                wm_roll_batch = None

                    wm_roll_batch = _apply_anchor_reward_lookup(
                        wm_roll_batch, real_anchor_reward_lookup, wm_only=True
                    )
                    wm_roll_valid_stats = _valid_response_token_stats(
                        wm_roll_batch, wm_only=False
                    )
                    metrics["wm/rollout_valid_sample_count"] = float(
                        wm_roll_valid_stats["valid_sample_count"]
                    )
                    metrics["wm/rollout_valid_token_count"] = float(
                        wm_roll_valid_stats["token_count"]
                    )
                    metrics["wm/rollout_valid_token_mean"] = float(
                        wm_roll_valid_stats["mean_tokens"]
                    )
                    if (
                        _is_valid_dataproto(wm_roll_batch)
                        and wm_roll_valid_stats["token_count"] <= 0
                    ):
                        message = (
                            "[WM CONTRACT] generated imagined rollout has no valid "
                            "response tokens after dummy/pred-valid filtering."
                        )
                        metrics["wm/valid_token_gate_blocked"] = 1.0
                        if strict_mode_assert and train_mode == "MERL":
                            raise RuntimeError(message)
                        print(message + " Fallback to pure real batch.", flush=True)
                        wm_roll_batch = None
                        self._r_wm = 0.0

                    if weak_update_active:
                        scheduler_ratio = _clip_wm_ratio(
                            getattr(self, "_r_wm", planned_r_wm_prev)
                            or planned_r_wm_prev
                        )
                        if weak_update_calibration_step and _is_valid_dataproto(
                            wm_ratio_eval_batch
                        ):
                            wm_confidence_stats = _compute_wm_rollout_confidence(
                                wm_ratio_eval_batch
                            )
                            pred_valid_ratio = float(
                                wm_confidence_stats.get("valid_prediction_ratio", 0.0)
                            )
                            metrics["wm/pred_valid_ratio"] = pred_valid_ratio
                            if pred_valid_ratio <= 0.0:
                                scheduler_ratio = 0.0
                                self._r_wm = 0.0
                                self._wm_weak_update_active = False
                                metrics["wm/weak_update_exit_pending"] = 1.0
                                metrics["wm/ratio_target_confidence"] = 0.0
                                metrics["wm/ratio_target_combined"] = 0.0
                                print(
                                    "[WM WEAK] Sparse calibration produced no valid WM prediction; exit weak-update next step.",
                                    flush=True,
                                )
                            else:
                                batch_confidence = float(
                                    wm_confidence_stats["confidence"].mean().item()
                                )
                                prev_confidence_ema = getattr(
                                    self, "_imag_confidence_ema", None
                                )
                                if prev_confidence_ema is None:
                                    confidence_ema = batch_confidence
                                else:
                                    confidence_ema = (
                                        imag_confidence_ema_alpha * prev_confidence_ema
                                        + (1.0 - imag_confidence_ema_alpha)
                                        * batch_confidence
                                    )
                                self._imag_confidence_ema = float(confidence_ema)
                                self._imagined_horizon = (
                                    _compute_horizon_from_confidence(confidence_ema)
                                )

                                confidence_target = _compute_alpha_from_confidence(
                                    batch_confidence
                                )
                                signal_target, signal_rel_improve, signal_metrics = (
                                    _compute_ratio_signal_target()
                                )
                                combined_target = confidence_target
                                if signal_target is not None:
                                    combined_target = min(
                                        confidence_target, signal_target
                                    )
                                scheduler_ratio = _track_wm_ratio(
                                    combined_target, signal_rel_improve
                                )
                                self._r_wm = float(scheduler_ratio)

                                metrics["wm/chunk_confidence_mean"] = batch_confidence
                                metrics["wm/chunk_confidence_min"] = float(
                                    wm_confidence_stats["confidence"].min().item()
                                )
                                metrics["wm/chunk_confidence_max"] = float(
                                    wm_confidence_stats["confidence"].max().item()
                                )
                                metrics["wm/chunk_obs_error_mean"] = float(
                                    wm_confidence_stats["obs_error"].mean().item()
                                )
                                metrics["wm/chunk_done_error_mean"] = float(
                                    wm_confidence_stats["done_error"].mean().item()
                                )
                                metrics["wm/confidence_ema"] = float(confidence_ema)
                                metrics["wm/ratio_target_confidence"] = float(
                                    confidence_target
                                )
                                metrics["wm/ratio_target_combined"] = float(
                                    combined_target
                                )
                                metrics["wm/ratio_scheduler_signal_used"] = (
                                    1.0 if signal_target is not None else 0.0
                                )
                                metrics.update(signal_metrics)
                                if (
                                    scheduler_ratio
                                    < wm_weak_update_exit_ratio_threshold
                                ):
                                    self._wm_weak_update_active = False
                                    metrics["wm/weak_update_exit_pending"] = 1.0
                                else:
                                    self._wm_weak_update_active = True
                                print(
                                    "[WM WEAK] "
                                    f"pred_valid_ratio={pred_valid_ratio:.4f}, "
                                    f"batch_conf={batch_confidence:.4f}, "
                                    f"conf_ema={confidence_ema:.4f}, "
                                    f"scheduler_ratio={scheduler_ratio:.4f}, "
                                    f"policy_alpha=1.0000, "
                                    f"next_horizon={self._imagined_horizon}",
                                    flush=True,
                                )
                        elif weak_update_calibration_step:
                            scheduler_ratio = 0.0
                            self._r_wm = 0.0
                            self._wm_weak_update_active = False
                            metrics["wm/weak_update_exit_pending"] = 1.0
                            metrics["wm/weak_update_missing_calibration"] = 1.0
                        else:
                            metrics["wm/weak_update_skip_calibration"] = 1.0

                        r_wm = 1.0
                        metrics["wm/ratio_real"] = 0.0
                        metrics["wm/ratio_wm"] = 1.0
                        metrics["wm/ratio_real_scheduler"] = float(
                            1.0 - self._safe_float(getattr(self, "_r_wm", None), 0.0)
                        )
                        metrics["wm/ratio_wm_scheduler"] = float(
                            self._safe_float(getattr(self, "_r_wm", None), 0.0)
                        )
                        metrics["wm/next_imag_horizon"] = int(
                            getattr(self, "_imagined_horizon", current_imagined_horizon)
                        )
                    elif policy_mode == "MBRL":
                        # force pure WM
                        r_wm = 1.0
                        self._r_wm = 1.0
                        metrics["wm/ratio_real"] = 0.0
                        metrics["wm/ratio_wm"] = 1.0
                        metrics["wm/next_imag_horizon"] = current_imagined_horizon
                        print("[Mode MBRL] forcing r_wm=1.0 (pure WM imagined data)")
                    elif _is_valid_dataproto(wm_roll_batch):
                        wm_confidence_stats = _compute_wm_rollout_confidence(
                            wm_roll_batch
                        )
                        pred_valid_ratio = float(
                            wm_confidence_stats.get("valid_prediction_ratio", 0.0)
                        )
                        metrics["wm/pred_valid_ratio"] = pred_valid_ratio
                        if pred_valid_ratio <= 0.0:
                            wm_roll_batch = None
                            r_wm = 0.0
                            self._r_wm = 0.0
                            metrics["wm/ratio_real"] = 1.0
                            metrics["wm/ratio_wm"] = 0.0
                            metrics["wm/next_imag_horizon"] = current_imagined_horizon
                            print(
                                "[WM CONF] no valid WM predictions were produced; fallback to pure real batch.",
                                flush=True,
                            )
                        else:
                            batch_confidence = float(
                                wm_confidence_stats["confidence"].mean().item()
                            )
                            prev_confidence_ema = getattr(
                                self, "_imag_confidence_ema", None
                            )
                            if prev_confidence_ema is None:
                                confidence_ema = batch_confidence
                            else:
                                confidence_ema = (
                                    imag_confidence_ema_alpha * prev_confidence_ema
                                    + (1.0 - imag_confidence_ema_alpha)
                                    * batch_confidence
                                )
                            self._imag_confidence_ema = float(confidence_ema)
                            self._imagined_horizon = _compute_horizon_from_confidence(
                                confidence_ema
                            )

                            confidence_target = _compute_alpha_from_confidence(
                                batch_confidence
                            )
                            signal_target, signal_rel_improve, signal_metrics = (
                                _compute_ratio_signal_target()
                            )
                            combined_target = confidence_target
                            if signal_target is not None:
                                combined_target = min(confidence_target, signal_target)
                            r_wm = _track_wm_ratio(combined_target, signal_rel_improve)
                            self._r_wm = float(r_wm)

                            metrics["wm/chunk_confidence_mean"] = batch_confidence
                            metrics["wm/chunk_confidence_min"] = float(
                                wm_confidence_stats["confidence"].min().item()
                            )
                            metrics["wm/chunk_confidence_max"] = float(
                                wm_confidence_stats["confidence"].max().item()
                            )
                            metrics["wm/chunk_obs_error_mean"] = float(
                                wm_confidence_stats["obs_error"].mean().item()
                            )
                            metrics["wm/chunk_done_error_mean"] = float(
                                wm_confidence_stats["done_error"].mean().item()
                            )
                            metrics["wm/confidence_ema"] = float(confidence_ema)
                            metrics["wm/ratio_target_confidence"] = float(
                                confidence_target
                            )
                            metrics["wm/ratio_target_combined"] = float(combined_target)
                            metrics["wm/ratio_scheduler_signal_used"] = (
                                1.0 if signal_target is not None else 0.0
                            )
                            metrics.update(signal_metrics)
                            metrics["wm/ratio_real"] = 1.0 - r_wm
                            metrics["wm/ratio_wm"] = r_wm
                            metrics["wm/next_imag_horizon"] = int(
                                self._imagined_horizon
                            )

                            print(
                                "[WM CONF] "
                                f"pred_valid_ratio={pred_valid_ratio:.4f}, "
                                f"batch_conf={batch_confidence:.4f}, "
                                f"conf_ema={confidence_ema:.4f}, "
                                f"conf_target={confidence_target:.4f}, "
                                f"ratio_target={combined_target:.4f}, "
                                f"alpha_imag={r_wm:.4f}, "
                                f"next_horizon={self._imagined_horizon}"
                            )
                    else:
                        wm_roll_batch = None
                        r_wm = 0.0
                        self._r_wm = 0.0
                        metrics["wm/ratio_real"] = 1.0
                        metrics["wm/ratio_wm"] = 0.0
                        metrics["wm/next_imag_horizon"] = current_imagined_horizon
                        print(
                            "[WM] wm_roll_batch invalid after generation/filter -> fallback to pure real."
                        )
                else:
                    wm_roll_batch = None
                    r_wm = 0.0
                    self._r_wm = 0.0
                    if train_mode == "MFRL":
                        r_wm = 0.0
                        wm_roll_batch = None
                        print("[Mode MFRL] forcing r_wm=0.0 (pure real env data)")
                    metrics["wm/ratio_real"] = 1.0
                    metrics["wm/ratio_wm"] = 0.0
                    metrics["wm/next_imag_horizon"] = current_imagined_horizon

                # =====================================================
                # construct mixed batch
                # =====================================================
                total_needed = batch_size * n_samples
                real_pool_for_backfill: Optional[DataProto] = None

                if r_wm <= 0.0 or wm_roll_batch is None:
                    mixed_batch = clone_dataproto_for_replay(valid_batch)
                    mixed_batch = _attach_real_reward_anchors(
                        mixed_batch, is_wm=(policy_mode == "MBRL")
                    )
                    real_pool_for_backfill = (
                        clone_dataproto_for_replay(mixed_batch)
                        if policy_mode != "MBRL"
                        else None
                    )
                    mixed_samples = None
                else:
                    debug_dummy(valid_batch, "valid_batch (before)")
                    debug_dummy(wm_roll_batch, "wm_roll_batch (before)")

                    if policy_mode == "MBRL":
                        real_pool = DataProto.empty_like(wm_roll_batch)
                        wm_pool = DataProtoFilter.filter_rollout_samples(wm_roll_batch)
                        print(
                            "[Filter] Policy batch is WM-only; skipping real_pool add."
                        )
                        print(f"  - wm_pool: {len(wm_roll_batch)} -> {len(wm_pool)}")
                        metrics["filter/real_pool_filtered"] = 0
                        metrics["filter/wm_pool_filtered"] = len(wm_roll_batch) - len(
                            wm_pool
                        )
                    else:
                        real_pool = DataProtoFilter.filter_rollout_samples(valid_batch)
                        wm_pool = DataProtoFilter.filter_rollout_samples(wm_roll_batch)

                        print(f"[Filter] After filter_rollout_samples:")
                        print(f"  - real_pool: {len(valid_batch)} -> {len(real_pool)}")
                        print(f"  - wm_pool: {len(wm_roll_batch)} -> {len(wm_pool)}")

                        metrics["filter/real_pool_filtered"] = len(valid_batch) - len(
                            real_pool
                        )
                        metrics["filter/wm_pool_filtered"] = len(wm_roll_batch) - len(
                            wm_pool
                        )

                    real_pool = _attach_real_reward_anchors(real_pool, is_wm=False)
                    wm_pool = _attach_real_reward_anchors(wm_pool, is_wm=True)
                    wm_pool = _apply_anchor_reward_lookup(
                        wm_pool, real_anchor_reward_lookup, wm_only=True
                    )
                    raw_wm_pool_len = len(wm_pool)
                    if (
                        train_mode == "MERL"
                        and require_wm_anchor_reward
                        and _is_valid_dataproto(wm_pool)
                    ):
                        wm_actor_admission_mask, wm_actor_admission_stats = (
                            _wm_actor_admission_mask(wm_pool)
                        )
                        for stat_key, stat_value in wm_actor_admission_stats.items():
                            metrics[f"wm/actor_pool_{stat_key}"] = float(stat_value)
                        admitted_wm_count = int(
                            wm_actor_admission_mask.detach().cpu().sum().item()
                        )
                        metrics["wm/actor_pool_wm_sample_count_before_anchor_filter"] = float(
                            raw_wm_pool_len
                        )
                        metrics["wm/actor_pool_wm_anchor_sample_count"] = float(
                            admitted_wm_count
                        )
                        metrics["wm/actor_pool_wm_anchor_coverage"] = (
                            float(admitted_wm_count) / float(raw_wm_pool_len)
                            if raw_wm_pool_len > 0
                            else 0.0
                        )
                        metrics["wm/actor_pool_wm_unanchored_filtered"] = float(
                            max(0, raw_wm_pool_len - admitted_wm_count)
                        )
                        wm_pool = _filter_dataproto_by_sample_mask(
                            wm_pool,
                            wm_actor_admission_mask,
                            context="WM actor anchor filter",
                        )
                    else:
                        metrics["wm/actor_pool_wm_sample_count_before_anchor_filter"] = float(
                            raw_wm_pool_len
                        )
                        metrics["wm/actor_pool_wm_anchor_sample_count"] = float(
                            raw_wm_pool_len
                        )
                        metrics["wm/actor_pool_wm_anchor_coverage"] = (
                            1.0 if raw_wm_pool_len > 0 else 0.0
                        )
                        metrics["wm/actor_pool_wm_unanchored_filtered"] = 0.0

                    real_pool = clone_dataproto_for_replay(real_pool)
                    wm_pool = clone_dataproto_for_replay(wm_pool)
                    real_pool_for_backfill = real_pool
                    real_pool, wm_pool = align_keys_between_pools(real_pool, wm_pool)

                    if wm_confidence_stats is not None:
                        expected_wm_stats = int(
                            wm_confidence_stats["sample_weight"].numel()
                        )
                        if expected_wm_stats != len(wm_pool):
                            print(
                                "[WM CONF] confidence stats length "
                                f"{expected_wm_stats} != wm_pool length {len(wm_pool)}; "
                                "recomputing after rollout filtering.",
                                flush=True,
                            )
                            wm_confidence_stats = (
                                _compute_wm_rollout_confidence(wm_pool)
                                if _is_valid_dataproto(wm_pool)
                                else None
                            )

                    real_sample_weights = torch.ones(
                        (len(real_pool),), dtype=torch.float32
                    )
                    real_priorities = torch.ones(len(real_pool), dtype=torch.float32)
                    if wm_confidence_stats is not None:
                        wm_sample_weights = wm_confidence_stats["sample_weight"].float()
                        wm_priorities = wm_confidence_stats["priority"].float()
                        metrics["wm/sample_weight_mean"] = float(
                            wm_sample_weights.mean().item()
                        )
                    else:
                        wm_sample_weights = torch.ones(
                            (len(wm_pool),), dtype=torch.float32
                        )
                        wm_priorities = torch.ones(len(wm_pool), dtype=torch.float32)

                    _add_pool_samples(
                        self.real_prioritized_pool,
                        real_pool,
                        is_wm=False,
                        sample_weights=real_sample_weights,
                        priorities=real_priorities,
                    )
                    _add_pool_samples(
                        self.wm_prioritized_pool,
                        wm_pool,
                        is_wm=True,
                        sample_weights=wm_sample_weights,
                        priorities=wm_priorities,
                    )
                    merl_group_aware_mix = bool(
                        train_mode == "MERL"
                        and policy_mode != "MBRL"
                        and require_wm_anchor_reward
                        and wm_real_anchor_reward
                    )
                    if merl_group_aware_mix:
                        # MERL actor updates use current real groups as the spine:
                        # anchored WM observations may replace rows inside those
                        # groups, but replay never rebuilds a pure-real step.
                        target_wm_count, prio_metrics = _compute_target_wm_count(
                            total_needed, r_wm
                        )
                        group_aware_real_pool = clone_dataproto_for_replay(valid_batch)
                        group_aware_real_pool = _attach_real_reward_anchors(
                            group_aware_real_pool, is_wm=False
                        )
                        group_aware_real_pool, wm_pool = align_keys_between_pools(
                            group_aware_real_pool, wm_pool
                        )
                        real_pool_for_backfill = group_aware_real_pool
                        if wm_ratio_rounding == "carry":
                            self._wm_ratio_carry = float(
                                prio_metrics.get("wm/ratio_carry_out", 0.0)
                            )
                            metrics["wm/ratio_carry"] = float(self._wm_ratio_carry)
                        metrics.update(prio_metrics)
                        metrics["wm/real_avail"] = float(len(group_aware_real_pool))
                        metrics["wm/wm_avail"] = float(len(wm_pool))
                        metrics["wm/is_weights_mean"] = 1.0
                        if target_wm_count <= 0:
                            mixed_batch = clone_dataproto_for_replay(valid_batch)
                            mixed_batch = _attach_real_reward_anchors(
                                mixed_batch, is_wm=False
                            )
                            mixed_batch = _set_sample_source_fields(
                                mixed_batch, is_wm=False, is_weight=1.0
                            )
                            mixed_samples = None
                            metrics["wm/group_aware_enabled"] = 1.0
                            metrics["wm/group_aware_target_zero_kept_current_real"] = (
                                1.0
                            )
                            metrics["wm/num_wm_sample"] = 0
                            metrics["wm/num_real_sample"] = int(len(mixed_batch))
                            metrics["wm/mixed_size"] = int(len(mixed_batch))
                            metrics["wm/ratio_wm"] = 0.0
                            metrics["wm/ratio_real"] = 1.0
                        else:
                            mixed_batch, mixed_samples, group_mix_metrics = (
                                _build_merl_group_aware_actor_mix(
                                    current_real_batch=valid_batch,
                                    real_pool=group_aware_real_pool,
                                    wm_pool=wm_pool,
                                    wm_sample_weights=wm_sample_weights,
                                    wm_priorities=wm_priorities,
                                    total_needed=total_needed,
                                    target_wm_count=target_wm_count,
                                )
                            )
                            metrics.update(group_mix_metrics)
                            if not _is_valid_dataproto(mixed_batch):
                                mixed_batch = clone_dataproto_for_replay(valid_batch)
                                mixed_batch = _attach_real_reward_anchors(
                                    mixed_batch, is_wm=False
                                )
                                mixed_batch = _set_sample_source_fields(
                                    mixed_batch, is_wm=False, is_weight=1.0
                                )
                                mixed_samples = None
                                metrics["wm/group_aware_fallback_real_batch"] = 1.0
                            actual_wm = 0
                            if mixed_samples is not None:
                                actual_wm = int(
                                    sum(
                                        len(s["dataproto"])
                                        for s in mixed_samples
                                        if bool(s.get("is_wm", False))
                                    )
                                )
                            metrics["wm/num_wm_sample"] = int(actual_wm)
                            metrics["wm/num_real_sample"] = int(
                                max(0, len(mixed_batch) - actual_wm)
                            )
                            metrics["wm/mixed_size"] = int(len(mixed_batch))
                            actual_total = max(len(mixed_batch), 1)
                            metrics["wm/ratio_wm"] = float(actual_wm / actual_total)
                            metrics["wm/ratio_real"] = float(
                                1.0 - metrics["wm/ratio_wm"]
                            )
                    else:
                        mixed_samples, prio_metrics = prioritized_mixed_sample(
                            prioritized_real_pool=self.real_prioritized_pool,
                            prioritized_wm_pool=self.wm_prioritized_pool,
                            total_needed=total_needed,
                            r_wm=r_wm,
                            alpha=0.0 if train_mode == "ONLINE_MBRL" else 0.6,
                            beta=0.0 if train_mode == "ONLINE_MBRL" else 0.4,
                            ratio_rounding=wm_ratio_rounding,
                            ratio_carry=(
                                float(self._wm_ratio_carry)
                                if wm_ratio_rounding == "carry"
                                else 0.0
                            ),
                            allow_replace=bool(
                                train_mode == "MERL" and require_wm_anchor_reward
                            ),
                            strict_preferred_source=policy_mode in ("MBRL", "MFRL"),
                        )
                        if wm_ratio_rounding == "carry":
                            self._wm_ratio_carry = float(
                                prio_metrics.get("wm/ratio_carry_out", 0.0)
                            )
                            metrics["wm/ratio_carry"] = float(self._wm_ratio_carry)
                        metrics.update(prio_metrics)
                        aligned_samples = align_dataproto_list_for_concat(
                            [s["dataproto"] for s in mixed_samples]
                        )
                        mixed_batch = DataProto.concat(aligned_samples)

                        actual_total = max(len(mixed_batch), 1)
                        actual_wm = int(metrics.get("wm/num_wm_sample", 0))
                        metrics["wm/ratio_wm"] = float(actual_wm / actual_total)
                        metrics["wm/ratio_real"] = float(
                            1.0 - metrics["wm/ratio_wm"]
                        )

                _ensure_uid(mixed_batch)

                len_mixed_batch_before = len(mixed_batch)

                print(f"### Debug DataProto Consistency ###")
                print(f"Expected Batch Size (len): {len(mixed_batch)}")
                for k, v in mixed_batch.batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_dim_size = v.shape[0]
                        if batch_dim_size != len(mixed_batch):
                            print(
                                f"[CHECK] MISMATCH Found: Key '{k}' has shape {v.shape}, expected batch dim {len(mixed_batch)}"
                            )
                        else:
                            print(f"[CHECK] OK: Key '{k}' has shape {v.shape}")
                print(f"### End DataProto Consistency Debug ###")

                is_weight_list = []
                is_wm_list = []
                for s in (
                    mixed_samples
                    if r_wm > 0.0
                    and wm_roll_batch is not None
                    and mixed_samples is not None
                    else [{"dataproto": mixed_batch, "is_weight": 1.0, "is_wm": False}]
                ):
                    n = len(s["dataproto"])
                    is_weight_list.extend([s["is_weight"]] * n)
                    is_wm_list.extend([float(bool(s.get("is_wm", False)))] * n)

                if len(is_weight_list) != len(mixed_batch):
                    print(
                        "[is_weight] WARNING: length mismatch "
                        f"{len(is_weight_list)} vs batch {len(mixed_batch)}; "
                        "truncating/padding with weight=1.0.",
                        flush=True,
                    )
                    is_weight_list = list(is_weight_list[: len(mixed_batch)])
                    is_weight_list.extend(
                        [1.0] * max(0, len(mixed_batch) - len(is_weight_list))
                    )
                if len(is_wm_list) != len(mixed_batch):
                    print(
                        "[is_wm] WARNING: length mismatch "
                        f"{len(is_wm_list)} vs batch {len(mixed_batch)}; "
                        "truncating/padding as real samples.",
                        flush=True,
                    )
                    is_wm_list = list(is_wm_list[: len(mixed_batch)])
                    is_wm_list.extend(
                        [0.0] * max(0, len(mixed_batch) - len(is_wm_list))
                    )

                mixed_batch.batch["is_weight"] = torch.tensor(
                    is_weight_list,
                    dtype=torch.float32,
                    device=mixed_batch.batch["input_ids"].device,
                )
                mixed_batch.batch["is_wm"] = torch.tensor(
                    is_wm_list,
                    dtype=torch.float32,
                    device=mixed_batch.batch["input_ids"].device,
                )
                mixed_real_anchor_lookup = _build_anchor_reward_lookup(
                    mixed_batch, real_only=True
                )
                for key, value in mixed_real_anchor_lookup.items():
                    real_anchor_reward_lookup[key] = max(
                        float(real_anchor_reward_lookup.get(key, 0.0)),
                        float(value),
                    )
                mixed_batch = _apply_anchor_reward_lookup(
                    mixed_batch, real_anchor_reward_lookup, wm_only=True
                )
                mixed_batch, shared_filter_stats = (
                    _keep_only_wm_with_current_real_anchor(
                        mixed_batch,
                        real_backfill_pool=real_pool_for_backfill,
                        context="pre_ppo",
                    )
                )
                for stat_key, stat_value in shared_filter_stats.items():
                    metrics[f"wm/pre_ppo_shared_anchor_{stat_key}"] = float(
                        stat_value
                    )
                pre_filter_wm_token_stats = _valid_response_token_stats(
                    mixed_batch, wm_only=True
                )
                metrics["wm/pre_filter_valid_imag_sample_count"] = float(
                    pre_filter_wm_token_stats["valid_sample_count"]
                )
                metrics["wm/pre_filter_valid_imag_token_count"] = float(
                    pre_filter_wm_token_stats["token_count"]
                )
                if (
                    strict_mode_assert
                    and train_mode == "MERL"
                    and pre_filter_wm_token_stats["sample_count"] > 0
                    and pre_filter_wm_token_stats["token_count"] <= 0
                ):
                    raise RuntimeError(
                        "[WM CONTRACT] mixed batch contains imagined samples before "
                        "PPO filtering, but all have zero valid response tokens."
                    )
                print(
                    f"[is_weight] Before filter: len={len(is_weight_list)}, batch_size={len(mixed_batch)}"
                )

                print(f"### Before filter_ppo_samples ###")
                print(f"mixed_batch len: {len(mixed_batch)}")
                print(f"mixed_batch batch keys: {list(mixed_batch.batch.keys())}")
                if hasattr(mixed_batch, "non_tensor_batch"):
                    print(
                        f"mixed_batch non_tensor_batch keys: {list(mixed_batch.non_tensor_batch.keys())}"
                    )
                    for k, v in mixed_batch.non_tensor_batch.items():
                        if isinstance(v, np.ndarray):
                            print(f"  non_tensor_batch['{k}']: len={len(v)}")

                mixed_batch = DataProtoFilter.filter_ppo_samples(
                    mixed_batch,
                    require_pixel_values=False,
                    action_token_len=self.config.actor_rollout_ref.model.action_token_len,
                    require_valid_response_tokens=True,
                )

                print(f"### After filter_ppo_samples ###")
                print(f"mixed_batch len: {len(mixed_batch)}")
                if hasattr(mixed_batch, "non_tensor_batch"):
                    for k, v in mixed_batch.non_tensor_batch.items():
                        if isinstance(v, np.ndarray):
                            if len(v) != len(mixed_batch):
                                print(
                                    f"[CHECK] MISMATCH: non_tensor_batch['{k}'] len={len(v)} != batch_size={len(mixed_batch)}"
                                )
                            else:
                                print(
                                    f"[CHECK] OK: non_tensor_batch['{k}'] len={len(v)} == batch_size={len(mixed_batch)}"
                                )
                print(f"### End filter_ppo_samples Debug ###")

                if "is_weight" in mixed_batch.batch:
                    if mixed_batch.batch["is_weight"].shape[0] != len(mixed_batch):
                        print(
                            f"[Warning] is_weight dimension mismatch after filter, rebuilding..."
                        )
                        del mixed_batch.batch["is_weight"]
                        _rebuild_is_weight = True
                    else:
                        _rebuild_is_weight = False
                        print(
                            f"[is_weight] After filter: len={mixed_batch.batch['is_weight'].shape[0]}, batch_size={len(mixed_batch)}"
                        )
                else:
                    _rebuild_is_weight = True

                if _rebuild_is_weight:
                    # Keep PPO gradients valid after filtering.
                    # A zero fallback here silently nulls policy gradients while
                    # AdamW still applies weight decay.
                    is_weight_rebuild = torch.ones(
                        (len(mixed_batch),),
                        dtype=torch.float32,
                        device=mixed_batch.batch["input_ids"].device,
                    )
                    mixed_batch.batch["is_weight"] = is_weight_rebuild
                    print(
                        f"[is_weight] Rebuilt after filter: len={is_weight_rebuild.shape[0]}, batch_size={len(mixed_batch)}"
                    )

                if "is_wm" in mixed_batch.batch:
                    if mixed_batch.batch["is_wm"].shape[0] != len(mixed_batch):
                        message = (
                            "[WM CONTRACT] is_wm dimension mismatch after "
                            f"filter: {mixed_batch.batch['is_wm'].shape[0]} "
                            f"vs batch_size {len(mixed_batch)}"
                        )
                        if strict_mode_assert and train_mode == "MERL":
                            raise RuntimeError(message)
                        print(message + "; rebuilding as real-only.", flush=True)
                        mixed_batch.batch["is_wm"] = torch.zeros(
                            (len(mixed_batch),),
                            dtype=torch.float32,
                            device=mixed_batch.batch["input_ids"].device,
                        )
                else:
                    if (
                        strict_mode_assert
                        and train_mode in ("MERL", "MBRL", "ONLINE_MBRL")
                        and pre_filter_wm_token_stats["sample_count"] > 0
                    ):
                        raise RuntimeError(
                            "[WM CONTRACT] is_wm missing after PPO filtering."
                        )
                    mixed_batch.batch["is_wm"] = torch.zeros(
                        (len(mixed_batch),),
                        dtype=torch.float32,
                        device=mixed_batch.batch["input_ids"].device,
                    )

                if mixed_batch.batch["is_weight"].shape[0] != len(mixed_batch):
                    print(
                        "[is_weight] WARNING: final shape mismatch "
                        f"{mixed_batch.batch['is_weight'].shape[0]} vs {len(mixed_batch)}; "
                        "rebuilding with weight=1.0.",
                        flush=True,
                    )
                    mixed_batch.batch["is_weight"] = torch.ones(
                        (len(mixed_batch),),
                        dtype=torch.float32,
                        device=mixed_batch.batch["input_ids"].device,
                    )

                mixed_batch, final_shared_filter_stats = (
                    _keep_only_wm_with_current_real_anchor(
                        mixed_batch,
                        real_backfill_pool=real_pool_for_backfill,
                        context="post_ppo",
                    )
                )
                for stat_key, stat_value in final_shared_filter_stats.items():
                    metrics[f"wm/post_ppo_shared_anchor_{stat_key}"] = float(
                        stat_value
                    )

                anchor_weight_stats = _enforce_wm_anchor_weight_contract(mixed_batch)
                for stat_key, stat_value in anchor_weight_stats.items():
                    metrics[f"wm/anchor_weight_{stat_key}"] = float(stat_value)

                actual_wm_after_filter = int(mixed_batch.batch["is_wm"].sum().item())
                target_wm_for_ratio = float(
                    metrics.get("wm/target_wm_sample_count", 0.0) or 0.0
                )
                realized_vs_target = (
                    float(actual_wm_after_filter) / max(target_wm_for_ratio, 1e-6)
                    if target_wm_for_ratio > 0.0
                    else 1.0
                )
                metrics["wm/ratio_realized_vs_target"] = float(realized_vs_target)
                metrics["wm/ratio_realized_vs_target_clipped"] = float(
                    np.clip(realized_vs_target, 0.0, 1.0)
                )
                metrics["wm/actual_wm_after_filter"] = float(actual_wm_after_filter)
                post_filter_wm_token_stats = _valid_response_token_stats(
                    mixed_batch, wm_only=True
                )
                metrics["wm/valid_imag_sample_count"] = float(
                    post_filter_wm_token_stats["valid_sample_count"]
                )
                metrics["wm/valid_imag_token_count"] = float(
                    post_filter_wm_token_stats["token_count"]
                )
                metrics["wm/valid_imag_token_mean"] = float(
                    post_filter_wm_token_stats["mean_tokens"]
                )
                if (
                    strict_mode_assert
                    and train_mode in ("MERL", "MBRL", "ONLINE_MBRL")
                    and actual_wm_after_filter > 0
                    and post_filter_wm_token_stats["token_count"] <= 0
                ):
                    raise RuntimeError(
                        "[WM CONTRACT] mixed batch contains imagined samples but "
                        "no valid imagined response tokens after PPO filtering."
                    )
                metrics["wm/num_wm_sample"] = actual_wm_after_filter
                metrics["wm/num_real_sample"] = int(
                    len(mixed_batch) - actual_wm_after_filter
                )
                if len(mixed_batch) > 0:
                    metrics["wm/ratio_wm"] = float(
                        actual_wm_after_filter / len(mixed_batch)
                    )
                    metrics["wm/ratio_real"] = float(1.0 - metrics["wm/ratio_wm"])

                print(
                    f"[Filter] After filter_ppo_samples: {len_mixed_batch_before} -> {len(mixed_batch)}"
                )
                metrics["filter/len_mixed_batch_filtered"] = (
                    len_mixed_batch_before - len(mixed_batch)
                )

                if len(mixed_batch) == 0:
                    print(
                        "[Warning] mixed_batch is empty after filtering, skip this update"
                    )
                    continue

                if len(mixed_batch) == 0:
                    print("[Warning] mixed_batch empty before actor update, skip")
                    continue

                uid_group_stats = _apply_grpo_uid_grouping(mixed_batch)
                metrics["wm/grpo_uid_anchor_mode"] = float(
                    uid_group_stats.get("mode_anchor", 0.0)
                )
                metrics["wm/grpo_uid_source_mode"] = float(
                    uid_group_stats.get("mode_source", 0.0)
                )
                metrics["wm/grpo_uid_none_mode"] = float(
                    uid_group_stats.get("mode_none", 0.0)
                )
                metrics["wm/grpo_uid_grouping_applied"] = float(
                    uid_group_stats.get("applied", 0.0)
                )
                metrics["wm/grpo_uid_cross_source_shared_count"] = float(
                    uid_group_stats.get("cross_source_shared_count", 0.0)
                )
                metrics["wm/grpo_uid_real_group_count"] = float(
                    uid_group_stats.get("real_group_count", 0.0)
                )
                metrics["wm/grpo_uid_imag_group_count"] = float(
                    uid_group_stats.get("imag_group_count", 0.0)
                )
                metrics["wm/grpo_uid_anchor_group_count"] = float(
                    uid_group_stats.get("anchor_group_count", 0.0)
                )
                metrics["wm/grpo_uid_unanchored_wm_count"] = float(
                    uid_group_stats.get("unanchored_wm_count", 0.0)
                )
                metrics["wm/grpo_uid_singleton_group_count"] = float(
                    uid_group_stats.get("grpo_uid_singleton_group_count", 0.0)
                )
                metrics["wm/effective_grpo_group_count"] = float(
                    uid_group_stats.get("effective_grpo_group_count", 0.0)
                )
                metrics["wm/grpo_uid_mean_group_size"] = float(
                    uid_group_stats.get("grpo_uid_mean_group_size", 0.0)
                )
                metrics["wm/grpo_uid_min_group_size"] = float(
                    uid_group_stats.get("grpo_uid_min_group_size", 0.0)
                )
                metrics["wm/grpo_uid_max_group_size"] = float(
                    uid_group_stats.get("grpo_uid_max_group_size", 0.0)
                )

                metrics["wm/merl_reward_guard_enabled"] = 1.0 if (
                    train_mode == "MERL" and merl_imagined_reward_hard_constraint
                ) else 0.0
                metrics["wm/use_wm_reward_proxy"] = (
                    1.0 if use_wm_reward_proxy else 0.0
                )
                metrics["wm/real_anchor_reward_enabled"] = (
                    1.0 if wm_real_anchor_reward else 0.0
                )
                metrics["wm/require_anchor_reward"] = (
                    1.0 if require_wm_anchor_reward else 0.0
                )
                metrics["wm/imag_adv_clip_abs"] = float(imag_advantage_abs_clip)
                metrics.update(
                    {
                        k: float(v)
                        for k, v in getattr(
                            self, "_wm_ratio_health_metrics", {}
                        ).items()
                    }
                )

                if (
                    train_mode == "MERL"
                    and merl_imagined_reward_hard_constraint
                    and not wm_real_anchor_reward
                    and "is_weight" in mixed_batch.batch
                    and "is_wm" in mixed_batch.batch
                ):
                    imag_sample_mask = mixed_batch.batch["is_wm"].view(-1) > 0.5
                    if imag_sample_mask.any():
                        imag_weight_before = mixed_batch.batch["is_weight"][
                            imag_sample_mask
                        ].float()
                        constrained_imag_weight = torch.minimum(
                            imag_weight_before * merl_imagined_reward_confidence,
                            torch.full_like(
                                imag_weight_before,
                                merl_imagined_reward_weight_cap,
                            ),
                        )
                        # Temporary MERL-only safeguard: keep imagined observations
                        # in the batch, but do not let imagined reward proxy dominate
                        # actor updates. Remove once reward routing is made principled.
                        mixed_batch.batch["is_weight"] = mixed_batch.batch[
                            "is_weight"
                        ].clone()
                        mixed_batch.batch["is_weight"][imag_sample_mask] = (
                            constrained_imag_weight.to(
                                dtype=mixed_batch.batch["is_weight"].dtype
                            )
                        )
                        metrics["wm/merl_imagined_reward_confidence"] = float(
                            merl_imagined_reward_confidence
                        )
                        metrics["wm/merl_imagined_weight_cap"] = float(
                            merl_imagined_reward_weight_cap
                        )
                        metrics["wm/merl_imagined_weight_mean_before"] = float(
                            imag_weight_before.mean().item()
                        )
                        metrics["wm/merl_imagined_weight_mean_after"] = float(
                            constrained_imag_weight.mean().item()
                        )

                print(
                    f"rollout batch size: {len(mixed_batch)} (real_frac={1.0 - r_wm:.3f}, wm_frac={r_wm:.3f})"
                )
                metrics["wm/ratio_real_target"] = 1.0 - float(r_wm)
                metrics["wm/ratio_wm_target"] = float(r_wm)
                metrics.setdefault("wm/ratio_real", 1.0 - float(r_wm))
                metrics.setdefault("wm/ratio_wm", float(r_wm))
                mixed_batch.meta_info["alpha_imag"] = float(r_wm)
                mixed_batch.meta_info["imagined_horizon"] = int(
                    current_imagined_horizon
                )
                if train_mode == "ONLINE_MBRL":
                    from merl.modes import online_mbrl_actor_contract
                    online_mbrl_actor_contract(mixed_batch.batch)
                    metrics["wm/trust_enabled"] = 0.0
                    metrics["wm/online_real_training_samples"] = float(len(calibration_real_batch))
                actor_contract_stats = inject_actor_token_contract(
                    mixed_batch,
                    action_token_len=self.config.actor_rollout_ref.model.action_token_len,
                    strict=bool(
                        strict_mode_assert
                        and train_mode in ("MERL", "MBRL", "ONLINE_MBRL")
                        and float(metrics.get("wm/ratio_wm", 0.0)) > 0.0
                    ),
                    require_wm_anchor_reward=require_wm_anchor_reward,
                    context=f"{train_mode}/actor_input",
                )
                for stat_key, stat_value in actor_contract_stats.items():
                    metrics[f"wm/actor_input_{stat_key}"] = stat_value
                if (
                    strict_mode_assert
                    and train_mode in ("MERL", "MBRL", "ONLINE_MBRL")
                    and float(metrics.get("wm/ratio_wm", 0.0)) > 0.0
                ):
                    if actor_contract_stats["imag_sample_count"] != float(
                        actual_wm_after_filter
                    ):
                        raise RuntimeError(
                            "[WM CONTRACT] actor token contract disagrees with "
                            "filtered is_wm samples: "
                            f"contract={actor_contract_stats['imag_sample_count']}, "
                            f"filtered={actual_wm_after_filter}."
                        )
                    if actor_contract_stats["imag_token_count"] <= 0.0:
                        raise RuntimeError(
                            "[WM CONTRACT] imagined samples reached actor input "
                            "but token-level imagined mask is empty."
                        )
                    if actor_contract_stats["imag_weight_mean"] <= 0.0:
                        metrics["wm/actor_input_zero_imag_weight_warning"] = 1.0
                        print(
                            "[WM CONTRACT] imagined samples reached actor input "
                            "but token-level imagined weights are zero. "
                            "Continuing because unanchored WM samples are intentionally "
                            "masked from actor loss.",
                            flush=True,
                        )

                # optional reference log prob
                if self.use_reference_policy:
                    with Timer(
                        name="ref", text="{name}: {seconds:.1f} seconds"
                    ) as timer:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                            mixed_batch
                        )
                        mixed_batch = mixed_batch.union(ref_log_prob)
                        metrics["timing/ref"] = timer.last

                # reward / GAE / KL / advantage
                with Timer(
                    name="reward", text="{name}: {seconds:.1f} seconds"
                ) as timer:
                    reward_tensor_dict, reward_metrics = self.reward_fn(mixed_batch)
                    mixed_batch.batch["token_level_scores"] = reward_tensor_dict["all"]
                    for k, v in reward_metrics.items():
                        metrics["train_reward/" + k] = v
                    for k, v in reward_tensor_dict.items():
                        mixed_batch.batch[k] = v
                    metrics["timing/reward_model"] = timer.last

                print("3. Calculating GAE")
                with Timer(name="adv", text="{name}: {seconds:.1f} seconds") as timer:
                    mixed_batch, kl_metrics = apply_kl_penalty(
                        mixed_batch,
                        kl_ctrl=self.kl_ctrl,
                        kl_penalty=self.config.algorithm.kl_penalty,
                        action_token_len=self.config.actor_rollout_ref.model.action_token_len,
                        action_chunks_len=self.config.actor_rollout_ref.model.action_chunks_len,
                        config=self.config,
                    )
                    metrics.update(kl_metrics)
                    mixed_batch = compute_advantage(
                        mixed_batch,
                        self.config.algorithm.gamma,
                        self.config.algorithm.lam,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        config=self.config,
                    )
                    if (
                        imag_advantage_abs_clip > 0.0
                        and "advantages" in mixed_batch.batch
                        and "is_wm" in mixed_batch.batch
                    ):
                        imag_sample_mask = mixed_batch.batch["is_wm"].view(-1) > 0.5
                        if imag_sample_mask.any():
                            imag_adv_before = mixed_batch.batch["advantages"][
                                imag_sample_mask
                            ].float()
                            clipped_imag_adv = imag_adv_before.clamp(
                                -imag_advantage_abs_clip,
                                imag_advantage_abs_clip,
                            )
                            mixed_batch.batch["advantages"] = mixed_batch.batch[
                                "advantages"
                            ].clone()
                            mixed_batch.batch["advantages"][imag_sample_mask] = (
                                clipped_imag_adv.to(
                                    dtype=mixed_batch.batch["advantages"].dtype
                                )
                            )
                            metrics["wm/imag_adv_abs_mean_before"] = float(
                                imag_adv_before.abs().mean().item()
                            )
                            metrics["wm/imag_adv_abs_mean_after"] = float(
                                clipped_imag_adv.abs().mean().item()
                            )
                    if (
                        train_mode == "MERL"
                        and merl_imagined_reward_hard_constraint
                        and "returns" in mixed_batch.batch
                        and "is_weight" in mixed_batch.batch
                        and "is_wm" in mixed_batch.batch
                    ):
                        imag_sample_mask = mixed_batch.batch["is_wm"].view(-1) > 0.5
                        if imag_sample_mask.any():
                            imag_returns_before = mixed_batch.batch["returns"][
                                imag_sample_mask
                            ].float()
                            imag_return_scale = mixed_batch.batch["is_weight"][
                                imag_sample_mask
                            ].float()
                            imag_return_scale = imag_return_scale.view(
                                -1,
                                *([1] * max(imag_returns_before.dim() - 1, 0)),
                            )
                            scaled_imag_returns = (
                                imag_returns_before * imag_return_scale
                            )
                            mixed_batch.batch["returns"] = mixed_batch.batch[
                                "returns"
                            ].clone()
                            mixed_batch.batch["returns"][imag_sample_mask] = (
                                scaled_imag_returns.to(
                                    dtype=mixed_batch.batch["returns"].dtype
                                )
                            )
                            # Keep MERL critic targets aligned with the same
                            # low-confidence imagined reward contract used by the actor.
                            metrics["wm/merl_imagined_return_weight_mean"] = float(
                                imag_return_scale.mean().item()
                            )
                            metrics["wm/merl_imagined_return_abs_mean_before"] = float(
                                imag_returns_before.abs().mean().item()
                            )
                            metrics["wm/merl_imagined_return_abs_mean_after"] = float(
                                scaled_imag_returns.abs().mean().item()
                            )
                    metrics["timing/adv"] = timer.last

                # update actor
                print("4. Updating Actor Model")
                if self.config.trainer.critic_warmup <= global_steps:
                    with Timer(
                        name="update_actor", text="{name}: {seconds:.1f} seconds"
                    ) as timer:
                        mixed_batch.meta_info["is_filtered"] = True
                        mixed_batch.meta_info["train_mode"] = False
                        actor_lr_scale_for_update = float(
                            np.clip(
                                getattr(self, "_actor_lr_health_scale", 1.0),
                                0.10,
                                1.0,
                            )
                        )
                        mixed_batch.meta_info["actor_lr_scale"] = (
                            actor_lr_scale_for_update
                        )
                        actor_output = self.actor_rollout_wg.update_actor(mixed_batch)
                        entropy_output = self.actor_rollout_wg.compute_entropy(
                            data=mixed_batch
                        )
                        metrics["timing/update_actor"] = timer.last
                        print(
                            "actor_output.meta_info['metrics']:",
                            actor_output.meta_info["metrics"],
                        )
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        print(
                            "actor_output_metrics after reduce_metrics:",
                            actor_output_metrics,
                        )
                        entropy_output_metrics = reduce_metrics(
                            entropy_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)
                        metrics.update(entropy_output_metrics)
                        metrics["actor/lr_health_scale_applied"] = float(
                            actor_lr_scale_for_update
                        )
                        if "actor/lr_effective(1e-4)" not in metrics:
                            try:
                                base_actor_lr = float(
                                    self.config.actor_rollout_ref.actor.optim.get(
                                        "lr", 0.0
                                    )
                                )
                            except Exception:
                                base_actor_lr = 0.0
                            metrics.setdefault("actor/lr(1e-4)", base_actor_lr * 1e4)
                            metrics["actor/lr_effective(1e-4)"] = (
                                base_actor_lr * actor_lr_scale_for_update * 1e4
                            )
                        hard_skip_ratio = float(
                            np.clip(
                                metrics.get("actor/ppo_kl_hard_skip_ratio", 0.0),
                                0.0,
                                1.0,
                            )
                        )
                        hard_kl_max = float(
                            max(0.0, metrics.get("actor/ppo_kl_hard_max", 0.0))
                        )
                        self._last_actor_ppo_kl_hard_skip_ratio = hard_skip_ratio
                        self._last_actor_ppo_kl_hard_max = hard_kl_max
                        target_wm_count_metric = float(
                            metrics.get("wm/target_wm_sample_count", 0.0) or 0.0
                        )
                        realized_ratio_metric = float(
                            metrics.get(
                                "wm/ratio_realized_vs_target",
                                1.0 if target_wm_count_metric <= 0.0 else 0.0,
                            )
                        )
                        pre_anchor_cov_metric = float(
                            metrics.get(
                                "wm/pre_ppo_shared_anchor_shared_anchor_coverage",
                                metrics.get("wm/actor_pool_wm_anchor_coverage", 1.0),
                            )
                        )
                        if (
                            float(metrics.get("wm/ratio_wm", 0.0)) <= 0.0
                            and float(
                                metrics.get(
                                    "wm/pre_ppo_shared_anchor_wm_candidate_count",
                                    0.0,
                                )
                            )
                            <= 0.0
                            and target_wm_count_metric <= 0.0
                        ):
                            pre_anchor_cov_metric = 1.0
                            realized_ratio_metric = 1.0
                        feedback_anchor_coverage = min(
                            pre_anchor_cov_metric, realized_ratio_metric
                        )
                        self._last_wm_actor_anchor_coverage = float(
                            np.clip(feedback_anchor_coverage, 0.0, 1.0)
                        )
                        prev_lr_health_scale = float(
                            getattr(self, "_actor_lr_health_scale", 1.0)
                        )
                        if hard_skip_ratio >= 1.0:
                            self._actor_lr_health_scale = max(
                                0.10,
                                prev_lr_health_scale * 0.35,
                            )
                        elif hard_skip_ratio >= 0.5:
                            self._actor_lr_health_scale = max(
                                0.15,
                                min(prev_lr_health_scale * 0.60, 0.50),
                            )
                        elif hard_skip_ratio > 0.0:
                            self._actor_lr_health_scale = max(
                                0.35,
                                min(prev_lr_health_scale, 1.0 - 0.75 * hard_skip_ratio),
                            )
                        else:
                            self._actor_lr_health_scale = min(
                                1.0,
                                prev_lr_health_scale + 0.10,
                            )
                        metrics["actor/lr_health_scale_next"] = float(
                            self._actor_lr_health_scale
                        )
                        metrics["wm/ratio_feedback_anchor_coverage"] = float(
                            self._last_wm_actor_anchor_coverage
                        )
                        metrics["wm/ratio_feedback_pre_anchor_coverage"] = float(
                            np.clip(pre_anchor_cov_metric, 0.0, 1.0)
                        )
                        metrics["wm/ratio_feedback_realized_vs_target"] = float(
                            np.clip(realized_ratio_metric, 0.0, 1.0)
                        )
                        metrics["wm/ratio_feedback_hard_skip_ratio"] = float(
                            hard_skip_ratio
                        )
                        if (
                            strict_mode_assert
                            and train_mode in ("MERL", "MBRL", "ONLINE_MBRL")
                            and float(metrics.get("wm/ratio_wm", 0.0)) > 0.0
                        ):
                            imag_tokens = float(
                                metrics.get("actor/imag_token_count", 0.0)
                            )
                            imag_weight = float(
                                metrics.get("actor/imag_weight_mean", 0.0)
                            )
                            valid_imag_tokens = float(
                                metrics.get("wm/valid_imag_token_count", 0.0)
                            )
                            actor_input_imag_tokens = float(
                                metrics.get("wm/actor_input_imag_token_count", 0.0)
                            )
                            if valid_imag_tokens <= 0.0:
                                raise RuntimeError(
                                    "[WM CONTRACT] wm/ratio_wm > 0 but no valid "
                                    "imagined PPO tokens reached the actor batch."
                                )
                            if actor_input_imag_tokens <= 0.0 or imag_tokens <= 0.0:
                                raise RuntimeError(
                                    "[WM CONTRACT] imagined samples reached the "
                                    "mixed batch but actor observed zero imagined "
                                    f"observation tokens: imag_tokens={imag_tokens}, "
                                    f"actor_input_imag_tokens={actor_input_imag_tokens}."
                                )
                            if imag_weight <= 0.0:
                                metrics["wm/actor_zero_imag_weight_warning"] = 1.0
                                print(
                                    "[WM CONTRACT] actor observed imagined "
                                    "observation tokens with zero imagined loss "
                                    "weight. Continuing because unanchored WM "
                                    "observations are allowed while WM proxy reward "
                                    "is disabled.",
                                    flush=True,
                                )

                # WM update (only if allowed by mode)
                print("///////// 5. Updating World Model /////////")
                success_updated_wm = False
                valid_wm_update = False
                wm_result = None

                should_run_wm_update = (
                    update_wm_effective and use_wm_now and wm_inner_steps > 0
                )
                if weak_update_active and (not weak_update_calibration_step):
                    should_run_wm_update = False
                    metrics["wm/update/skipped_sparse_weak"] = 1.0
                elif (
                    weak_update_active
                    and weak_update_calibration_step
                    and (not _is_valid_dataproto(calibration_real_batch))
                ):
                    should_run_wm_update = False
                    metrics["wm/update/skipped_sparse_weak_no_real"] = 1.0

                if should_run_wm_update:
                    if self.wm_trainer is None:
                        metrics["wm/update/skipped_no_wm_trainer"] = 1.0
                        print("[WM] wm_trainer is None, skip WM update.")
                    else:
                        metrics["wm/update/steps_requested"] = float(wm_inner_steps)
                        try:
                            with Timer(
                                name="update_wm", text="{name}: {seconds:.1f} seconds"
                            ) as timer_wm_up:
                                if ray.get(self.wm_trainer.is_updating.remote()):
                                    print("WM trainer busy, skip this update")
                                    success_updated_wm = False
                                else:
                                    wm_result = ray.get(
                                        self.wm_trainer.update_world_model.remote(
                                            global_steps,
                                            steps=wm_inner_steps,
                                        )
                                    )
                                    success_updated_wm = True
                            metrics["timing/update_wm"] = timer_wm_up.last
                        except Exception as e:
                            print(f"[WM] update_world_model failed: {e}")

                        if isinstance(wm_result, dict):
                            metrics["wm/update/steps_done"] = float(
                                wm_result.get("steps_done", 0)
                            )
                            metrics["wm/update/skipped"] = (
                                1.0 if bool(wm_result.get("skipped", False)) else 0.0
                            )
                            if bool(wm_result.get("skipped_no_data", False)):
                                metrics["wm/update/skipped_no_data"] = 1.0
                            if wm_result.get("skip_reason"):
                                print(
                                    f"[WM] update skipped: {wm_result.get('skip_reason')}",
                                    flush=True,
                                )
                            data_meta = wm_result.get("data_meta", None)
                            if isinstance(data_meta, dict):
                                metrics["wm/update/data_num_shards"] = float(
                                    data_meta.get("num_shards", 0)
                                )
                                metrics["wm/update/data_fallback_used"] = (
                                    1.0
                                    if bool(data_meta.get("fallback_used", False))
                                    else 0.0
                                )

                        valid_wm_update = (
                            success_updated_wm
                            and isinstance(wm_result, dict)
                            and wm_result.get("steps_done", 0) > 0
                        )

                        # total WM loss: keep for logging only
                        if valid_wm_update and "wm_loss" in wm_result:
                            metrics["wm/loss"] = float(wm_result["wm_loss"])
                            print(f"wm_loss(total): {wm_result['wm_loss']}")
                            new_wm_loss = float(wm_result["wm_loss"])
                            if (not hasattr(self, "_wm_loss_ema")) or (
                                self._wm_loss_ema is None
                            ):
                                self._wm_loss_ema = new_wm_loss
                            else:
                                self._wm_loss_ema = (
                                    wm_loss_smooth_alpha * self._wm_loss_ema
                                    + (1.0 - wm_loss_smooth_alpha) * new_wm_loss
                                )
                            metrics["wm/loss_ema"] = float(self._wm_loss_ema)

                        # ratio signal: dedicated proxy for scheduling (loss_noise)
                        if valid_wm_update and "wm_ratio_signal" in wm_result:
                            metrics["wm/ratio_signal"] = float(
                                wm_result["wm_ratio_signal"]
                            )
                            print(
                                f"wm_ratio_signal(loss_noise): {wm_result['wm_ratio_signal']}"
                            )
                            new_ratio_signal = float(wm_result["wm_ratio_signal"])
                            prev_ratio_signal_ema = getattr(
                                self, "_wm_ratio_signal_ema", None
                            )
                            if prev_ratio_signal_ema is not None:
                                self._last_wm_ratio_signal_ema = float(
                                    prev_ratio_signal_ema
                                )
                            if (not hasattr(self, "_wm_ratio_signal_ema")) or (
                                self._wm_ratio_signal_ema is None
                            ):
                                self._wm_ratio_signal_ema = new_ratio_signal
                            else:
                                self._wm_ratio_signal_ema = (
                                    wm_ratio_signal_smooth_alpha
                                    * self._wm_ratio_signal_ema
                                    + (1.0 - wm_ratio_signal_smooth_alpha)
                                    * new_ratio_signal
                                )
                            metrics["wm/ratio_signal_ema"] = float(
                                self._wm_ratio_signal_ema
                            )
                elif weak_update_active and (not weak_update_calibration_step):
                    print(
                        "[WM WEAK] No sparse calibration due on this step; skip WM update.",
                        flush=True,
                    )
                elif (
                    weak_update_active
                    and weak_update_calibration_step
                    and (not _is_valid_dataproto(calibration_real_batch))
                ):
                    print(
                        "[WM WEAK] Sparse calibration step produced no valid real batch; skip WM update.",
                        flush=True,
                    )

                if train_mode == "ONLINE_MBRL" and not valid_wm_update:
                    raise RuntimeError("ONLINE_MBRL did not complete a real-data WM update; refusing to report an online baseline")

                # WM eval
                online_eval_without_update = (
                    train_mode in ("MBRL", "MERL", "ONLINE_MBRL") and use_wm_now
                )
                should_run_fixed_shared_eval = (
                    fixed_eval_enabled
                    and self.wm_trainer is not None
                    and wm_eval_interval is not None
                    and int(wm_eval_interval) > 0
                    and (global_steps + 1) % int(wm_eval_interval) == 0
                )
                should_run_online_eval = (
                    (not fixed_eval_enabled)
                    and wm_eval_interval is not None
                    and int(wm_eval_interval) > 0
                    and (global_steps + 1) % int(wm_eval_interval) == 0
                    and (valid_wm_update or online_eval_without_update)
                )
                do_eval = should_run_fixed_shared_eval or should_run_online_eval

                if do_eval:
                    if self.wm_trainer is None:
                        metrics["wm/eval/skipped_no_wm_trainer"] = 1.0
                        print("[WM Eval] wm_trainer is None, skip evaluation.")
                    else:
                        if (
                            should_run_fixed_shared_eval or online_eval_without_update
                        ) and (not valid_wm_update):
                            metrics["wm/eval/without_wm_update"] = 1.0
                        print(
                            f"[WM Eval] Start evaluation at global_steps={global_steps + 1}"
                        )
                        metrics.update(
                            _collect_wm_eval_metrics(eval_global_steps=global_steps)
                        )

                if train_mode in ("MBRL", "MERL", "ONLINE_MBRL"):
                    metrics.update(
                        self._cleanup_stale_rollout_shards(
                            keep_from_global_steps=global_steps + 1
                        )
                    )

                # validation / logging / saving
                if (
                    self.val_reward_fn is not None
                    and (global_steps + 1) % self.config.trainer.test_freq == 0
                ):
                    with Timer(
                        name="testing", text="{name}: {seconds:.1f} seconds"
                    ) as timer:
                        val_metrics: dict = self._validate(
                            global_steps=global_steps + 1
                        )
                        val_metrics = {
                            f"val/{key}": val for key, val in val_metrics.items()
                        }
                        metrics["timing/testing"] = timer.last
                        metrics.update(val_metrics)

                with Timer(
                    name="logging1", text="{name}: {seconds:.1f} seconds"
                ) as timer:
                    data_metrics = compute_data_metrics(
                        batch=mixed_batch, config=self.config
                    )

                with Timer(
                    name="logging2", text="{name}: {seconds:.1f} seconds"
                ) as timer:
                    metrics.update(data_metrics)
                    _record_merl_policy_health(metrics)
                    metrics.update(
                        {
                            k: float(v)
                            for k, v in getattr(
                                self, "_wm_horizon_health_metrics", {}
                            ).items()
                        }
                    )

                # Write explicit training progress into the metric log.
                # This makes log-based resume much more robust than pure step-only logs.
                metrics["train/epoch"] = int(epoch)
                metrics["train/global_step"] = int(global_steps)

                with Timer(
                    name="logging3", text="{name}: {seconds:.1f} seconds"
                ) as timer:
                    logger.log(data=metrics, step=global_steps)

                # checkpointing
                print("x1. Saving actor checkpoint")
                # Only publish resume snapshots when model checkpoints are physically saved.
                # This keeps "state file -> checkpoint dirs" one-to-one and safe for restart.
                checkpoint_saved = False
                actor_ckpt_this_step = None
                critic_ckpt_this_step = None
                if (
                    self.config.trainer.save_freq > 0
                    and (global_steps + 1) % self.config.trainer.save_freq == 0
                ):
                    actor_ckpt_this_step = os.path.join(
                        self.config.trainer.default_local_dir,
                        "actor",
                        f"global_step_{global_steps}",
                    )
                    actor_remote_path = None
                    self.actor_rollout_wg.save_checkpoint(
                        actor_ckpt_this_step, actor_remote_path
                    )
                    print(f"Saved ckpt step_{global_steps} into {actor_ckpt_this_step}")
                    latest_actor_ckpt_dir = actor_ckpt_this_step
                    max_saved_actor_checkpoints = max(
                        1,
                        int(
                            getattr(
                                self.config.trainer,
                                "max_saved_actor_checkpoints",
                                2,
                            )
                        ),
                    )
                    removed_actor_ckpts = self._cleanup_old_component_checkpoints(
                        component="actor",
                        keep=max_saved_actor_checkpoints,
                        protected_paths=[latest_actor_ckpt_dir],
                    )
                    if len(removed_actor_ckpts) > 0:
                        print(
                            f"[checkpoint] Removed old actor checkpoints: {removed_actor_ckpts}"
                        )
                    checkpoint_saved = True

                    if self.use_critic:
                        critic_ckpt_this_step = os.path.join(
                            self.config.trainer.default_local_dir,
                            "critic",
                            f"global_step_{global_steps}",
                        )
                        critic_remote_path = None
                        self.critic_wg.save_checkpoint(
                            critic_ckpt_this_step, critic_remote_path
                        )
                        latest_critic_ckpt_dir = critic_ckpt_this_step
                        max_saved_critic_checkpoints = max(
                            1,
                            int(
                                getattr(
                                    self.config.trainer,
                                    "max_saved_critic_checkpoints",
                                    1,
                                )
                            ),
                        )
                        removed_critic_ckpts = self._cleanup_old_component_checkpoints(
                            component="critic",
                            keep=max_saved_critic_checkpoints,
                            protected_paths=[latest_critic_ckpt_dir],
                        )
                        if len(removed_critic_ckpts) > 0:
                            print(
                                f"[checkpoint] Removed old critic checkpoints: {removed_critic_ckpts}"
                            )

                    if self.use_rm:
                        prm_local_path = os.path.join(
                            self.config.trainer.default_local_dir,
                            "prm",
                            f"global_step_{global_steps}",
                        )
                        prm_remote_path = None
                        self.rm_wg.save_checkpoint(prm_local_path, prm_remote_path)

                # save world model mapping
                should_save_wm = (
                    valid_wm_update
                    and (train_mode == "ONLINE_MBRL" or (
                        save_freq_wm_inner > 0 and (global_steps + 1) % save_freq_wm_inner == 0))
                )

                if should_save_wm:
                    should_persist_wm = train_mode == "ONLINE_MBRL" or (global_steps + 1) % save_freq_wm_outer == 0
                    world_model_ckpt_candidate = None

                    if should_persist_wm:
                        world_model_ckpt_candidate = os.path.join(
                            self.config.trainer.default_local_dir,
                            "world_model",
                            f"global_step_{global_steps + 1}",
                        )
                        print(
                            f"///////// x2-1. Saving world model checkpoint: {True} /////////"
                        )
                        try:
                            ray.get(
                                self.wm_trainer.save_world_model.remote(
                                    world_model_ckpt_candidate
                                )
                            )
                            latest_world_model_ckpt_dir = world_model_ckpt_candidate
                            max_saved_outer_checkpoints = max(
                                1,
                                int(getattr(wm_cfg, "max_saved_outer_checkpoints", 1)),
                            )
                            removed_wm_ckpts = self._cleanup_old_component_checkpoints(
                                component="world_model",
                                keep=max_saved_outer_checkpoints,
                                protected_paths=[latest_world_model_ckpt_dir],
                            )
                            if len(removed_wm_ckpts) > 0:
                                print(
                                    f"[checkpoint] Removed old world model checkpoints: {removed_wm_ckpts}"
                                )
                        except Exception as e:
                            if train_mode == "ONLINE_MBRL":
                                raise RuntimeError("ONLINE_MBRL failed to save its updated WM") from e
                            print(
                                "[WM Save] Warning: failed to persist world model checkpoint, "
                                f"continue with in-memory sync only: {e}"
                            )
                            logger.log(
                                data={
                                    "wm/save_failed": 1.0,
                                    "wm/save_failed_step": int(global_steps + 1),
                                },
                                step=global_steps,
                            )

                    print("x2-2. Syncing updated world model to each worker via memory")
                    sync_results = self.actor_rollout_wg.sync_world_model_mapping_from_trainer()
                    if train_mode == "ONLINE_MBRL":
                        from merl.modes import require_online_wm_sync
                        require_online_wm_sync(sync_results)
                    ray.get(ray.remote(lambda: None).remote())

                if checkpoint_saved:
                    if train_mode in ("MBRL", "MERL", "ONLINE_MBRL"):
                        self._save_replay_pool_state(global_step=global_steps)
                    self._write_resume_state(
                        epoch=epoch,
                        global_step=global_steps,
                        actor_ckpt_dir=latest_actor_ckpt_dir,
                        critic_ckpt_dir=latest_critic_ckpt_dir,
                        world_model_ckpt_dir=latest_world_model_ckpt_dir,
                        save_snapshot=True,
                    )

                global_steps += 1

            # end while epoch

        # end for epoch

        # final validation
        if (
            bool(getattr(self.config.trainer, "final_val_after_train", True))
            and self.val_reward_fn is not None
        ):
            val_metrics = self._validate(global_steps=global_steps)
            val_metrics = {f"val/{key}": val for key, val in val_metrics.items()}
            pprint(f"Final validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=global_steps)

        # Final refresh of latest state even if the last step was not a save_freq boundary.
        if train_mode in ("MBRL", "MERL", "ONLINE_MBRL"):
            self._save_replay_pool_state(global_step=global_steps)
        final_resume_epoch = (
            self.config.trainer.total_epochs
            if int(global_steps) > int(initial_global_steps)
            else int(start_epoch)
        )
        self._write_resume_state(
            epoch=final_resume_epoch,
            global_step=global_steps,
            actor_ckpt_dir=latest_actor_ckpt_dir,
            critic_ckpt_dir=latest_critic_ckpt_dir,
            world_model_ckpt_dir=latest_world_model_ckpt_dir,
            save_snapshot=False,
        )

    #######################################################################
    #######################################################################
    #######################################################################
    def filter_format(self, reward_tensor, batch, n_samples):
        """
        Filter responses based on accuracy and truncation criteria.

        Args:
            reward_tensor: Tensor containing accuracy scores
            batch: DataProto batch containing responses
            n_samples: Number of responses per prompt

        Returns:
            DataProto: Filtered batch
        """
        if self.config.data.filter_format:
            reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            counts = Counter(acc_tensor.tolist())
            print(
                "Format distribution:",
                " ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items())),
            )

            acc_mask = acc_tensor >= 1
        else:
            # If accuracy filtering disabled, keep all samples
            acc_mask = torch.ones(
                len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device
            )
        # Then do truncation filtering if enabled

        # Combine both masks
        combined_mask = acc_mask

        # Expand mask to cover all samples for each prompt
        final_mask = combined_mask.repeat_interleave(n_samples)

        # Apply the mask to the batch
        filtered_batch = batch.slice(final_mask)

        print(
            f"Filtered format batch size: {len(filtered_batch)} (from original size: {len(batch)})"
        )

        return filtered_batch

    def filter(self, reward_tensor, batch, n_samples):
        """
        Filter responses based on accuracy and truncation criteria.

        Args:
            reward_tensor: Tensor containing accuracy scores
            batch: DataProto batch containing responses
            n_samples: Number of responses per prompt

        Returns:
            DataProto: Filtered batch
        """
        # First do accuracy filtering if enabled
        if self.config.data.filter_accuracy:
            reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            counts = Counter(acc_tensor.tolist())
            print(
                "Accuracy distribution:",
                " ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items())),
            )

            acc_mask = (acc_tensor >= self.config.data.accuracy_lower_bound) & (
                acc_tensor <= self.config.data.accuracy_upper_bound
            )
        else:
            # If accuracy filtering disabled, keep all samples
            acc_mask = torch.ones(
                len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device
            )
        # Then do truncation filtering if enabled
        if self.config.data.filter_truncated:
            responses = batch.batch["responses"]
            attention_mask = batch.batch["attention_mask"]
            response_mask = attention_mask[:, -responses.size(1) :]

            # Calculate response lengths
            response_lengths = response_mask.sum(-1)  # (batch_size,)
            response_lengths = response_lengths.reshape(
                -1, n_samples
            )  # (num_prompts, n_samples)

            # Get max possible length from config
            max_len = self.config.data.max_response_length

            # Check if any response in the group hits max length (indicating possible truncation)
            has_truncated = (response_lengths >= max_len).any(dim=-1)

            # Print distribution of truncated vs non-truncated
            truncated_counts = Counter(has_truncated.tolist())
            print(
                "Truncation distribution:",
                f"Truncated: {truncated_counts[True] if True in truncated_counts else 0}, "
                f"Non-truncated: {truncated_counts[False] if False in truncated_counts else 0}",
            )
            # Keep only prompts where no response was truncated
            trunc_mask = ~has_truncated
        else:
            # If truncation filtering disabled, keep all samples
            trunc_mask = torch.ones(
                len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device
            )

        # Combine both masks
        combined_mask = acc_mask & trunc_mask

        # Expand mask to cover all samples for each prompt
        final_mask = combined_mask.repeat_interleave(n_samples)

        # Apply the mask to the batch
        filtered_batch = batch.slice(final_mask)

        print(
            f"Filtered batch size: {len(filtered_batch)} (from original size: {len(batch)})"
        )
        return filtered_batch

    def add_to_buffer(self, batch, batch_size, n_samples):
        target_size = int(batch_size) * int(n_samples)
        if len(batch) <= target_size:
            return batch
        if len(batch) % max(int(n_samples), 1) != 0:
            print(
                f"[Batch] add_to_buffer received non group-aligned samples: "
                f"len={len(batch)}, n_samples={n_samples}; keeping first {target_size}.",
                flush=True,
            )
        return batch.slice(slice(0, target_size))

    # Helper: expand a prompt-level DataProto (len=batch_size) to sample-level (len=batch_size * n_repeat)
    def expand_data_proto_to_samples(proto: DataProto, n_repeat: int) -> DataProto:
        if n_repeat <= 1:
            return proto
        parts = []
        for i in range(len(proto)):
            one = proto[i : i + 1]  # single-sample DataProto
            # Repeat each entry n_repeat times to create sample-level expansion
        return DataProto.concat(parts)

    def get_trainer_handle(self, trainer_rank: int = 0):
        # Get the underlying actor handles from self.actor_rollout_wg
        # trainer_rank = 0  # default trainer rank

        # 1) Get actor handles from the worker group (different APIs for different wg types)
        handles = None
        if hasattr(self.actor_rollout_wg, "actors"):  # e.g. wg.actors
            handles = self.actor_rollout_wg.actors
        elif hasattr(self.actor_rollout_wg, "actor_handles"):  # e.g. wg.actor_handles
            handles = self.actor_rollout_wg.actor_handles
        elif hasattr(
            self.actor_rollout_wg, "worker_dict"
        ):  # e.g. WorkerDict internal handles
            handles = list(self.actor_rollout_wg.worker_dict.values())

        if handles is None:
            raise RuntimeError(
                "Cannot find actor handles from actor_rollout_wg; check the worker group API"
            )

        # Get trainer handle by rank (assumes handles are ordered by spawn rank)
        if trainer_rank < 0 or trainer_rank >= len(handles):
            raise IndexError(
                f"trainer_rank={trainer_rank} out of range for {len(handles)} handles"
            )

        trainer_handle = handles[trainer_rank]
        return trainer_handle
