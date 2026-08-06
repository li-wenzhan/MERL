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
The main entry point to run the PPO algorithm

MERL memory patch:
1. Save rollout videos on worker-side datasets/eval shards only.
2. Strip video tensors from returned DataProto to avoid Ray driver OOM.
3. Keep action tensors for PPO/filter contracts.
"""

import gc
import glob
import importlib
import json
import logging
import os
import random
import shutil
import sys
import time
import traceback
import uuid
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import einops
import math
import numpy as np
import ray
import torch

# import torch.distributed
import torch.distributed as dist
import verl.utils.hdfs_io as hdfs_io
import verl.utils.torch_functional as verl_F
from verl.utils.task_description_contract import normalize_task_descriptions

# from colossalai.booster import Booster
# from colossalai.cluster import DistCoordinator
# from colossalai.nn.optimizer import HybridAdam
# from colossalai.utils import get_current_device, set_seed
# from verl.workers.utils import get_current_device, set_seed
import webdataset as wds
from accelerate import Accelerator
from accelerate.utils import set_seed
from codetiming import Timer
from modules.ctrl_world.dataset.dataset_libero_online import DatasetLiberoOnlineV2
from modules.ctrl_world.model_loading import (
    load_trusted_state_dict,
    resolve_ctrl_world_ckpt_path,
)
from modules.ctrl_world.models.ctrl_world_new import CtrlWorld
from modules.ctrl_world.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from modules.ctrl_world.scripts.validate_new import encode_img_to_latent_on_gpu
from modules.utils.video_metric import compute_all_metrics
from modules.opensora.opensora.acceleration.parallel_states import (
    get_data_parallel_group,
)

# from modules.opensora.opensora.datasets.dataloader import prepare_dataloader
from omegaconf import DictConfig, open_dict
from torch.nn import functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_fsdp_wrap_policy_vla,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_grad,
    load_fsdp_optimizer,
    load_fsdp_param_and_grad,
    offload_fsdp_grad,
    offload_fsdp_optimizer,
    offload_fsdp_param_and_grad,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.openvla_utils import check_model_logic_mismatch, update_auto_map
from verl.utils.py_functional import append_to_dict
from verl.workers.utils import prepare_dataloader
from webdataset import ShardWriter, TarWriter

from ..trainer.ppo import core_algos

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


def _sum_metric_value(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().sum().item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value, dtype=np.float64).sum())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return float(np.asarray(value, dtype=np.float64).sum())
    try:
        return float(value)
    except Exception:
        return float(default)


def _prepare_actor_lr_for_update(
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    data: DataProto,
) -> Tuple[float, List[float]]:
    if optimizer is None:
        return 1.0, []
    try:
        lr_scale = float((getattr(data, "meta_info", {}) or {}).get("actor_lr_scale", 1.0))
    except Exception:
        lr_scale = 1.0
    lr_scale = float(np.clip(lr_scale, 0.10, 1.0))
    try:
        base_lrs = list(scheduler.get_last_lr()) if scheduler is not None else []
    except Exception:
        base_lrs = []
    if len(base_lrs) != len(optimizer.param_groups):
        base_lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr) * lr_scale
    return lr_scale, [float(lr) for lr in base_lrs]


def _finalize_actor_lr_after_update(
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    metrics: Dict[str, Any],
    *,
    lr_scale: float,
    base_lrs: List[float],
) -> None:
    optimizer_step_count = _sum_metric_value(
        metrics, "actor/optimizer_step_count", default=1.0
    )
    if scheduler is not None and optimizer_step_count > 0.0:
        scheduler.step()
        metrics["actor/lr_scheduler_skipped"] = 0.0
    elif scheduler is not None:
        metrics["actor/lr_scheduler_skipped"] = 1.0

    try:
        scheduled_lrs = list(scheduler.get_last_lr()) if scheduler is not None else []
    except Exception:
        scheduled_lrs = []
    if len(scheduled_lrs) == 0:
        scheduled_lrs = base_lrs
    if optimizer is not None and len(scheduled_lrs) == len(optimizer.param_groups):
        for group, scheduled_lr in zip(optimizer.param_groups, scheduled_lrs):
            group["lr"] = float(scheduled_lr)

    scheduled_lr = float(scheduled_lrs[0]) if len(scheduled_lrs) > 0 else 0.0
    effective_lr = scheduled_lr * float(lr_scale) if optimizer_step_count > 0.0 else 0.0
    metrics["actor/lr(1e-4)"] = scheduled_lr * 1e4
    metrics["actor/lr_effective(1e-4)"] = effective_lr * 1e4
    metrics["actor/lr_health_scale"] = float(lr_scale)


def _normalize_socket_ifname(ifname: str) -> str:
    return str(ifname or "").split(",")[0].strip()


def _guard_dist_init_env() -> None:
    dist_debug = str(os.environ.get("TORCH_DISTRIBUTED_DEBUG", "")).strip().upper()
    allow_detail = str(
        os.environ.get("VERL_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL", "")
    ).lower() in ("1", "true", "yes", "on")
    if dist_debug == "DETAIL" and not allow_detail:
        logger.warning(
            "Downgrading TORCH_DISTRIBUTED_DEBUG from DETAIL to INFO before init_process_group "
            "to avoid Ray/Gloo wrapper crashes. Set VERL_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL=1 to keep DETAIL."
        )
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    if not _normalize_socket_ifname(os.environ.get("GLOO_SOCKET_IFNAME", "")):
        nccl_ifname = _normalize_socket_ifname(os.environ.get("NCCL_SOCKET_IFNAME", ""))
        if nccl_ifname:
            os.environ["GLOO_SOCKET_IFNAME"] = nccl_ifname
            logger.warning(
                "Setting GLOO_SOCKET_IFNAME=%s from NCCL_SOCKET_IFNAME before init_process_group.",
                nccl_ifname,
            )


def _ensure_dist_process_group(backend: str = "nccl") -> None:
    if dist.is_initialized():
        return
    _guard_dist_init_env()
    dist.init_process_group(backend=backend)


def _find_vla_dataset_statistics_file(*model_dirs: Optional[str]) -> Optional[str]:
    seen_paths = set()
    for model_dir in model_dirs:
        if not model_dir:
            continue
        stats_path = os.path.abspath(
            os.path.join(str(model_dir), "dataset_statistics.json")
        )
        if stats_path in seen_paths:
            continue
        seen_paths.add(stats_path)
        if os.path.isfile(stats_path):
            return stats_path
    return None


def _load_vla_dataset_statistics(
    *model_dirs: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    stats_path = _find_vla_dataset_statistics_file(*model_dirs)
    if stats_path is None:
        return None, None

    with open(stats_path, "r") as file_obj:
        norm_stats = json.load(file_obj)
    return norm_stats, stats_path


def _copy_vla_dataset_statistics(
    target_ckpt_dir: Optional[str], *source_model_dirs: Optional[str]
) -> Optional[str]:
    if not target_ckpt_dir:
        return None

    os.makedirs(target_ckpt_dir, exist_ok=True)
    target_path = os.path.abspath(
        os.path.join(str(target_ckpt_dir), "dataset_statistics.json")
    )
    if os.path.isfile(target_path):
        return target_path

    source_path = _find_vla_dataset_statistics_file(*source_model_dirs)
    if source_path is None:
        return None

    if os.path.abspath(source_path) != target_path:
        shutil.copy2(source_path, target_path)
    return target_path


_VLA_ATTN_IMPLEMENTATION = {
    "openvla-oft": "eager",
    "openvla": "flash_attention_2",
}


def _load_vla_config_from_pretrained(
    vla_name: str,
    pretrained_checkpoint: str,
    trust_remote_code: bool = False,
):
    normalized_name = str(vla_name or "").strip()
    if normalized_name == "openvla-oft":
        from verl.utils.vla_utils.openvla_oft.configuration_prismatic import (
            OpenVLAConfig,
        )

        return OpenVLAConfig.from_pretrained(pretrained_checkpoint)

    if normalized_name == "openvla":
        from verl.utils.vla_utils.openvla.configuration_prismatic import OpenVLAConfig

        return OpenVLAConfig.from_pretrained(pretrained_checkpoint)

    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        pretrained_checkpoint,
        trust_remote_code=trust_remote_code,
    )


def _sync_vla_runtime_files(local_path: str, vla_name: str, rank: int) -> None:
    if rank == 0:
        update_auto_map(local_path)
        check_model_logic_mismatch(local_path, vla_name=vla_name)
    torch.distributed.barrier()


def _load_vla_model_from_pretrained(
    auto_model_cls,
    vla_name: str,
    pretrained_model_name_or_path: str,
    torch_dtype,
    trust_remote_code: bool,
    config=None,
    **extra_kwargs,
):
    normalized_name = str(vla_name or "").strip()
    if normalized_name == "openvla-oft":
        from verl.utils.vla_utils.openvla_oft.modeling_prismatic import (
            OpenVLAForActionPrediction,
        )

        model_cls = OpenVLAForActionPrediction
    elif normalized_name == "openvla":
        from verl.utils.vla_utils.openvla.modeling_prismatic import (
            OpenVLAForActionPrediction,
        )

        model_cls = OpenVLAForActionPrediction
    else:
        model_cls = None

    if model_cls is not None:
        if config is None:
            config = _load_vla_config_from_pretrained(
                normalized_name,
                pretrained_model_name_or_path,
                trust_remote_code=False,
            )
        attn_implementation = _VLA_ATTN_IMPLEMENTATION.get(normalized_name)
        if attn_implementation is not None:
            setattr(config, "_attn_implementation", attn_implementation)
        load_kwargs = dict(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            config=config,
        )
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        load_kwargs.update(extra_kwargs)
        return model_cls.from_pretrained(**load_kwargs)

    load_kwargs = dict(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    attn_implementation = _VLA_ATTN_IMPLEMENTATION.get(normalized_name)
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
        if config is not None:
            setattr(config, "_attn_implementation", attn_implementation)
    if config is not None:
        load_kwargs["config"] = config
    load_kwargs.update(extra_kwargs)
    return auto_model_cls.from_pretrained(**load_kwargs)


def save_output_to_dataset(
    output: DataProto,
    prompts: DataProto,
    rollout_base_dir: str,
    save_train_dataset: bool = True,
):
    # output data
    ## tensors
    responses = output.batch["responses"]
    input_ids = output.batch["input_ids"]
    attention_mask = output.batch["attention_mask"]
    pixel_values = output.batch["pixel_values"]  # 224
    action = output.batch["action"]
    video = output.batch.get("video", None)  # 256
    env_video = output.batch.get("env_video", None)  # 256
    env_dones = output.batch.get("env_dones", None)
    ## tensors of bool & int
    complete = output.batch["complete"]
    finish_step = output.batch["finish_step"]
    is_dummy = output.batch.get("is_dummy", None)

    if video is None and env_video is not None:
        video = env_video

    if video is None:
        raise ValueError(
            "save_output_to_dataset requires rollout output to contain 'video' or 'env_video'; "
            f"got keys {list(output.batch.keys())}"
        )

    if env_video is None:
        env_video = video

    if env_dones is None:
        total_steps = int(video.shape[1]) if video.ndim >= 2 else int(action.shape[1])
        batch_env_dones = []
        for sample_idx in range(len(video)):
            dones = torch.zeros((total_steps,), dtype=torch.int64)
            valid_steps = int(finish_step[sample_idx].item())
            valid_steps = max(0, min(valid_steps, total_steps))
            if valid_steps < total_steps:
                dones[valid_steps:] = 1
            if bool(complete[sample_idx].item()) and valid_steps > 0:
                dones[valid_steps - 1] = 1
            batch_env_dones.append(dones)
        env_dones = torch.stack(batch_env_dones, dim=0)

    if is_dummy is None:
        dummy_steps = responses.shape[1] if responses.ndim >= 2 else 1
        is_dummy = torch.zeros((len(video), dummy_steps), dtype=torch.int64)

    # prompt data
    os.makedirs(rollout_base_dir, exist_ok=True)
    global_steps = prompts.meta_info.get("global_steps", 0)
    eval = prompts.meta_info.get("save_eval", False)
    train_split = str(prompts.meta_info.get("train_split", "train"))
    eval_split = str(prompts.meta_info.get("eval_split", "eval"))
    rank = dist.get_rank()
    # task_descriptions: List[str] = output.meta_info["task_descriptions"]  # [bsz] of str
    task_descriptions = normalize_task_descriptions(
        (output.non_tensor_batch or {}).get("task_descriptions", None),
        len(video),
        context="fsdp_workers.save_rollout_dataset",
    )

    total = len(video)
    if eval:
        val_indices = set()
        train_indices = set()
        val_index = random.randint(
            0, total - 1
        )  # Randomly select an index as the validation set
        val_indices.add(val_index)
        train_indices = (
            set(range(total)) - val_indices
        )  # The rest are used as the training set
    else:
        train_indices = set(range(total))

    def build_sample(i):
        sample_i = {
            "__key__": f"{i:09d}",
            "video.npy": video[i].numpy().astype(np.uint8),  # [steps, H, W, c]
            "action.npy": action[i].numpy().astype(np.float32),
            "input_ids.npy": input_ids[i].numpy().astype(np.int64),
            "attention_mask.npy": attention_mask[i].numpy().astype(np.int64),
            "responses.npy": responses[i].numpy().astype(np.int64),
            "is_dummy.npy": is_dummy[i].numpy().astype(np.int64),
            # "pixel_values.npy": pixel_values[i].numpy().astype(np.float32),
            "pixel_values.npy": pixel_values[i].to(torch.float32).numpy(),
            "meta.json": json.dumps(
                {
                    "complete": complete[i].item(),
                    "finish_step": finish_step[i].item(),
                    "unique_id": str(uuid.uuid4()),
                    "task_description": str(task_descriptions[i]),
                    "train_split": train_split,
                    "eval_split": eval_split,
                }
            ).encode("utf-8"),
        }
        if env_video is not None and env_video.numel() > 0:
            sample_i["env_video.npy"] = (
                env_video[i].numpy().astype(np.uint8)
            )  # [steps, H, W, c]
        if env_dones is not None and env_dones.numel() > 0:
            sample_i["env_dones.npy"] = env_dones[i].numpy().astype(np.uint8)  # [steps]

        return sample_i

    # Save eval dataset
    if eval:
        val_dir = os.path.join(
            rollout_base_dir,
            eval_split,
            f"global_steps_{global_steps}_rank_{rank}",
        )
        os.makedirs(val_dir, exist_ok=True)

        for i in val_indices:
            sample = build_sample(i)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            tar_path = os.path.join(val_dir, f"{timestamp}.tar")
            with TarWriter(tar_path) as sink:
                sink.write(sample)
            print(f"[rank {rank}] Saved eval_dataset.tar: {tar_path}")

    if save_train_dataset:
        # Save train dataset
        train_dir = os.path.join(
            rollout_base_dir,
            train_split,
            f"global_steps_{global_steps}_rank_{rank}",
        )
        os.makedirs(train_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        shard_pattern = os.path.join(train_dir, f"shard_{timestamp}_%05d.tar")
        with ShardWriter(
            shard_pattern, maxsize=4 << 30
        ) as sink:  # 4 * 2^30 bytes = 4 GB
            for i in sorted(train_indices):
                sample = build_sample(i)
                sink.write(sample)
        print(
            f"[rank {rank}] Saved train_dataset.tar in: {train_dir}, totol: {len(train_indices)}"
        )
    else:
        print(
            f"[rank {rank}] Skip saving train dataset (save_train_dataset=False) at global_steps={global_steps}."
        )
    return


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, (ListConfig, DictConfig)):
        return (
            {k: convert_to_regular_types(v) for k, v in obj.items()}
            if isinstance(obj, DictConfig)
            else list(obj)
        )
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


def apply_world_model_overrides(wm_args, wm_overrides):
    overrides = convert_to_regular_types(wm_overrides or {})
    if not isinstance(overrides, dict):
        return wm_args

    alias_targets = {
        "lr": ["learning_rate"],
        "batch_size": [
            "train_batch_size",
            "train_real_batch_size",
            "imag_train_batch_size",
            "eval_real_batch_size",
        ],
    }

    reward_model_cfg = overrides.get("reward_model", None)
    if isinstance(reward_model_cfg, dict):
        reward_thr = reward_model_cfg.get("reward_thr", None)
        if reward_thr is not None:
            setattr(wm_args, "reward_threshold", float(reward_thr))

    for key, value in overrides.items():
        if key in {"config_path", "reward_model"}:
            continue
        if isinstance(value, dict):
            continue
        target_keys = alias_targets.get(key, [key])
        for target_key in target_keys:
            setattr(wm_args, target_key, value)

    return wm_args


def resolve_wm_batch_size(
    wm_args,
    requested_type: str,
    actual_type: str,
    explicit_batch_size: Optional[int] = None,
) -> int:
    if explicit_batch_size is not None:
        return max(1, int(explicit_batch_size))

    candidate_keys = []
    if requested_type in ("train", "eval"):
        candidate_keys.append(f"{requested_type}_batch_size")
    else:
        candidate_keys.append(f"{actual_type}_batch_size")
        candidate_keys.append(f"{requested_type}_batch_size")

    if "train" in requested_type:
        candidate_keys.extend(
            ["train_real_batch_size", "train_batch_size", "batch_size"]
        )
    elif "eval" in requested_type:
        candidate_keys.extend(
            [
                "eval_real_batch_size",
                "eval_batch_size",
                "train_batch_size",
                "batch_size",
            ]
        )

    seen = set()
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        value = getattr(wm_args, key, None)
        if value is None:
            continue
        try:
            return max(1, int(value))
        except Exception:
            continue
    return 1


def format_num_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{float(num_bytes):.2f}B"


def materialize_cpu_state_dict(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _estimate_tensor_like_num_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel()) * int(value.element_size())

    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        try:
            local_tensor = to_local()
            if torch.is_tensor(local_tensor):
                return int(local_tensor.numel()) * int(local_tensor.element_size())
        except Exception:
            pass

    local_tensor = getattr(value, "_local_tensor", None)
    if torch.is_tensor(local_tensor):
        return int(local_tensor.numel()) * int(local_tensor.element_size())

    local_shards = getattr(value, "local_shards", None)
    if callable(local_shards):
        try:
            total_bytes = 0
            for shard in local_shards():
                shard_tensor = getattr(shard, "tensor", None)
                if torch.is_tensor(shard_tensor):
                    total_bytes += int(shard_tensor.numel()) * int(
                        shard_tensor.element_size()
                    )
            if total_bytes > 0:
                return total_bytes
        except Exception:
            pass

    return 0


def estimate_state_dict_num_bytes(state_dict: Dict[str, Any]) -> int:
    total_bytes = 0
    for value in state_dict.values():
        total_bytes += _estimate_tensor_like_num_bytes(value)
    return int(total_bytes)


def safe_save_state_dict_file(
    state_dict: Dict[str, Any],
    save_path: str,
    *,
    reserve_ratio: float = 0.05,
    reserve_bytes: int = 512 * 1024 * 1024,
) -> Dict[str, Any]:
    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)

    estimated_bytes = estimate_state_dict_num_bytes(state_dict)
    free_bytes = None
    try:
        free_bytes = int(shutil.disk_usage(save_dir).free)
    except Exception:
        free_bytes = None

    required_bytes = estimated_bytes + max(
        reserve_bytes, int(float(estimated_bytes) * float(reserve_ratio))
    )
    if (free_bytes is not None) and (free_bytes < required_bytes):
        raise RuntimeError(
            "Insufficient free space for checkpoint save: "
            f"path={save_path}, estimated={format_num_bytes(estimated_bytes)}, "
            f"required~={format_num_bytes(required_bytes)}, available={format_num_bytes(free_bytes)}"
        )

    tmp_path = f"{save_path}.tmp-{uuid.uuid4().hex}"
    try:
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, save_path)
        return {
            "path": save_path,
            "estimated_bytes": estimated_bytes,
            "free_bytes": free_bytes,
        }
    except Exception as e:
        free_after = None
        try:
            free_after = int(shutil.disk_usage(save_dir).free)
        except Exception:
            free_after = None
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(
            "Failed to save checkpoint state_dict: "
            f"path={save_path}, estimated={format_num_bytes(estimated_bytes)}, "
            f"available_before={format_num_bytes(free_bytes)}, "
            f"available_after={format_num_bytes(free_after)}, error={e}"
        ) from e


FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT = "fsdp_local_state_dict"
FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT = "fsdp_sharded_state_dict"
HF_FULL_STATE_DICT_CHECKPOINT_FORMAT = "hf_full_state_dict"
FSDP_CHECKPOINT_META_FILENAME = "checkpoint_meta.json"


def normalize_fsdp_checkpoint_format(config_value: Any) -> str:
    checkpoint_format = (
        str(config_value or FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT).strip().lower()
    )
    if checkpoint_format in {
        "local",
        "local_state_dict",
        "fsdp_local_state_dict",
        "resume_local",
    }:
        return FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT
    if checkpoint_format in {
        "sharded",
        "sharded_state_dict",
        "fsdp_sharded_state_dict",
        "dtensor",
        "resume_sharded",
        "resume_dtensor",
    }:
        return FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT
    if checkpoint_format in {
        "full",
        "full_state_dict",
        "hf_full",
        "hf_full_state_dict",
        "pretrained",
    }:
        return HF_FULL_STATE_DICT_CHECKPOINT_FORMAT
    return checkpoint_format


def is_fsdp_lightweight_checkpoint_format(checkpoint_format: Any) -> bool:
    normalized = str(checkpoint_format or "").strip().lower()
    return normalized in {
        FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT,
        FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT,
    }


def resolve_fsdp_lightweight_save_format(checkpoint_format: Any) -> str:
    normalized = normalize_fsdp_checkpoint_format(checkpoint_format)
    if normalized == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
        return FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT
    return normalized


def _fsdp_checkpoint_state_dict_context(module, checkpoint_format: str):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    normalized = str(checkpoint_format or "").strip().lower()
    state_dict_config = None

    if normalized == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
        try:
            from torch.distributed.fsdp import LocalStateDictConfig

            state_dict_config = LocalStateDictConfig(offload_to_cpu=True)
        except Exception:
            state_dict_config = None
        state_dict_type = StateDictType.LOCAL_STATE_DICT
    elif normalized == FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
        try:
            from torch.distributed.fsdp import ShardedStateDictConfig
        except Exception:
            from torch.distributed.fsdp.api import ShardedStateDictConfig

        state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
        state_dict_type = StateDictType.SHARDED_STATE_DICT
    else:
        raise ValueError(
            f"Unsupported FSDP checkpoint_format={checkpoint_format!r} for lightweight checkpoint context."
        )

    if state_dict_config is None:
        return FSDP.state_dict_type(module, state_dict_type)
    return FSDP.state_dict_type(module, state_dict_type, state_dict_config)


def _fsdp_checkpoint_meta_path(local_path: str) -> str:
    return os.path.join(local_path, FSDP_CHECKPOINT_META_FILENAME)


def _fsdp_checkpoint_rank_path(local_path: str, rank: int) -> str:
    return os.path.join(local_path, f"rank_{int(rank):05d}.pt")


def _fsdp_checkpoint_tmp_path(local_path: str) -> str:
    parent_dir = os.path.dirname(os.path.abspath(local_path))
    checkpoint_name = os.path.basename(os.path.normpath(local_path))
    return os.path.join(parent_dir, f".{checkpoint_name}.incomplete")


def _load_fsdp_checkpoint_meta(local_path: str) -> Optional[Dict[str, Any]]:
    meta_path = _fsdp_checkpoint_meta_path(local_path)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        print(f"[checkpoint] Failed to load metadata from {meta_path}: {exc}")
        return None


def _write_fsdp_checkpoint_meta(local_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(local_path, exist_ok=True)
    meta_path = _fsdp_checkpoint_meta_path(local_path)
    tmp_path = f"{meta_path}.tmp-{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=True, indent=2, sort_keys=True)
    os.replace(tmp_path, meta_path)


def _save_fsdp_lightweight_state_dict_checkpoint(
    module,
    local_path: str,
    *,
    rank: int,
    world_size: int,
    component: str,
    base_model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> Dict[str, Any]:
    checkpoint_format = FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT

    tmp_local_path = _fsdp_checkpoint_tmp_path(local_path)

    if rank == 0:
        if os.path.isdir(tmp_local_path):
            shutil.rmtree(tmp_local_path, ignore_errors=True)
        os.makedirs(tmp_local_path, exist_ok=True)
    dist.barrier()

    with _fsdp_checkpoint_state_dict_context(module, checkpoint_format):
        state_dict = module.state_dict()

    shard_path = _fsdp_checkpoint_rank_path(tmp_local_path, rank)
    save_info = safe_save_state_dict_file(state_dict, shard_path)

    if rank == 0:
        _write_fsdp_checkpoint_meta(
            tmp_local_path,
            {
                "format": checkpoint_format,
                "legacy_format": FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT,
                "state_dict_type": "sharded",
                "component": component,
                "world_size": int(world_size),
                "base_model_path": base_model_path,
                "tokenizer_path": tokenizer_path,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            },
        )
    dist.barrier()

    if rank == 0:
        if os.path.isdir(local_path):
            shutil.rmtree(local_path, ignore_errors=True)
        os.replace(tmp_local_path, local_path)
    dist.barrier()

    return {
        "format": checkpoint_format,
        "component": component,
        "path": _fsdp_checkpoint_rank_path(local_path, rank),
        "estimated_bytes": save_info.get("estimated_bytes"),
        "world_size": int(world_size),
    }


def _load_fsdp_lightweight_state_dict_checkpoint(
    module,
    local_path: str,
    *,
    rank: int,
    world_size: int,
    component: str,
    strict: bool = True,
) -> Dict[str, Any]:
    meta = _load_fsdp_checkpoint_meta(local_path)
    if not isinstance(meta, dict):
        raise FileNotFoundError(
            f"Missing checkpoint metadata under {local_path}; cannot load {component} checkpoint."
        )

    checkpoint_format = str(meta.get("format", "")).strip().lower()
    if checkpoint_format == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
        raise RuntimeError(
            "Legacy fsdp_local_state_dict checkpoints are not resumable in the current DeviceMesh runtime. "
            f"Please resume from a {FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT} or {HF_FULL_STATE_DICT_CHECKPOINT_FORMAT} checkpoint instead."
        )
    if checkpoint_format != FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"Checkpoint at {local_path} uses format={checkpoint_format!r}, expected {FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT!r}."
        )

    expected_world_size = int(meta.get("world_size", world_size))
    if expected_world_size != int(world_size):
        raise RuntimeError(
            f"Checkpoint world_size mismatch for {component}: saved={expected_world_size}, current={int(world_size)}."
        )

    shard_path = _fsdp_checkpoint_rank_path(local_path, rank)
    if not os.path.isfile(shard_path):
        raise FileNotFoundError(
            f"Missing local shard for rank={rank} at {shard_path}; cannot load {component} checkpoint."
        )

    state_dict = torch.load(shard_path, map_location="cpu")
    with _fsdp_checkpoint_state_dict_context(module, checkpoint_format):
        incompatible = module.load_state_dict(state_dict, strict=strict)

    missing_keys = (
        list(getattr(incompatible, "missing_keys", []))
        if incompatible is not None
        else []
    )
    unexpected_keys = (
        list(getattr(incompatible, "unexpected_keys", []))
        if incompatible is not None
        else []
    )
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        raise RuntimeError(
            f"Loaded {component} checkpoint from {local_path}, but found missing_keys={missing_keys} unexpected_keys={unexpected_keys}."
        )

    return {
        "loaded": True,
        "format": checkpoint_format,
        "component": component,
        "path": shard_path,
        "world_size": expected_world_size,
    }


# todo: may deprecate
class RobActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config

        _ensure_dist_process_group(backend="nccl")

        # build device mesh
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        self._is_lora = self.config.model.get("lora_rank", 0) > 0
        self.role = role
        assert self.role in [
            "actor",
            "rollout",
            "ref",
            "actor_rollout",
            "actor_rollout_ref",
        ]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in [
            "rollout",
            "actor_rollout",
            "actor_rollout_ref",
        ]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get(
                "param_offload", False
            )
            self._is_offload_grad = self.config.actor.fsdp_config.get(
                "grad_offload", False
            )
            self._is_offload_optimizer = self.config.actor.fsdp_config.get(
                "optimizer_offload", False
            )
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get(
                "param_offload", False
            )

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.shape[0]
            self.config.actor.ppo_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.shape[0]

        rollout_base_dir = self.config.get("rollout_base_dir", "./tmp_files/rollout")
        self.rollout_base_dir = os.path.abspath(str(rollout_base_dir))
        try:
            self.config.rollout_base_dir = self.rollout_base_dir
        except Exception:
            pass
        os.makedirs(self.rollout_base_dir, exist_ok=True)

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForCausalLM,
            AutoModelForVision2Seq,
            AutoProcessor,
            AutoTokenizer,
        )
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        log_gpu_memory_usage("Before init from HF AutoModel", logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)
        # add oft

        if self.config.model.vla == "openvla-oft":
            from verl.utils.vla_utils.openvla_oft.configuration_prismatic import (
                OpenVLAConfig,
            )
            from verl.utils.vla_utils.openvla_oft.modeling_prismatic import (
                OpenVLAForActionPrediction,
            )
            from verl.utils.vla_utils.openvla_oft.processing_prismatic import (
                PrismaticImageProcessor,
                PrismaticProcessor,
            )

            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            _sync_vla_runtime_files(local_path, self.config.model.vla, self.rank)

        elif self.config.model.vla == "openvla":
            from verl.utils.vla_utils.openvla.configuration_prismatic import (
                OpenVLAConfig,
            )
            from verl.utils.vla_utils.openvla.modeling_prismatic import (
                OpenVLAForActionPrediction,
            )
            from verl.utils.vla_utils.openvla.processing_prismatic import (
                PrismaticImageProcessor,
                PrismaticProcessor,
            )

            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            _sync_vla_runtime_files(local_path, self.config.model.vla, self.rank)

        # add end

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        tokenizer_path = self.config.model.get("tokenizer_path", None)
        tokenizer_local_path = (
            copy_local_path_from_hdfs(tokenizer_path)
            if tokenizer_path is not None
            else local_path
        )
        self.tokenizer = hf_tokenizer(
            tokenizer_local_path,
            trust_remote_code=trust_remote_code,
            model=self.config.model.vla,
        )

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = _load_vla_config_from_pretrained(
            self.config.model.vla,
            local_path,
            trust_remote_code=trust_remote_code,
        )
        if self.config.model.use_remove_padding:
            from verl.models.registry import check_model_support_rmpad

            check_model_support_rmpad(actor_model_config.model_type)
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.config.rollout.use_proprio:
            override_config_kwargs["use_proprio"] = True
            override_config_kwargs["proprio_dim"] = self.config.model.action_token_len
        else:
            override_config_kwargs["use_proprio"] = False
            override_config_kwargs["proprio_dim"] = self.config.model.action_token_len

        override_config_kwargs.update(override_model_config)
        update_model_config(
            actor_model_config, override_config_kwargs=override_config_kwargs
        )
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.config.model.vla == "openvla-oft":
                actor_module = _load_vla_model_from_pretrained(
                    AutoModelForVision2Seq,
                    self.config.model.vla,
                    pretrained_model_name_or_path=local_path,
                    torch_dtype=torch_dtype,
                    config=actor_model_config,
                    trust_remote_code=True,
                )
                if (
                    self.config.rollout.use_proprio
                    and self.config.model.resume == False
                ):
                    # Load proprio projector weights if available
                    actor_module.load_proprio_projector_weights(local_path)
                    print("******Loaded pre-trained proprio projector weights*********")
                # oft add
                actor_module.vision_backbone.set_num_images_in_input(
                    self.config.actor.num_images_in_input
                )

                norm_stats, norm_stats_path = _load_vla_dataset_statistics(
                    local_path, tokenizer_local_path
                )
                if norm_stats is not None:
                    actor_module.norm_stats = norm_stats
                    if os.path.abspath(norm_stats_path) != os.path.abspath(
                        os.path.join(local_path, "dataset_statistics.json")
                    ):
                        print(
                            "Loaded dataset_statistics.json for actor model from fallback path: "
                            f"{norm_stats_path}"
                        )
                else:
                    print(
                        "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
                        "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
                        "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
                    )
            elif self.config.model.vla == "openvla":
                actor_module = _load_vla_model_from_pretrained(
                    AutoModelForVision2Seq,
                    self.config.model.vla,
                    pretrained_model_name_or_path=local_path,
                    torch_dtype=torch_dtype,
                    config=actor_model_config,
                    trust_remote_code=True,
                )

            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable()
            # lora add
            if self._is_lora:
                print("Applying LoRA to actor module")

                lora_config = {
                    #'task_type': TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "lora_dropout": 0,
                    "target_modules": convert_to_regular_types(
                        self.config.model.target_modules
                    ),
                    "init_lora_weights": "gaussian",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
                actor_module.print_trainable_parameters()
            # lora end

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage("After init from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        if self._is_ref:
            mixed_precision = None

        # oft add
        auto_wrap_policy = get_fsdp_wrap_policy_vla(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        # oft add end

        print(f"wrap_policy: {auto_wrap_policy}")

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(
            actor_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
        )

        log_gpu_memory_usage("After Actor FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 0.0),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            actor_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
            )
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        return (
            actor_module_fsdp,
            actor_optimizer,
            actor_lr_scheduler,
            actor_model_config,
        )

    def _build_rollout(self):
        if self.config.rollout.name == "hf":
            from verl.utils.libero_path import (
                ensure_libero_pro_root,
                ensure_libero_root,
            )
            from verl.workers.hybrid_engine import BaseShardingManager

            if getattr(self.config.rollout, "use_libero_pro", False):
                ensure_libero_pro_root(
                    evaluation_config_path=getattr(
                        self.config.rollout, "libero_pro_eval_config_path", None
                    )
                )
                from verl.workers.rollout import RobWMHFRolloutPro

                rollout = RobWMHFRolloutPro(
                    module=self.actor_module_fsdp,
                    config=self.config.rollout,
                    world_model_mapping=None,
                )
            else:
                ensure_libero_root()
                from verl.workers.rollout import RobHFRollout

                rollout = RobHFRollout(
                    module=self.actor_module_fsdp, config=self.config.rollout
                )
            sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == "vllm":
            raise ValueError
            # from verl.workers.rollout.vllm_rollout import vLLMRollout
            # from verl.workers.hybrid_engine import FSDPVLLMShardingManager
            # log_gpu_memory_usage('Before building vllm rollout', logger=None)
            # rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
            #                       config=self.config.rollout,
            #                       tokenizer=self.tokenizer,
            #                       model_hf_config=self.actor_model_config)
            # log_gpu_memory_usage('After building vllm rollout', logger=None)
            # if torch.distributed.get_world_size() == 1:
            #     self.config.rollout.load_format = 'dummy_hf'
            # sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
            #                                            inference_engine=rollout.inference_engine,
            #                                            model_config=self.actor_model_config,
            #                                            full_params='hf' in self.config.rollout.load_format)
            # log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import RobDataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                enable_gradient_checkpointing=self.config.model.get(
                    "enable_gradient_checkpointing", False
                ),
                trust_remote_code=True,
            )  # self.config.model.get('trust_remote_code', True)

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage(
                    "After offload actor grad during init", logger=logger
                )
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage(
                    "After offload actor optimizer during init", logger=logger
                )
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = RobDataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                trust_remote_code=True,
            )[
                0
            ]  # self.config.model.get('trust_remote_code', False)

            if self._is_offload_param:
                offload_fsdp_param_and_grad(
                    module=self.ref_module_fsdp, offload_grad=self._is_offload_grad
                )

            OmegaConf.set_struct(self.config.ref, True)
            self.ref_policy = RobDataParallelPPOActor(
                config=self.config.ref, actor_module=self.ref_module_fsdp
            )

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.actor_optimizer, device_id=torch.cuda.current_device()
            )

        # data.batch = data.batch.cuda()

        log_gpu_memory_usage("Before update policy", logger=logger)

        actor_lr_scale, actor_base_lrs = _prepare_actor_lr_for_update(
            self.actor_optimizer, self.actor_lr_scheduler, data
        )
        metrics = self.actor.update_policy(data=data)

        _finalize_actor_lr_after_update(
            self.actor_optimizer,
            self.actor_lr_scheduler,
            metrics,
            lr_scale=actor_lr_scale,
            base_lrs=actor_base_lrs,
        )

        log_gpu_memory_usage("After update policy", logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_entropy(self, data: DataProto):

        data = data.to("cuda")

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        data.batch = data.batch.cuda()

        log_gpu_memory_usage("Before compute entropy", logger=logger)

        metrics = self.actor.compute_entropy(bacth_data=data)

        log_gpu_memory_usage("After compute entropy", logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        prompts = prompts.to("cuda")
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get("recompute_log_prob", True)
        save_to_hdfs = prompts.meta_info.get("save_to_hdfs", False)
        save_eval = prompts.meta_info.get("save_eval", False)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        prompts.batch = prompts.batch.cuda()
        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        # tmp_sample = prompts.meta_info.get('n_samples', -1)
        # with Timer(name=f'gen seq will start, and the num samples are: {tmp_sample}', text="{name}: {seconds:.1f} seconds") as timer:
        #     print(f"gen seq will start, and the num samples are: {tmp_sample}")

        with self.sharding_manager:
            log_gpu_memory_usage("After entering sharding manager", logger=logger)
            prompts = self.sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.sharding_manager.postprocess_data(output)
            torch.cuda.synchronize()

        # with Timer(name=f'gen seq end ,  old log will begin', text="{name}: {seconds:.1f} seconds") as timer:
        #     print("gen seq end ,  old log will begin")
        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            gc.collect()
            torch.cuda.empty_cache()
            output.meta_info["micro_batch_size"] = (
                self.config.rollout.log_prob_micro_batch_size
            )
            print(f"[fsdp] micro_batch_size: {output.meta_info['micro_batch_size']}")
            output.meta_info["temperature"] = self.config.rollout.temperature
            output.meta_info["use_dynamic_bsz"] = (
                self.config.rollout.log_prob_use_dynamic_bsz
            )
            output.meta_info["max_token_len"] = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            output.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
            old_log_probs = self.actor.compute_log_prob(data=output)
            output.batch["old_log_probs"] = old_log_probs

        output = output.to("cpu")

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )

        # clear kv cache
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After recompute log prob", logger=logger)

        if save_to_hdfs or save_eval:
            save_output_to_dataset(
                output,
                prompts,
                self.rollout_base_dir,
                save_train_dataset=bool(save_to_hdfs),
            )
        if bool(prompts.meta_info.get("strip_rollout_media", True)):
            media_keys = [
                key
                for key in ("video", "env_video", "env_dones")
                if key in output.batch
            ]
            if media_keys:
                output.pop(batch_keys=media_keys)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to("cuda")

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.ref_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={"ref_log_prob": output})

        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.ref_module_fsdp, offload_grad=self._is_offload_grad
            )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor

        import torch.distributed as dist
        import transformers
        from peft import PeftModel
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoModelForVision2Seq

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        checkpoint_format = normalize_fsdp_checkpoint_format(
            getattr(self.config.model, "checkpoint_format", None)
        )
        if checkpoint_format not in {
            FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT,
            FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT,
            HF_FULL_STATE_DICT_CHECKPOINT_FORMAT,
        }:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Unknown actor checkpoint_format={checkpoint_format!r}; fallback to {FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT}."
                )
            checkpoint_format = FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT
        elif checkpoint_format == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Actor checkpoint_format={checkpoint_format!r} is incompatible with DeviceMesh autosave; upgrade to {FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT}."
                )
            checkpoint_format = resolve_fsdp_lightweight_save_format(checkpoint_format)

        if checkpoint_format == FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
            save_info = _save_fsdp_lightweight_state_dict_checkpoint(
                self.actor.actor_module,
                local_path,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                component="actor",
                base_model_path=getattr(self.config.model, "path", None),
                tokenizer_path=getattr(self.config.model, "tokenizer_path", None),
            )
            if dist.get_rank() == 0:
                print(
                    f"Saved lightweight actor checkpoint to {local_path} ({format_num_bytes(save_info.get('estimated_bytes')) if save_info.get('estimated_bytes') is not None else 'size=unknown'})"
                )
                if hdfs_path is not None:
                    print(f"Uploading actor checkpoint to {hdfs_path}")
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=local_path, dst=hdfs_path)
            if self._is_offload_param:
                offload_fsdp_param_and_grad(
                    module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
                )
            return

        # lora add
        if self._is_lora and isinstance(self.actor_module, PeftModel):
            if dist.get_rank() == 0:
                os.makedirs(local_path, exist_ok=True)

            lora_save_path = os.path.join(local_path, "lora_adapter")

            if isinstance(self.actor_module_fsdp, FSDP):
                with FSDP.summon_full_params(
                    self.actor_module_fsdp, writeback=False, offload_to_cpu=True
                ):
                    if dist.get_rank() == 0:
                        from typing import OrderedDict

                        lora_params = OrderedDict()
                        model = (
                            self.actor_module_fsdp._fsdp_wrapped_module.base_model.model
                        )
                        for name, param in model.named_parameters():
                            if ".lora_" in name:
                                name = "base_model.model." + name.replace(
                                    "._fsdp_wrapped_module.", "."
                                )
                                lora_params[name] = param
                        self.actor_module_fsdp.save_pretrained(
                            lora_save_path,
                            state_dict=lora_params,
                            safe_serialization=True,
                        )
            else:
                self.actor_module.save_pretrained(
                    lora_save_path, safe_serialization=True
                )

            dist.barrier()
            if dist.get_rank() == 0:
                print(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}")

            # save total model
            base_vla = _load_vla_model_from_pretrained(
                AutoModelForVision2Seq,
                self.config.model.vla,
                self.config.model.path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="cpu",
            )
            merged_vla = PeftModel.from_pretrained(base_vla, lora_save_path)
            merged_vla = merged_vla.merge_and_unload()

            if dist.get_rank() == 0:
                merged_vla.save_pretrained(local_path)
                print(f"Saved merged model at: {local_path}")

            # Wait for merged model to be saved
            dist.barrier()

        # TODO: support DCP and save sharded checkpoints
        else:
            import torch.distributed
            from torch.distributed.fsdp import FullStateDictConfig
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg
            ):
                state_dict = self.actor.actor_module.state_dict()
            if self.rank == 0:
                print(f"Saving actor checkpoint to {local_path}")
                os.makedirs(local_path, exist_ok=True)
                self.actor_module.save_pretrained(local_path, state_dict=state_dict)
                self.tokenizer.save_pretrained(local_path)
                synced_stats_path = _copy_vla_dataset_statistics(
                    local_path,
                    self.config.model.path,
                    getattr(self.config.model, "tokenizer_path", None),
                )
                if synced_stats_path is not None:
                    print(f"Saved dataset statistics to {synced_stats_path}")
                if hdfs_path is not None:
                    print(f"Uploading actor checkpoint to {hdfs_path}")
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor

        checkpoint_meta = _load_fsdp_checkpoint_meta(local_path)
        checkpoint_format = (
            str((checkpoint_meta or {}).get("format", "")).strip().lower()
        )
        if checkpoint_format == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
            raise RuntimeError(
                "Legacy fsdp_local_state_dict actor checkpoints are not resumable with the current DeviceMesh runtime."
            )
        if checkpoint_format != FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Actor checkpoint at {local_path} is format={checkpoint_format or 'unknown'}; assume HF/preloaded path and skip explicit shard restore."
                )
            torch.distributed.barrier()
            return {"loaded": False, "format": checkpoint_format or "unknown"}

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        load_info = _load_fsdp_lightweight_state_dict_checkpoint(
            self.actor.actor_module,
            local_path,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            component="actor",
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        torch.cuda.empty_cache()
        return load_info


class RobWMActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config

        _ensure_dist_process_group(backend="nccl")

        # build device mesh
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        self._is_lora = self.config.model.get("lora_rank", 0) > 0
        self.role = role
        assert self.role in [
            "actor",
            "rollout",
            "ref",
            "actor_rollout",
            "actor_rollout_ref",
        ]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in [
            "rollout",
            "actor_rollout",
            "actor_rollout_ref",
        ]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get(
                "param_offload", False
            )
            self._is_offload_grad = self.config.actor.fsdp_config.get(
                "grad_offload", False
            )
            self._is_offload_optimizer = self.config.actor.fsdp_config.get(
                "optimizer_offload", False
            )
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get(
                "param_offload", False
            )

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.shape[0]
            self.config.actor.ppo_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.shape[0]

        self.rollout_base_dir = os.path.abspath(self.config.rollout_base_dir)
        self.config.rollout_base_dir = self.rollout_base_dir
        os.makedirs(self.rollout_base_dir, exist_ok=True)

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForCausalLM,
            AutoModelForVision2Seq,
            AutoProcessor,
            AutoTokenizer,
        )
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        log_gpu_memory_usage("Before init from HF AutoModel", logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)
        # add oft

        if self.config.model.vla == "openvla-oft":
            from verl.utils.vla_utils.openvla_oft.configuration_prismatic import (
                OpenVLAConfig,
            )
            from verl.utils.vla_utils.openvla_oft.modeling_prismatic import (
                OpenVLAForActionPrediction,
            )
            from verl.utils.vla_utils.openvla_oft.processing_prismatic import (
                PrismaticImageProcessor,
                PrismaticProcessor,
            )

            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            _sync_vla_runtime_files(local_path, self.config.model.vla, self.rank)

        elif self.config.model.vla == "openvla":
            from verl.utils.vla_utils.openvla.configuration_prismatic import (
                OpenVLAConfig,
            )
            from verl.utils.vla_utils.openvla.modeling_prismatic import (
                OpenVLAForActionPrediction,
            )
            from verl.utils.vla_utils.openvla.processing_prismatic import (
                PrismaticImageProcessor,
                PrismaticProcessor,
            )

            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            _sync_vla_runtime_files(local_path, self.config.model.vla, self.rank)

        # add end

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        tokenizer_path = self.config.model.get("tokenizer_path", None)
        tokenizer_local_path = (
            copy_local_path_from_hdfs(tokenizer_path)
            if tokenizer_path is not None
            else local_path
        )
        self.tokenizer = hf_tokenizer(
            tokenizer_local_path,
            trust_remote_code=trust_remote_code,
            model=self.config.model.vla,
        )

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = _load_vla_config_from_pretrained(
            self.config.model.vla,
            local_path,
            trust_remote_code=trust_remote_code,
        )
        if self.config.model.use_remove_padding:
            from verl.models.registry import check_model_support_rmpad

            check_model_support_rmpad(actor_model_config.model_type)
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.config.rollout.use_proprio:
            override_config_kwargs["use_proprio"] = True
            override_config_kwargs["proprio_dim"] = self.config.model.action_token_len
        else:
            override_config_kwargs["use_proprio"] = False
            override_config_kwargs["proprio_dim"] = self.config.model.action_token_len

        override_config_kwargs.update(override_model_config)
        update_model_config(
            actor_model_config, override_config_kwargs=override_config_kwargs
        )
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.config.model.vla == "openvla-oft":
                actor_module = _load_vla_model_from_pretrained(
                    AutoModelForVision2Seq,
                    self.config.model.vla,
                    pretrained_model_name_or_path=local_path,
                    torch_dtype=torch_dtype,
                    config=actor_model_config,
                    trust_remote_code=True,
                )
                if (
                    self.config.rollout.use_proprio
                    and self.config.model.resume == False
                ):
                    # Load proprio projector weights if available
                    actor_module.load_proprio_projector_weights(local_path)
                    print("******Loaded pre-trained proprio projector weights*********")
                # oft add
                actor_module.vision_backbone.set_num_images_in_input(
                    self.config.actor.num_images_in_input
                )

                norm_stats, norm_stats_path = _load_vla_dataset_statistics(
                    local_path, tokenizer_local_path
                )
                if norm_stats is not None:
                    actor_module.norm_stats = norm_stats
                    if os.path.abspath(norm_stats_path) != os.path.abspath(
                        os.path.join(local_path, "dataset_statistics.json")
                    ):
                        print(
                            "Loaded dataset_statistics.json for actor model from fallback path: "
                            f"{norm_stats_path}"
                        )
                else:
                    print(
                        "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
                        "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
                        "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
                    )
            elif self.config.model.vla == "openvla":
                actor_module = _load_vla_model_from_pretrained(
                    AutoModelForVision2Seq,
                    self.config.model.vla,
                    pretrained_model_name_or_path=local_path,
                    torch_dtype=torch_dtype,
                    config=actor_model_config,
                    trust_remote_code=True,
                )

            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable()
            # lora add
            if self._is_lora:
                print("Applying LoRA to actor module")

                lora_config = {
                    #'task_type': TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "lora_dropout": 0,
                    "target_modules": convert_to_regular_types(
                        self.config.model.target_modules
                    ),
                    "init_lora_weights": "gaussian",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
                actor_module.print_trainable_parameters()
            # lora end

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage("After init from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        if self._is_ref:
            mixed_precision = None

        # oft add
        auto_wrap_policy = get_fsdp_wrap_policy_vla(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        # oft add end

        print(f"wrap_policy: {auto_wrap_policy}")

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(
            actor_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
        )

        log_gpu_memory_usage("After Actor FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 0.0),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            actor_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
            )
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        return (
            actor_module_fsdp,
            actor_optimizer,
            actor_lr_scheduler,
            actor_model_config,
        )

    def _build_rollout(self):
        if self.config.rollout.name == "hf":
            from verl.utils.libero_path import (
                ensure_libero_pro_root,
                ensure_libero_root,
            )
            from verl.workers.hybrid_engine import BaseShardingManager

            #! use wm rollouter
            rollout = None
            if not getattr(self.config.rollout, "use_libero_pro", False):
                ensure_libero_root()
                from verl.workers.rollout import RobWMHFRollout

                # Use the original libero environment by default
                rollout = RobWMHFRollout(
                    module=self.actor_module_fsdp,
                    config=self.config.rollout,
                    world_model_mapping=self.world_model_mapping,
                )
            else:
                ensure_libero_pro_root(
                    evaluation_config_path=getattr(
                        self.config.rollout, "libero_pro_eval_config_path", None
                    )
                )
                from verl.workers.rollout import RobWMHFRolloutPro

                rollout = RobWMHFRolloutPro(
                    module=self.actor_module_fsdp,
                    config=self.config.rollout,
                    world_model_mapping=self.world_model_mapping,
                )
            sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == "vllm":
            raise ValueError("vllm rollout not supported yet.")

        return rollout, sharding_manager

    #! wm added
    @staticmethod
    def _read_config(cfg_path):
        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"The configuration file does not exist: {cfg_path}"
            )
        if cfg_path.suffix != ".py":
            raise ValueError(f"The configuration file must be a .py file.: {cfg_path}")

        original_path = sys.path.copy()
        try:
            cfg_dir = cfg_path.parent.absolute()
            cfg_name = cfg_path.stem
            if str(cfg_dir) not in sys.path:
                sys.path.insert(0, str(cfg_dir))
            config_module = importlib.import_module(cfg_name)
            if not hasattr(config_module, "wm_args"):
                raise AttributeError(
                    f"The 'wm_args' class was not found in the configuration file {cfg_path}"
                )
            config_instance = config_module.wm_args()
            return config_instance
        except Exception as e:
            raise RuntimeError(
                f"Failed to read the configuration file {cfg_path}: {str(e)}"
            ) from e
        finally:
            sys.path = original_path

    #! wm added, v4
    def _build_world_model_mapping(self) -> Dict[str, Any]:
        """
        Build inference-only world model mapping for rollout workers.
        """
        wm_args = self._read_config(self.config.world_model.config_path)
        wm_args = apply_world_model_overrides(wm_args, self.config.world_model)
        if not hasattr(wm_args, "_dtype_obj"):
            dtype = torch.bfloat16
        elif isinstance(wm_args.dtype_obj, torch.dtype):
            dtype = wm_args.dtype_obj
        else:
            raise TypeError(f"Invalid dtype type: {type(wm_args.dtype_obj)}")
        device = torch.cuda.current_device()
        world_model = CtrlWorld(wm_args)

        # load checkpoint if provided
        ckpt_path = resolve_ctrl_world_ckpt_path(wm_args)
        if ckpt_path is not None:
            state_dict = load_trusted_state_dict(ckpt_path, map_location="cpu")
            world_model.load_state_dict(state_dict, strict=True)

        world_model.to(device=device, dtype=dtype)
        world_model.eval()  #! Always eval when building wm

        return {
            "world_model": world_model,
            "wm_args": wm_args,
            "rm_threshold": float(
                getattr(
                    wm_args,
                    "reward_threshold",
                    self.config.world_model.reward_model.reward_thr,
                )
            ),
            "device": device,
            "dtype": dtype,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        #! wm added
        if self.config.world_model.enable:
            self.world_model_mapping = self._build_world_model_mapping()
            print(f"World Model has been initialized on {dist.get_rank()}.")

        from verl.workers.actor import RobDataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                enable_gradient_checkpointing=self.config.model.get(
                    "enable_gradient_checkpointing", False
                ),
                trust_remote_code=True,
            )  # self.config.model.get('trust_remote_code', True)

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage(
                    "After offload actor grad during init", logger=logger
                )
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage(
                    "After offload actor optimizer during init", logger=logger
                )

        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = RobDataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )
            print("Actor has been initialized.")

        #!
        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout()
            print("Rollouter has been initialized.")

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                trust_remote_code=True,
            )[
                0
            ]  # self.config.model.get('trust_remote_code', False)

            if self._is_offload_param:
                offload_fsdp_param_and_grad(
                    module=self.ref_module_fsdp, offload_grad=self._is_offload_grad
                )

            OmegaConf.set_struct(self.config.ref, True)
            self.ref_policy = RobDataParallelPPOActor(
                config=self.config.ref, actor_module=self.ref_module_fsdp
            )
            print("Ref policy has been initialized.")

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.actor_optimizer, device_id=torch.cuda.current_device()
            )

        # data.batch = data.batch.cuda()

        log_gpu_memory_usage("Before update policy", logger=logger)

        actor_lr_scale, actor_base_lrs = _prepare_actor_lr_for_update(
            self.actor_optimizer, self.actor_lr_scheduler, data
        )
        metrics = self.actor.update_policy(data=data)

        _finalize_actor_lr_after_update(
            self.actor_optimizer,
            self.actor_lr_scheduler,
            metrics,
            lr_scale=actor_lr_scale,
            base_lrs=actor_base_lrs,
        )

        log_gpu_memory_usage("After update policy", logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_entropy(self, data: DataProto):

        data = data.to("cuda")

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        data.batch = data.batch.cuda()

        log_gpu_memory_usage("Before compute entropy", logger=logger)

        metrics = self.actor.compute_entropy(bacth_data=data)

        log_gpu_memory_usage("After compute entropy", logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    #! modified
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto, use_wm: bool = False) -> DataProto:
        prompts = prompts.to("cuda")
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get("recompute_log_prob", True)
        use_wm = prompts.meta_info.get("use_wm", False)
        save_to_hdfs = prompts.meta_info.get("save_to_hdfs", False)
        save_eval = prompts.meta_info.get("save_eval", False)
        return_rollouts = prompts.meta_info.get("return_rollouts", False)
        assert self._is_rollout

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        prompts.batch = prompts.batch.cuda()
        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        # tmp_sample = prompts.meta_info.get('n_samples', -1)
        # with Timer(name=f'gen seq will start, and the num samples are: {tmp_sample}', text="{name}: {seconds:.1f} seconds") as timer:
        #     print(f"gen seq will start, and the num samples are: {tmp_sample}")

        with self.sharding_manager:
            log_gpu_memory_usage("After entering sharding manager", logger=logger)
            prompts = self.sharding_manager.preprocess_data(prompts)
            if use_wm:
                try:
                    output = self.rollout.generate_sequences_evolving(prompts=prompts)
                except Exception as e:
                    tb = traceback.format_exc()
                    rank = dist.get_rank() if dist.is_initialized() else -1
                    err_log = f"./tmp_files/fsdp_worker_error_{int(time.time())}_rank{rank}.log"
                    with open(err_log, "w") as f:
                        f.write("EXCEPTION in rollout.generate_sequences_evolving\n")
                        f.write(tb)
                    # flush to stdout/stderr so Ray picks it up
                    print(
                        f"[FATAL] rollout failed on rank={rank}, wrote {err_log}",
                        file=sys.stderr,
                    )
                    # re-raise so Ray will surface the worker exception immediately
                    raise
            else:
                output = self.rollout.generate_sequences(prompts=prompts)
            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.sharding_manager.postprocess_data(output)
            torch.cuda.synchronize()

        # gc.collect()  #! added gc collect
        # with Timer(name=f'gen seq end ,  old log will begin', text="{name}: {seconds:.1f} seconds") as timer:
        #     print("gen seq end ,  old log will begin")

        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            gc.collect()
            torch.cuda.empty_cache()
            output.meta_info["micro_batch_size"] = (
                self.config.rollout.log_prob_micro_batch_size
            )
            print(f"[fsdp] micro_batch_size: {output.meta_info['micro_batch_size']}")
            output.meta_info["temperature"] = self.config.rollout.temperature
            output.meta_info["use_dynamic_bsz"] = (
                self.config.rollout.log_prob_use_dynamic_bsz
            )
            output.meta_info["max_token_len"] = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            output.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
            old_log_probs = self.actor.compute_log_prob(data=output)
            output.batch["old_log_probs"] = old_log_probs
        output: DataProto = output.to("cpu")
        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )

        # clear kv cache
        torch.cuda.synchronize()
        assert torch.distributed.is_initialized()  #! 确认为dist模式
        torch.distributed.barrier()  # 保证同步
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After recompute log prob", logger=logger)

        # log_gpu_memory_usage("Before save output to hdfs", logger=logger)
        if save_to_hdfs or save_eval:
            save_output_to_dataset(
                output,
                prompts,
                self.rollout_base_dir,
                save_train_dataset=bool(save_to_hdfs),
            )
        if bool(prompts.meta_info.get("strip_rollout_media", True)):
            media_keys = [
                key
                for key in ("video", "env_video", "env_dones")
                if key in output.batch
            ]
            if media_keys:
                output.pop(
                    batch_keys=media_keys
                )  # delete rollout media before Ray return
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to("cuda")

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.ref_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={"ref_log_prob": output})

        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.ref_module_fsdp, offload_grad=self._is_offload_grad
            )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor

        import torch
        import torch.distributed as dist
        import transformers
        from peft import PeftModel
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoModelForVision2Seq

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        checkpoint_format = normalize_fsdp_checkpoint_format(
            getattr(self.config.model, "checkpoint_format", None)
        )
        if checkpoint_format not in {
            FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT,
            FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT,
            HF_FULL_STATE_DICT_CHECKPOINT_FORMAT,
        }:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Unknown actor checkpoint_format={checkpoint_format!r}; fallback to {FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT}."
                )
            checkpoint_format = FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT
        elif checkpoint_format == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Actor checkpoint_format={checkpoint_format!r} is incompatible with DeviceMesh autosave; upgrade to {FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT}."
                )
            checkpoint_format = resolve_fsdp_lightweight_save_format(checkpoint_format)

        if checkpoint_format == FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
            save_info = _save_fsdp_lightweight_state_dict_checkpoint(
                self.actor.actor_module,
                local_path,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                component="actor",
                base_model_path=getattr(self.config.model, "path", None),
                tokenizer_path=getattr(self.config.model, "tokenizer_path", None),
            )
            if dist.get_rank() == 0:
                print(
                    f"Saved lightweight actor checkpoint to {local_path} ({format_num_bytes(save_info.get('estimated_bytes')) if save_info.get('estimated_bytes') is not None else 'size=unknown'})"
                )
                if hdfs_path is not None:
                    print(f"Uploading actor checkpoint to {hdfs_path}")
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=local_path, dst=hdfs_path)
            if self._is_offload_param:
                offload_fsdp_param_and_grad(
                    module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
                )
            return

        # lora add
        if self._is_lora and isinstance(self.actor_module, PeftModel):
            if dist.get_rank() == 0:
                os.makedirs(local_path, exist_ok=True)

            lora_save_path = os.path.join(local_path, "lora_adapter")

            if isinstance(self.actor_module_fsdp, FSDP):
                with FSDP.summon_full_params(
                    self.actor_module_fsdp, writeback=False, offload_to_cpu=True
                ):
                    if dist.get_rank() == 0:
                        from typing import OrderedDict

                        lora_params = OrderedDict()
                        model = (
                            self.actor_module_fsdp._fsdp_wrapped_module.base_model.model
                        )
                        for name, param in model.named_parameters():
                            if ".lora_" in name:
                                name = "base_model.model." + name.replace(
                                    "._fsdp_wrapped_module.", "."
                                )
                                lora_params[name] = param
                        self.actor_module_fsdp.save_pretrained(
                            lora_save_path,
                            state_dict=lora_params,
                            safe_serialization=True,
                        )
            else:
                self.actor_module.save_pretrained(
                    lora_save_path, safe_serialization=True
                )

            dist.barrier()
            if dist.get_rank() == 0:
                print(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}")

            # save total model
            base_vla = _load_vla_model_from_pretrained(
                AutoModelForVision2Seq,
                self.config.model.vla,
                self.config.model.path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="cpu",
            )
            merged_vla = PeftModel.from_pretrained(base_vla, lora_save_path)
            merged_vla = merged_vla.merge_and_unload()

            if dist.get_rank() == 0:
                merged_vla.save_pretrained(local_path)
                print(f"Saved merged model at: {local_path}")

            # Wait for merged model to be saved
            dist.barrier()

        # TODO: support DCP and save sharded checkpoints
        else:
            import torch.distributed
            from torch.distributed.fsdp import FullStateDictConfig
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg
            ):
                state_dict = self.actor.actor_module.state_dict()
            if self.rank == 0:
                print(f"Saving actor checkpoint to {local_path}")
                os.makedirs(local_path, exist_ok=True)
                self.actor_module.save_pretrained(local_path, state_dict=state_dict)
                self.tokenizer.save_pretrained(local_path)
                synced_stats_path = _copy_vla_dataset_statistics(
                    local_path,
                    self.config.model.path,
                    getattr(self.config.model, "tokenizer_path", None),
                )
                if synced_stats_path is not None:
                    print(f"Saved dataset statistics to {synced_stats_path}")
                if hdfs_path is not None:
                    print(f"Uploading actor checkpoint to {hdfs_path}")
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor

        checkpoint_meta = _load_fsdp_checkpoint_meta(local_path)
        checkpoint_format = (
            str((checkpoint_meta or {}).get("format", "")).strip().lower()
        )
        if checkpoint_format == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
            raise RuntimeError(
                "Legacy fsdp_local_state_dict actor checkpoints are not resumable with the current DeviceMesh runtime."
            )
        if checkpoint_format != FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Actor checkpoint at {local_path} is format={checkpoint_format or 'unknown'}; assume HF/preloaded path and skip explicit shard restore."
                )
            torch.distributed.barrier()
            return {"loaded": False, "format": checkpoint_format or "unknown"}

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        load_info = _load_fsdp_lightweight_state_dict_checkpoint(
            self.actor.actor_module,
            local_path,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            component="actor",
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        torch.cuda.empty_cache()
        return load_info

    #! wm added: may be deprecated
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_world_model_mapping(self, actor_local_path: str):
        checkpoint_file = os.path.join(actor_local_path, "world_model.pth")
        assert os.path.isfile(
            checkpoint_file
        ), f"Error: Missing checkpoint: {checkpoint_file}"
        # load checkpoint on CPU
        state_dict = torch.load(checkpoint_file, map_location="cpu")
        world_model: CtrlWorld = self.world_model_mapping["world_model"]
        # load weights
        world_model.load_state_dict(state_dict, strict=True)
        world_model.eval()
        # safety check
        assert (
            not world_model.training
        ), "World model must stay in eval mode in rollout workers"
        # light cleanup only
        torch.cuda.empty_cache()
        return {"loaded": True, "path": checkpoint_file}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_world_model_mapping_from_trainer(
        self, wm_actor_name: str = "world_model_trainer"
    ):
        wm_map = getattr(self, "world_model_mapping", None)
        if wm_map is None:
            return {"loaded": False, "error": "no world_model_mapping"}

        world_model: CtrlWorld = wm_map.get("world_model", None)
        if world_model is None:
            return {"loaded": False, "error": "no world_model in mapping"}

        use_dist = dist.is_available() and dist.is_initialized()
        src_rank = 0
        local_rank = dist.get_rank() if use_dist else 0
        status_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if not use_dist:
            wm_trainer = ray.get_actor(wm_actor_name)
            state_dict = ray.get(wm_trainer.export_world_model_state_dict.remote())
            world_model.load_state_dict(state_dict, strict=True)
            world_model.eval()
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
            return {"loaded": True, "source": wm_actor_name, "mode": "memory"}

        status = torch.zeros(1, dtype=torch.int32, device=status_device)
        error_message = [""]

        if local_rank == src_rank:
            state_dict = None
            try:
                wm_trainer = ray.get_actor(wm_actor_name)
                state_dict = ray.get(wm_trainer.export_world_model_state_dict.remote())
                world_model.load_state_dict(state_dict, strict=True)
                world_model.eval()
                status.fill_(1)
            except Exception as e:
                error_message[0] = repr(e)
                status.fill_(0)
            finally:
                try:
                    del state_dict
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

        dist.broadcast(status, src=src_rank)
        if int(status.item()) != 1:
            dist.broadcast_object_list(error_message, src=src_rank)
            dist.barrier()
            raise RuntimeError(
                f"Failed to sync world model from trainer '{wm_actor_name}': {error_message[0]}"
            )

        with torch.no_grad():
            for _, param in world_model.named_parameters():
                dist.broadcast(param.data, src=src_rank)
            for _, buffer in world_model.named_buffers():
                dist.broadcast(buffer.data, src=src_rank)

        world_model.eval()
        dist.barrier()
        torch.cuda.empty_cache()
        return {"loaded": True, "source": wm_actor_name, "mode": "memory"}

    #! wm added: may be deprecated
    def _build_world_model_dataloader(self, global_steps: int = 0):
        """
        Build dataloader for world model training for actor-side trainer.
        Mirrors WorldModelTrainer._build_world_model_dataloader but uses the actor's
        rollout_base_dir and world_model_mapping["wm_args"].
        Returns: train_dataloader, None
        """
        try:
            wm_map = getattr(self, "world_model_mapping", None)
            if wm_map is None:
                raise RuntimeError(
                    "_build_world_model_dataloader: world_model_mapping not found"
                )

            wm_args = wm_map.get("wm_args", None)
            if wm_args is None:
                raise RuntimeError(
                    "_build_world_model_dataloader: wm_args missing in world_model_mapping"
                )

            train_shards_pattern = os.path.join(
                self.rollout_base_dir, f"train/global_steps_{global_steps}_rank_*/*.tar"
            )

            # DatasetLiberoOnlineV2 and prepare_dataloader are expected to exist in repo
            train_dataset = DatasetLiberoOnlineV2(shards_pattern=train_shards_pattern)

            # choose process_group: actor may be in distributed setting
            try:
                process_group = get_data_parallel_group()
            except Exception:
                process_group = None

            train_dataloader_args = dict(
                dataset=train_dataset,
                batch_size=getattr(wm_args, "train_batch_size", 1),
                num_workers=getattr(wm_args, "num_workers", 0),
                seed=getattr(wm_args, "seed", 1024),
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                process_group=process_group,
                prefetch_factor=getattr(wm_args, "prefetch_factor", None),
                persistent_workers=False,
                cache_pin_memory=False,
            )

            train_dataloader, _ = prepare_dataloader(
                bucket_config=wm_args.get("bucket_config", None),
                num_bucket_build_workers=1,
                **train_dataloader_args,
            )
            return train_dataloader, None
        except Exception as e:
            print(f"[_build_world_model_dataloader] ERROR: {e}")
            raise

    #! wm added: may be deprecated
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_world_model(self, global_steps: int = 0, steps: int = None):
        """
        Actor-side WM training that is compatible with WorldModelTrainer.update_world_model.

        - If this actor is not the designated trainer (world_model_mapping['is_trainer'] == False),
        participates in barrier and returns {}.
        - Builds a fresh dataloader per-call, runs up to `steps` (or wm_args.training_steps_per_epoch),
        accumulates loss, does best-effort cleanup of shard dirs, and returns
        {"wm_loss": avg_loss, "steps_done": n_steps_done, "cleanup_tmp_data": {...}} on success.
        """
        # ensure required mapping
        wm_map = getattr(self, "world_model_mapping", None)
        if wm_map is None:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return {}

        # only trainer rank should execute; others sync and return
        if not bool(wm_map.get("is_trainer", True)):
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return {}

        try:
            wm_args = wm_map.get("wm_args", None)
            world_model: CtrlWorld = wm_map.get("world_model", None)
            device = wm_map.get(
                "device",
                (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                ),
            )
            dtype = wm_map.get("dtype", torch.float32)

            if world_model is None or wm_args is None:
                print("[update_world_model] missing world_model or wm_args in mapping")
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                return {}

            # ensure model on desired device/dtype and in train mode
            world_model.to(device=device, dtype=dtype)
            world_model.train()

            # determine steps to run this call
            requested_steps = (
                int(getattr(wm_args, "training_steps_per_epoch", 100))
                if steps is None
                else int(steps)
            )
            steps_to_run = max(1, requested_steps)

            # prepare optimizer in mapping if absent
            wm_optimizer = wm_map.get("wm_optimizer", None)
            if wm_optimizer is None:
                optim_cfg = (
                    getattr(wm_args, "optim", None)
                    or getattr(self.config.world_model, "optim", None)
                    or {}
                )
                lr = float(optim_cfg.get("lr", getattr(wm_args, "learning_rate", 1e-4)))
                weight_decay = float(optim_cfg.get("weight_decay", 1e-2))
                wm_optimizer = torch.optim.AdamW(
                    world_model.parameters(), lr=lr, weight_decay=weight_decay
                )
                wm_map["wm_optimizer"] = wm_optimizer

            use_amp = bool(
                getattr(wm_args, "use_amp", False)
                or self.config.world_model.get("use_amp", False)
            )
            scaler = GradScaler() if use_amp else None

            # build fresh dataloader for this global_steps
            try:
                train_dataloader, _ = self._build_world_model_dataloader(global_steps)
            except Exception as e:
                print(f"[update_world_model] failed build dataloader: {e}")
                return {}

            sum_loss = 0.0
            n_steps_done = 0

            # iterate safely
            for step_idx, batch in enumerate(train_dataloader):
                if step_idx >= steps_to_run:
                    break

                # move tensors to device IN-PLACE (to free CPU memory asap)
                for k, v in list(batch.items()):
                    if torch.is_tensor(v):
                        batch[k] = v.to(device, non_blocking=True)

                # forward/backward
                try:
                    if scaler is not None:
                        with autocast(device_type=device.type, dtype=torch.float16):
                            loss_dict, _ = world_model(batch)
                            loss_noise = loss_dict.get("loss_noise", None)
                            if loss_noise is None:
                                # fallback: take first value
                                loss_noise = list(loss_dict.values())[0]
                            loss_reward = loss_dict.get(
                                "loss_reward", torch.zeros_like(loss_noise)
                            )
                            loss_sf = loss_dict.get(
                                "loss_self_forcing", torch.tensor(0.0, device=device)
                            )
                            sf_weight = getattr(wm_args, "self_forcing_weight", 1.0)
                            loss = loss_noise + loss_reward + sf_weight * loss_sf
                        scaler.scale(loss).backward()
                        scaler.step(wm_optimizer)
                        scaler.update()
                    else:
                        loss_dict, _ = world_model(batch)
                        loss_noise = loss_dict.get("loss_noise", None)
                        if loss_noise is None:
                            loss_noise = list(loss_dict.values())[0]
                        loss_reward = loss_dict.get(
                            "loss_reward", torch.zeros_like(loss_noise)
                        )
                        loss_sf = loss_dict.get(
                            "loss_self_forcing", torch.tensor(0.0, device=device)
                        )
                        sf_weight = getattr(wm_args, "self_forcing_weight", 1.0)
                        loss = loss_noise + loss_reward + sf_weight * loss_sf
                        loss.backward()
                        wm_optimizer.step()
                        wm_optimizer.zero_grad(set_to_none=True)

                    # safe scalar extraction
                    try:
                        scalar = float(loss.detach().cpu().item())
                    except Exception:
                        try:
                            scalar = float(loss)
                        except Exception:
                            scalar = 0.0
                    sum_loss += scalar
                    n_steps_done += 1

                finally:
                    # cleanup references to free memory immediately
                    try:
                        if "loss" in locals():
                            del loss
                        if "loss_dict" in locals():
                            del loss_dict
                    except Exception:
                        pass

                    # clear large batch tensors
                    try:
                        for k in list(batch.keys()):
                            batch[k] = None
                        del batch
                    except Exception:
                        pass

                # periodic cleanup
                if (step_idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            # teardown dataloader/dataset best-effort
            try:
                ds = getattr(train_dataloader, "dataset", None)
                if ds is not None:
                    for fn in ("shutdown", "close", "shutdown_worker", "stop"):
                        if hasattr(ds, fn):
                            try:
                                getattr(ds, fn)()
                            except Exception:
                                pass
            except Exception:
                pass

            try:
                del train_dataloader
            except Exception:
                pass

            gc.collect()
            torch.cuda.empty_cache()

            if n_steps_done == 0:
                world_model.eval()
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                return {}

            avg_loss = sum_loss / max(1, n_steps_done)

            # attempt to cleanup shards (best-effort) if we did some training
            cleanup_info = None
            try:
                pattern = os.path.join(
                    self.rollout_base_dir, f"train/global_steps_{global_steps}_rank_*"
                )
                shard_dirs = glob.glob(pattern)
                removed = []
                failed = []
                if len(shard_dirs) > 0:
                    for d in shard_dirs:
                        try:
                            shutil.rmtree(d)
                            removed.append(d)
                        except Exception as e:
                            failed.append((d, str(e)))
                    cleanup_info = {
                        "cleaned": True,
                        "removed": removed,
                        "failed": failed,
                    }
                else:
                    cleanup_info = {"cleaned": False, "reason": "no matching shards"}
            except Exception as e:
                cleanup_info = {"cleaned": False, "error": str(e)}

            # set model back to eval mode (rollout expects eval)
            world_model.eval()

            # store optimizer back to mapping
            wm_map["wm_optimizer"] = wm_optimizer

            # distributed sync
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            return {
                "wm_loss": float(avg_loss),
                "steps_done": int(n_steps_done),
                "cleanup_tmp_data": cleanup_info,
            }

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[update_world_model] EXCEPTION: {e}\n{tb}")
            try:
                if world_model is not None:
                    world_model.eval()
            except Exception:
                pass
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return {}

    #! wm added: may be deprecated
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_world_model_mapping(
        self, actor_local_path: str, save_for_resume: bool = False
    ):
        """
        Save world_model state_dict to actor_local_path/world_model.pth (CPU tensors).
        This mirrors WorldModelTrainer.save_world_model but uses actor-side mapping.
        Only the trainer rank will actually write the file; others participate in barrier.
        Returns: {"saved": True, "path": save_path} or {"saved": False, "error": ...}
        """
        try:
            os.makedirs(actor_local_path, exist_ok=True)
            wm_map = getattr(self, "world_model_mapping", None)
            if wm_map is None:
                return {"saved": False, "error": "no world_model_mapping"}

            is_trainer = bool(wm_map.get("is_trainer", True))
            if not is_trainer:
                # non-trainers wait for the trainer to finish saving
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                return {"saved": False, "reason": "not trainer"}

            world_model = wm_map.get("world_model", None)
            if world_model is None:
                return {"saved": False, "error": "no world_model"}

            # if accelerator present in mapping, try to unwrap
            accelerator = wm_map.get("accelerator", None)
            try:
                model_to_save = (
                    accelerator.unwrap_model(world_model)
                    if accelerator is not None
                    else world_model
                )
            except Exception:
                model_to_save = world_model

            # move to cpu and save state_dict
            try:
                state_dict = materialize_cpu_state_dict(model_to_save)
                save_path = os.path.join(actor_local_path, "world_model.pth")
                save_info = safe_save_state_dict_file(state_dict, save_path)
                gc.collect()
                torch.cuda.empty_cache()
                # barrier to allow others to load
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                return {
                    "saved": True,
                    "path": save_path,
                    "estimated_bytes": save_info.get("estimated_bytes"),
                }
            except Exception as e:
                err = repr(e)
                print(f"[save_world_model_mapping] save error: {err}")
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                return {"saved": False, "error": err}
        except Exception as e:
            err = repr(e)
            print(f"[save_world_model_mapping] EXCEPTION: {err}")
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return {"saved": False, "error": err}


@ray.remote(num_gpus=1, concurrency_groups={"wm": 1})
class WorldModelTrainer:
    """
    Single-instance Ray actor that holds and trains the World Model.
    - 独立于 WorkerDict / WorkerGroup，避免广播导致的多份内存占用。
    - 提供 init_model(), update_world_model(global_steps) 和 save_world_model(local_path) 接口。
    """

    def __init__(
        self,
        wm_cfg_path: str,
        rollout_base_dir: str,
        rank: int = 0,
        wm_overrides: dict = None,
        options: dict = None,
    ):
        # minimal imports inside actor
        from omegaconf import OmegaConf

        self.wm_cfg_path = wm_cfg_path
        self.rollout_base_dir = os.path.abspath(rollout_base_dir)
        os.makedirs(self.rollout_base_dir, exist_ok=True)
        self.rank = rank  # logical rank (for logging); actor is single-process
        self.wm_overrides = convert_to_regular_types(wm_overrides or {})
        # placeholders
        self.wm_args = None
        self.world_model = None
        self.accelerator = None
        self.wm_optimizer = None
        self.device = None
        self.dtype = None
        self._inited = False
        self._wm_updating = False
        self.metric_models = None

    def _read_config(self, cfg_path: str):
        # reuse RobWMActorRolloutRefWorker._read_config if available
        try:
            cfg = RobWMActorRolloutRefWorker._read_config(cfg_path)
            return cfg
        except Exception:
            # fallback: small inline loader (same behaviour)
            cfg_path = Path(cfg_path)
            if not cfg_path.exists():
                raise FileNotFoundError(
                    f"The configuration file does not exist: {cfg_path}"
                )
            original_path = sys.path.copy()
            try:
                cfg_dir = cfg_path.parent.absolute()
                cfg_name = cfg_path.stem
                if str(cfg_dir) not in sys.path:
                    sys.path.insert(0, str(cfg_dir))
                config_module = importlib.import_module(cfg_name)
                if not hasattr(config_module, "wm_args"):
                    raise AttributeError(
                        f"The 'wm_args' class was not found in {cfg_path}"
                    )
                return config_module.wm_args()
            finally:
                sys.path = original_path

    def init_model(self):
        """
        Initialize world model and optimizer, wrapped by accelerate.Accelerator.
        This must be called once after actor creation.
        """
        # local imports to keep serialization small
        from accelerate import Accelerator

        set_seed(1024)

        # read config
        self.wm_args = self._read_config(self.wm_cfg_path)
        self.wm_args = apply_world_model_overrides(self.wm_args, self.wm_overrides)

        # dtype handling (be permissive)
        if not hasattr(self.wm_args, "_dtype_obj"):
            self.dtype = torch.bfloat16
        elif isinstance(self.wm_args.dtype_obj, torch.dtype):
            self.dtype = self.wm_args.dtype_obj
        else:
            # maybe string like "bfloat16"
            try:
                if str(self.wm_args.dtype_obj).lower().startswith("bfloat"):
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float32
            except Exception:
                self.dtype = torch.bfloat16

        # Use Accelerator for single-device BF16 training (keeps code consistent)
        mixed = "bf16" if self.dtype == torch.bfloat16 else "no"
        self.accelerator = Accelerator(mixed_precision=mixed)

        # build world model
        self.world_model = CtrlWorld(self.wm_args)
        ckpt_path = resolve_ctrl_world_ckpt_path(self.wm_args)
        if ckpt_path is not None:
            # load on cpu then to device
            state_dict = load_trusted_state_dict(ckpt_path, map_location="cpu")
            try:
                self.world_model.load_state_dict(state_dict, strict=True)
                print(f"[WM Trainer] Loaded checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"[WM Trainer] Warning: loading checkpoint failed: {e}")

        # optimizer
        self.wm_optimizer = torch.optim.AdamW(
            self.world_model.parameters(), lr=self.wm_args.learning_rate
        )

        # prepare (accelerator will move model/optimizer to device)
        self.world_model, self.wm_optimizer = self.accelerator.prepare(
            self.world_model, self.wm_optimizer
        )
        self.world_model.train()
        self.device = self.accelerator.device

        # mark ready
        self._inited = True
        # free CUDA caches
        torch.cuda.empty_cache()
        gc.collect()
        return {"inited": True, "device": str(self.device)}

    def _build_world_model_dataloader(
        self,
        global_steps: int = 0,
        type_: str = "train",
        *,
        base_dir: str = None,
        split_override: str = None,
        batch_size: int = None,
        max_windows_per_episode: int = None,
        window_selection: str = "sliding",
        shuffle: bool = True,
    ):
        """
        Build dataloader. Use the same DatasetLiberoOnlineV2 and prepare_dataloader as existing worker.
        Note: here we DO NOT use distributed process group (single actor), so pass process_group=None.
        """
        wm_args = self.wm_args
        requested_type = str(type_)
        base_dir = os.path.abspath(base_dir) if base_dir else self.rollout_base_dir
        fallback_used = False
        if split_override is not None:
            split_candidates = [str(split_override)]
        else:
            split_candidates = {
                "train": ["train_real", "train"],
                "train_real": ["train_real", "train"],
                "imag_train": ["imag_train"],
                "eval": ["eval_real", "eval", "train_real", "train"],
                "eval_real": ["eval_real", "eval", "train_real", "train"],
            }.get(requested_type, [requested_type])

        train_shards_pattern = ""
        shard_paths = []
        actual_type = requested_type
        for candidate in split_candidates:
            candidate_pattern = os.path.join(
                base_dir,
                f"{candidate}/global_steps_{global_steps}_rank_*/*.tar",
            )
            candidate_paths = sorted(glob.glob(candidate_pattern))
            if len(candidate_paths) > 0:
                train_shards_pattern = candidate_pattern
                shard_paths = candidate_paths
                actual_type = candidate
                fallback_used = candidate != requested_type
                if fallback_used:
                    print(
                        f"[WM Data] Fallback from split '{requested_type}' to '{candidate}' for global_steps={global_steps}."
                    )
                break

        type_ = actual_type

        self._last_wm_dataloader_meta = {
            "requested_type": requested_type,
            "actual_type": type_,
            "global_steps": int(global_steps),
            "shards_pattern": train_shards_pattern,
            "num_shards": int(len(shard_paths)),
            "fallback_used": bool(fallback_used),
            "base_dir": base_dir,
        }

        if len(shard_paths) == 0:
            raise FileNotFoundError(
                f"[WM Dataset] No shards found for split '{requested_type}' at global_steps={global_steps}."
            )

        train_dataset = DatasetLiberoOnlineV2(
            shards_pattern=train_shards_pattern,
            Ta=int(getattr(wm_args, "num_frames", 8)),
            To=int(getattr(wm_args, "num_history", 8)),
            max_windows_per_episode=max_windows_per_episode,
            window_selection=window_selection,
        )
        train_dataloader_args = dict(
            dataset=train_dataset,
            batch_size=resolve_wm_batch_size(
                wm_args,
                requested_type=requested_type,
                actual_type=type_,
                explicit_batch_size=batch_size,
            ),
            num_workers=0,
            seed=wm_args.get("seed", 1024),
            shuffle=bool(shuffle),
            drop_last=False if split_override is not None else True,
            pin_memory=False,
            process_group=None,  # single-actor: no DP
            prefetch_factor=None,
            persistent_workers=False,
            cache_pin_memory=False,
        )
        # train & eval dataloader
        train_dataloader, _ = prepare_dataloader(
            bucket_config=wm_args.get("bucket_config", None),
            num_bucket_build_workers=1,
            **train_dataloader_args,
        )
        return train_dataloader

    def _cleanup_data_shards(self, global_steps: int, type_: str = "train"):
        """
        Best-effort cleanup of data shards for a given global_steps.
        Safe to call AFTER dataloader / dataset are fully released.
        """
        pattern = os.path.join(
            self.rollout_base_dir, f"{type_}/global_steps_{global_steps}_rank_*"
        )
        shard_dirs = glob.glob(pattern)

        if len(shard_dirs) == 0:
            return {"cleaned": False, "reason": "no matching shards"}

        removed = []
        failed = []
        for d in shard_dirs:
            try:
                shutil.rmtree(d)
                removed.append(d)
            except Exception as e:
                failed.append((d, str(e)))

        return {
            "cleaned": True,
            "removed": removed,
            "failed": failed,
        }

    @staticmethod
    def _extract_rollout_step_from_dir(path: str):
        name = os.path.basename(os.path.normpath(path))
        prefix = "global_steps_"
        rank_token = "_rank_"
        if (not name.startswith(prefix)) or (rank_token not in name):
            return None
        try:
            return int(name[len(prefix) :].split(rank_token, 1)[0])
        except Exception:
            return None

    @staticmethod
    def _estimate_dir_size_bytes(path: str):
        total_bytes = 0
        for root, _, files in os.walk(path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    total_bytes += os.path.getsize(file_path)
                except OSError:
                    pass
        return total_bytes

    @ray.method(num_returns=1, concurrency_group="wm")
    def cleanup_stale_rollout_data(
        self,
        keep_from_global_steps: int,
        split_types: List[str] = None,
    ):
        """
        Remove stale rollout shard directories strictly older than keep_from_global_steps.

        This is a conservative best-effort GC for tmp rollout data under the current
        experiment's rollout_base_dir. Shared WM eval roots are not touched because
        they live outside rollout_base_dir.
        """
        keep_from_global_steps = int(keep_from_global_steps)
        managed_splits = [
            str(split_name)
            for split_name in (
                split_types
                or ["train_real", "eval_real", "imag_train", "train", "eval"]
            )
        ]

        removed = []
        failed = []
        scanned_dirs = 0
        freed_bytes = 0

        for split_name in managed_splits:
            pattern = os.path.join(
                self.rollout_base_dir,
                f"{split_name}/global_steps_*_rank_*",
            )
            for shard_dir in glob.glob(pattern):
                step = self._extract_rollout_step_from_dir(shard_dir)
                if step is None:
                    continue
                scanned_dirs += 1
                if step >= keep_from_global_steps:
                    continue
                try:
                    freed_bytes += self._estimate_dir_size_bytes(shard_dir)
                    shutil.rmtree(shard_dir)
                    removed.append(shard_dir)
                except Exception as exc:
                    failed.append((shard_dir, str(exc)))

        return {
            "cleaned": bool(removed),
            "keep_from_global_steps": keep_from_global_steps,
            "scanned_dirs": int(scanned_dirs),
            "removed_dirs": int(len(removed)),
            "failed_dirs": int(len(failed)),
            "freed_bytes": int(freed_bytes),
            "removed": removed,
            "failed": failed,
        }

    def is_updating(self):
        return bool(self._wm_updating)

    @ray.method(num_returns=1, concurrency_group="wm")
    def update_world_model_old(self, global_steps: int = 0, steps: int = None):
        """
        Robust training loop for WM inside single actor.
        Key safety: build dataloader per-call, limit steps per call, explicit cleanup.
        """
        if not self._inited:
            raise RuntimeError(
                "WorldModelTrainer not initialized. Call init_model() first."
            )
        if self._wm_updating:
            return {"skipped": True}
        self.world_model.train()
        self._wm_updating = True

        try:
            wm_args = self.wm_args
            # cap steps per call to avoid long-lived accumulation
            requested_steps = (
                int(getattr(wm_args, "training_steps_per_epoch", 100))
                if steps is None
                else int(steps)
            )
            # max_steps_per_call = int(getattr(wm_args, "max_steps_per_update", 200))
            # steps = min(requested_steps, max_steps_per_call)
            steps = requested_steps

            # build dataloader locally (fresh each call)
            train_dataloader = self._build_world_model_dataloader(
                global_steps, type_="train"
            )
            # iterate safely (avoid manual next() + huge steps)
            sum_loss = 0.0
            n_steps_done = 0

            for step_idx, batch in enumerate(train_dataloader):
                if step_idx >= steps:
                    break
                # Move tensors to device IN-PLACE to avoid keeping two copies too long.
                # Replace batch entries with moved tensors so old cpu tensors can be freed.
                for k, v in list(batch.items()):
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device, non_blocking=True)
                # forward/backward
                try:
                    with self.accelerator.accumulate(self.world_model):
                        loss_dict, _ = self.world_model(batch)

                        loss_noise = loss_dict["loss_noise"]
                        loss_reward = loss_dict.get(
                            "loss_reward", torch.zeros_like(loss_noise)
                        )
                        loss_sf = loss_dict.get(
                            "loss_self_forcing", torch.tensor(0.0, device=self.device)
                        )
                        sf_weight = getattr(wm_args, "self_forcing_weight", 1.0)
                        loss = loss_noise + loss_reward + sf_weight * loss_sf

                        # backward + step
                        self.accelerator.backward(loss)
                        self.wm_optimizer.step()
                        self.wm_optimizer.zero_grad(set_to_none=True)

                    # retrieve scalar and immediately detach
                    scalar = float(loss.detach().cpu().item())
                    sum_loss += scalar
                    n_steps_done += 1

                finally:
                    # CRITICAL: cut references to large tensors / graphs
                    # loss_dict may contain tensors referencing graph; delete and overwrite.
                    try:
                        # try to free inner tensors explicitly
                        for key in ("loss_noise", "loss_reward", "loss_self_forcing"):
                            if key in locals():
                                locals().pop(key, None)
                    except Exception:
                        pass

                    # delete commonly referenced objects
                    if "loss" in locals():
                        del loss
                    if "loss_dict" in locals():
                        del loss_dict
                    # clear the batch content to free CPU memory
                    try:
                        for k in list(batch.keys()):
                            batch[k] = None
                        del batch
                    except Exception:
                        pass

                # periodic cleanup to help Python release memory
                if (step_idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            # end loop
            # best-effort teardown of dataloader/dataset
            try:
                ds = getattr(train_dataloader, "dataset", None)
                # call dataset shutdown/close if present (webdataset may expose such method)
                if ds is not None:
                    for fn in ("shutdown", "close", "shutdown_worker", "stop"):
                        if hasattr(ds, fn):
                            try:
                                getattr(ds, fn)()
                            except Exception:
                                pass
            except Exception:
                pass

            # final cleanup
            try:
                del train_dataloader
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

            avg_loss = sum_loss / max(1, n_steps_done)

            # OPTIONAL: cleanup training shards for this global_steps
            cleanup_info = None
            if n_steps_done > 0:
                try:
                    print("[wm trainer] Clean up tmp training data")
                    cleanup_info = self._cleanup_data_shards(
                        global_steps, type_="train"
                    )
                except Exception as e:
                    cleanup_info = {"cleaned": False, "error": str(e)}

            return {
                "wm_loss": avg_loss,
                "steps_done": n_steps_done,
                "skipped": False,
                "cleanup_tmp_data": cleanup_info,
            }

        finally:
            # always reset updating flag (even on exception)
            self._wm_updating = False

    @ray.method(num_returns=1, concurrency_group="wm")
    def update_world_model(self, global_steps: int = 0, steps: int = None):
        """
        Robust training loop for WM inside single actor.

        Returns:
        - wm_loss: total training loss used to optimize WM
                = loss_noise + loss_reward + sf_weight * loss_self_forcing
        - wm_ratio_signal: dedicated proxy for ratio scheduling (loss_noise only)
        - steps_done, skipped, cleanup_tmp_data

        Key design:
        - training objective and ratio-scheduling signal are decoupled
        - ratio signal uses visual prediction main error (loss_noise), which is
        a cleaner proxy for imagined rollout fidelity than the full composite loss
        """
        if not self._inited:
            raise RuntimeError(
                "WorldModelTrainer not initialized. Call init_model() first."
            )
        if self._wm_updating:
            return {"skipped": True}

        self.world_model.train()
        self._wm_updating = True

        try:
            wm_args = self.wm_args

            requested_steps = (
                int(getattr(wm_args, "training_steps_per_epoch", 100))
                if steps is None
                else int(steps)
            )
            steps = requested_steps

            # build dataloader locally (fresh each call)
            try:
                train_dataloader = self._build_world_model_dataloader(
                    global_steps, type_="train_real"
                )
            except FileNotFoundError as exc:
                dl_meta = getattr(self, "_last_wm_dataloader_meta", {})
                return {
                    "wm_loss": None,
                    "wm_ratio_signal": None,
                    "steps_done": 0,
                    "skipped": True,
                    "skipped_no_data": True,
                    "skip_reason": str(exc),
                    "cleanup_tmp_data": None,
                    "data_meta": dl_meta if isinstance(dl_meta, dict) else None,
                }

            dl_meta = getattr(self, "_last_wm_dataloader_meta", {})
            cleanup_type = dl_meta.get("actual_type", "train_real")

            sum_loss = 0.0
            sum_ratio_signal = 0.0
            n_steps_done = 0

            for step_idx, batch in enumerate(train_dataloader):
                if step_idx >= steps:
                    break

                # move tensors in-place
                for k, v in list(batch.items()):
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device, non_blocking=True)

                try:
                    with self.accelerator.accumulate(self.world_model):
                        loss_dict, _ = self.world_model(batch)

                        # main visual prediction error
                        loss_noise = loss_dict["loss_noise"]

                        # optional auxiliary terms
                        loss_reward = loss_dict.get(
                            "loss_reward", torch.zeros_like(loss_noise)
                        )
                        loss_sf = loss_dict.get(
                            "loss_self_forcing", torch.tensor(0.0, device=self.device)
                        )
                        sf_weight = getattr(wm_args, "self_forcing_weight", 1.0)

                        # full training objective
                        loss = loss_noise + loss_reward + sf_weight * loss_sf

                        self.accelerator.backward(loss)
                        self.wm_optimizer.step()
                        self.wm_optimizer.zero_grad(set_to_none=True)

                    total_scalar = float(loss.detach().cpu().item())
                    ratio_scalar = float(loss_noise.detach().cpu().item())

                    sum_loss += total_scalar
                    sum_ratio_signal += ratio_scalar
                    n_steps_done += 1

                finally:
                    # aggressively drop references
                    try:
                        if "loss" in locals():
                            del loss
                        if "loss_dict" in locals():
                            del loss_dict
                        if "loss_noise" in locals():
                            del loss_noise
                        if "loss_reward" in locals():
                            del loss_reward
                        if "loss_sf" in locals():
                            del loss_sf
                    except Exception:
                        pass

                    try:
                        for k in list(batch.keys()):
                            batch[k] = None
                        del batch
                    except Exception:
                        pass

                if (step_idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            # best-effort teardown
            try:
                ds = getattr(train_dataloader, "dataset", None)
                if ds is not None:
                    for fn in ("shutdown", "close", "shutdown_worker", "stop"):
                        if hasattr(ds, fn):
                            try:
                                getattr(ds, fn)()
                            except Exception:
                                pass
            except Exception:
                pass

            try:
                del train_dataloader
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

            avg_loss = sum_loss / max(1, n_steps_done)
            avg_ratio_signal = sum_ratio_signal / max(1, n_steps_done)

            cleanup_info = None
            if n_steps_done > 0:
                try:
                    print("[wm trainer] Clean up tmp training data")
                    cleanup_info = self._cleanup_data_shards(
                        global_steps, type_=cleanup_type
                    )
                except Exception as e:
                    cleanup_info = {"cleaned": False, "error": str(e)}

            return {
                "wm_loss": float(avg_loss),
                "wm_ratio_signal": float(avg_ratio_signal),
                "steps_done": int(n_steps_done),
                "skipped": False,
                "cleanup_tmp_data": cleanup_info,
                "data_meta": dl_meta if isinstance(dl_meta, dict) else None,
            }

        finally:
            self._wm_updating = False

    def save_world_model(self, local_path: str):
        """
        Save world model state_dict (CPU) to local_path (directory). Returns {'saved': True, 'path': ...}
        After saving, user can call actor_rollout_wg.load_world_model_mapping(local_path) to load across rollout actors.
        """
        os.makedirs(local_path, exist_ok=True)
        # unwrap model if accelerator used
        try:
            model_to_save = self.accelerator.unwrap_model(self.world_model)
        except Exception:
            model_to_save = self.world_model

        # move to cpu and save full state_dict
        state_dict = materialize_cpu_state_dict(model_to_save)
        save_path = os.path.join(local_path, "world_model.pth")
        save_info = safe_save_state_dict_file(state_dict, save_path)
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "saved": True,
            "path": save_path,
            "estimated_bytes": save_info.get("estimated_bytes"),
        }

    def export_world_model_state_dict(self):
        """Export a CPU state_dict for rollout-worker sync without touching disk."""
        try:
            model_to_export = self.accelerator.unwrap_model(self.world_model)
        except Exception:
            model_to_export = self.world_model
        return materialize_cpu_state_dict(model_to_export)

    # todo: 当有训到一半的模型时启用
    def load_checkpoint(self, ckpt_path: str):
        """Load checkpoint from ckpt_path into the trainer's model (map to cpu then to device)."""
        assert os.path.isfile(ckpt_path), f"Missing checkpoint: {ckpt_path}"
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # unwrap model if necessary
        try:
            model_to_load = self.accelerator.unwrap_model(self.world_model)
        except Exception:
            model_to_load = self.world_model
        model_to_load.load_state_dict(state_dict, strict=True)
        # if using accelerator, rewrap is not necessary; ensure on device/dtype
        torch.cuda.empty_cache()
        gc.collect()
        return {"loaded": True, "path": ckpt_path}

    # For evaluation
    def _preprocess_img(self, img: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = img.shape
        img = img.view(B * T, C, H, W)  # [B*T, C, H, W]
        img: torch.Tensor = img.float() / 255.0 * 2 - 1  # [B*T, C, H, W]
        # resize H * W to h * w
        h, w = self.wm_args.img_resizes  # [192, 320]
        img = torch.nn.functional.interpolate(
            img,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )  # [B*T, C, h, w]
        img = img.view(B, T, C, h, w)
        return img

    @staticmethod
    def _first_event_step(event_mask: torch.Tensor, default_step: int) -> torch.Tensor:
        has_event = event_mask.any(dim=1)
        first_idx = event_mask.float().argmax(dim=1).to(torch.float32) + 1.0
        default_tensor = torch.full_like(first_idx, float(default_step))
        return torch.where(has_event, first_idx, default_tensor)

    def _compute_termination_metrics(
        self,
        pred_rewards: torch.Tensor,
        true_rewards: torch.Tensor,
        threshold: float,
    ) -> Dict[str, float]:
        horizon = int(pred_rewards.shape[1])
        pred_event = pred_rewards > threshold
        true_event = true_rewards > 0.5

        pred_step = self._first_event_step(pred_event, default_step=horizon + 1)
        true_step = self._first_event_step(true_event, default_step=horizon + 1)
        abs_error = (pred_step - true_step).abs()

        true_has = true_event.any(dim=1)
        pred_has = pred_event.any(dim=1)
        negative_mask = ~true_has
        positive_mask = true_has

        metrics = {
            "done_step_mae": float(abs_error.mean().item()),
            "done_step_median_ae": float(abs_error.median().item()),
            "done_within_1step": float((abs_error <= 1.0).float().mean().item()),
            "done_within_2step": float((abs_error <= 2.0).float().mean().item()),
            "termination_event_accuracy": float(
                (pred_has == true_has).float().mean().item()
            ),
            "pred_terminate_rate": float(pred_has.float().mean().item()),
            "true_terminate_rate": float(true_has.float().mean().item()),
        }

        if bool(positive_mask.any().item()):
            metrics["done_step_mae_on_gt_done"] = float(
                abs_error[positive_mask].mean().item()
            )
            metrics["termination_false_negative_rate"] = float(
                ((~pred_has) & positive_mask).float().sum().item()
                / max(1.0, positive_mask.float().sum().item())
            )
        else:
            metrics["done_step_mae_on_gt_done"] = 0.0
            metrics["termination_false_negative_rate"] = 0.0

        if bool(negative_mask.any().item()):
            metrics["termination_false_positive_rate"] = float(
                (pred_has & negative_mask).float().sum().item()
                / max(1.0, negative_mask.float().sum().item())
            )
        else:
            metrics["termination_false_positive_rate"] = 0.0

        return metrics

    @ray.method(num_returns=1, concurrency_group="wm")
    @torch.no_grad()
    def evaluate_world_model(self, global_steps: int = 0, max_batches: int = 10):
        device = self.device
        results = {"global_steps": global_steps}
        try:
            self.world_model.eval()

            wm_args = self.wm_args
            fixed_eval_root = str(getattr(wm_args, "fixed_eval_root", "") or "").strip()
            fixed_eval_enabled = bool(
                getattr(wm_args, "fixed_eval_enabled", False)
            ) and bool(fixed_eval_root)

            eval_jobs = []
            if fixed_eval_enabled:
                fixed_eval_global_steps = int(
                    getattr(wm_args, "fixed_eval_global_steps", 0)
                )
                window_selection = str(
                    getattr(wm_args, "fixed_eval_window_selection", "uniform")
                )
                mini_batch_size = int(getattr(wm_args, "fixed_eval_mini_batch_size", 4))
                mini_samples = max(
                    1, int(getattr(wm_args, "fixed_eval_mini_samples", 40))
                )
                mini_batches = max(
                    1, (mini_samples + mini_batch_size - 1) // mini_batch_size
                )
                eval_jobs.append(
                    {
                        "prefix": "",
                        "requested_type": "fixed_eval_mini",
                        "split_override": str(
                            getattr(
                                wm_args, "fixed_eval_mini_split", "wm_eval_fixed_mini"
                            )
                        ),
                        "base_dir": fixed_eval_root,
                        "global_steps": fixed_eval_global_steps,
                        "batch_size": mini_batch_size,
                        "max_batches": mini_batches,
                        "max_windows_per_episode": int(
                            getattr(wm_args, "fixed_eval_mini_windows_per_episode", 2)
                        ),
                        "window_selection": window_selection,
                        "shuffle": False,
                    }
                )

                full_interval = max(
                    1, int(getattr(wm_args, "fixed_eval_full_interval", 10))
                )
                if (global_steps + 1) % full_interval == 0:
                    full_batch_size = int(
                        getattr(wm_args, "fixed_eval_full_batch_size", 4)
                    )
                    full_samples = max(
                        1, int(getattr(wm_args, "fixed_eval_full_samples", 320))
                    )
                    full_batches = max(
                        1, (full_samples + full_batch_size - 1) // full_batch_size
                    )
                    eval_jobs.append(
                        {
                            "prefix": "full/",
                            "requested_type": "fixed_eval_full",
                            "split_override": str(
                                getattr(
                                    wm_args,
                                    "fixed_eval_full_split",
                                    "wm_eval_fixed_full",
                                )
                            ),
                            "base_dir": fixed_eval_root,
                            "global_steps": fixed_eval_global_steps,
                            "batch_size": full_batch_size,
                            "max_batches": full_batches,
                            "max_windows_per_episode": int(
                                getattr(
                                    wm_args, "fixed_eval_full_windows_per_episode", 4
                                )
                            ),
                            "window_selection": window_selection,
                            "shuffle": False,
                        }
                    )
            else:
                eval_jobs.append(
                    {
                        "prefix": "",
                        "requested_type": "eval_real",
                        "split_override": None,
                        "base_dir": None,
                        "global_steps": global_steps,
                        "batch_size": None,
                        "max_batches": max_batches,
                        "max_windows_per_episode": None,
                        "window_selection": "sliding",
                        "shuffle": True,
                    }
                )

            def run_eval_job(eval_job: Dict[str, Any]) -> Dict[str, Any]:
                job_results = {}
                try:
                    eval_dataloader = self._build_world_model_dataloader(
                        eval_job["global_steps"],
                        type_=eval_job["requested_type"],
                        base_dir=eval_job.get("base_dir", None),
                        split_override=eval_job.get("split_override", None),
                        batch_size=eval_job.get("batch_size", None),
                        max_windows_per_episode=eval_job.get(
                            "max_windows_per_episode", None
                        ),
                        window_selection=eval_job.get("window_selection", "sliding"),
                        shuffle=eval_job.get("shuffle", True),
                    )
                except FileNotFoundError as exc:
                    dl_meta = getattr(self, "_last_wm_dataloader_meta", None)
                    if isinstance(dl_meta, dict):
                        job_results["data_split_requested"] = dl_meta.get(
                            "requested_type", eval_job["requested_type"]
                        )
                        job_results["data_split_actual"] = dl_meta.get(
                            "actual_type", eval_job["requested_type"]
                        )
                        job_results["data_num_shards"] = dl_meta.get("num_shards", 0)
                        job_results["data_fallback_used"] = dl_meta.get(
                            "fallback_used", False
                        )
                        job_results["data_shards_pattern"] = dl_meta.get(
                            "shards_pattern", ""
                        )
                    job_results["skipped_no_data"] = 1.0
                    job_results["skip_reason"] = str(exc)
                    return job_results

                dl_meta = getattr(self, "_last_wm_dataloader_meta", None)
                if isinstance(dl_meta, dict):
                    job_results["data_split_requested"] = dl_meta.get(
                        "requested_type", eval_job["requested_type"]
                    )
                    job_results["data_split_actual"] = dl_meta.get(
                        "actual_type", eval_job["requested_type"]
                    )
                    job_results["data_num_shards"] = dl_meta.get("num_shards", 0)
                    job_results["data_fallback_used"] = dl_meta.get(
                        "fallback_used", False
                    )
                    job_results["data_shards_pattern"] = dl_meta.get(
                        "shards_pattern", ""
                    )

                model_inner = (
                    self.world_model.module
                    if hasattr(self.world_model, "module")
                    else self.world_model
                )
                pipeline: CtrlWorldDiffusionPipeline = model_inner.pipeline
                num_history = int(wm_args.get("num_history", 8))
                num_frames = int(wm_args.get("num_frames", 8))
                eval_num_inference_steps = int(
                    getattr(
                        wm_args,
                        "eval_num_inference_steps",
                        getattr(wm_args, "num_inference_steps", 50),
                    )
                )

                def to_0_1_tensor(x):
                    x = x.float()
                    if x.numel() == 0:
                        return x
                    x_min = float(x.detach().amin().item())
                    x_max = float(x.detach().amax().item())
                    if x_max > 1.5:
                        x = x / 255.0
                    elif x_min < 0.0:
                        x = (x + 1.0) / 2.0
                    return x.clamp(0.0, 1.0)

                def to_minus_1_1_tensor(x):
                    x = x.float()
                    if x.numel() == 0:
                        return x
                    x_min = float(x.detach().amin().item())
                    x_max = float(x.detach().amax().item())
                    if x_max > 1.5:
                        x = x / 255.0
                        x_min = float(x.detach().amin().item())
                    if x_min >= 0.0:
                        x = x * 2.0 - 1.0
                    return x.clamp(-1.0, 1.0)

                def maybe_sync_cuda():
                    if device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize(device)

                requested_video_metrics = ["psnr", "ssim", "lpips"]
                if bool(getattr(wm_args, "eval_enable_fid", True)):
                    requested_video_metrics.append("fid")
                if bool(getattr(wm_args, "eval_enable_fvd", True)):
                    requested_video_metrics.append("fvd")
                if bool(getattr(wm_args, "eval_enable_clips", True)):
                    requested_video_metrics.append("clips")
                core_metric_names = ["psnr", "ssim", "lpips"]

                print(f"[wm eval] pipeline inference ({eval_job['requested_type']})")
                extra_video_metric_names = [
                    name
                    for name in requested_video_metrics
                    if name not in core_metric_names
                ]
                rm_threshold = float(getattr(wm_args, "reward_threshold", 0.5))

                core_metric_sums = {name: 0.0 for name in core_metric_names}
                core_metric_weights = {name: 0.0 for name in core_metric_names}
                extra_pred_videos_cpu = []
                extra_gt_videos_cpu = []
                reward_pred_flat_chunks = []
                reward_true_flat_chunks = []
                term_abs_error_chunks = []
                term_pred_has_chunks = []
                term_true_has_chunks = []
                video_metric_error_logs = []
                reward_error_logs = []
                pipeline_error_logs = []
                processed_batches = 0
                eval_sample_count = 0

                for i, batch in enumerate(eval_dataloader):
                    if i >= int(eval_job["max_batches"]):
                        break

                    try:
                        imgs = self._preprocess_img(batch["img"].to(device))
                        actions = batch["action"].to(device)
                        rewards = batch["reward"].to(device)
                        texts = batch.get("text", [])
                    except Exception as e:
                        warnings.warn(f"Skipping bad batch {i}: {e}")
                        continue

                    B_total, T, C, H_, W_ = imgs.shape
                    eval_sample_count += int(B_total)

                    try:
                        imgs_flat = imgs.view(B_total * T, C, H_, W_)
                        latents_flat = encode_img_to_latent_on_gpu(
                            self.world_model, imgs_flat, device
                        )
                        latents = latents_flat.view(
                            B_total,
                            T,
                            latents_flat.shape[1],
                            latents_flat.shape[2],
                            latents_flat.shape[3],
                        )
                        actions_latent = model_inner.action_encoder(
                            actions,
                            texts,
                            model_inner.tokenizer,
                            model_inner.text_encoder,
                            wm_args.frame_level_cond,
                        )

                        his_latent = latents[:, :num_history]
                        current_latent = his_latent[:, -1]
                        pred_frames_list, pred_latents = (
                            CtrlWorldDiffusionPipeline.__call__(
                                pipeline,
                                image=current_latent,
                                text=actions_latent,
                                width=wm_args.width,
                                height=wm_args.height,
                                num_frames=num_frames,
                                history=(
                                    his_latent if his_latent is not None else None
                                ),
                                num_inference_steps=eval_num_inference_steps,
                                decode_chunk_size=num_frames,
                                max_guidance_scale=wm_args.guidance_scale,
                                fps=wm_args.fps,
                                motion_bucket_id=wm_args.motion_bucket_id,
                                mask=None,
                                output_type="frame",
                                return_dict=False,
                                frame_level_cond=wm_args.frame_level_cond,
                                his_cond_zero=wm_args.his_cond_zero,
                            )
                        )
                        maybe_sync_cuda()
                        pred_video = torch.stack(
                            [torch.from_numpy(f) for f in pred_frames_list], dim=0
                        )
                        pred_video = einops.rearrange(
                            pred_video, "b t h w c -> b t c h w"
                        ).to(device)
                    except Exception as pipeline_exc:
                        pipeline_error_logs.append(
                            f"batch_{i}: {type(pipeline_exc).__name__}: {pipeline_exc}"
                        )
                        warnings.warn(
                            f"WM eval batch failed during pipeline inference ({eval_job['requested_type']} batch {i}): {pipeline_exc}"
                        )
                        try:
                            del (
                                imgs_flat,
                                latents_flat,
                                latents,
                                pred_frames_list,
                                pred_latents,
                            )
                        except Exception:
                            pass
                        gc.collect()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        continue

                    processed_batches += 1

                    gt_video_decoded = to_0_1_tensor(imgs).to(device)
                    pred_video_decoded = to_0_1_tensor(pred_video).to(device)

                    gt_future = gt_video_decoded[:, num_history:, ...]
                    pred_future = pred_video_decoded
                    min_T = min(pred_future.shape[1], gt_future.shape[1])
                    if min_T <= 0:
                        video_metric_error_logs.append(
                            f"batch_{i}: pred/gt future video has zero overlap"
                        )
                        try:
                            del (
                                pred_frames_list,
                                pred_latents,
                                pred_video,
                                latents,
                                imgs_flat,
                            )
                        except Exception:
                            pass
                        gc.collect()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        continue

                    pred_future = pred_future[:, :min_T]
                    gt_future = gt_future[:, :min_T]
                    if pred_future.shape[2] != 3 or gt_future.shape[2] != 3:
                        video_metric_error_logs.append(
                            f"batch_{i}: pred/gt channels != 3 (likely latent vs rgb mismatch)"
                        )
                        try:
                            del (
                                pred_frames_list,
                                pred_latents,
                                pred_video,
                                latents,
                                imgs_flat,
                            )
                        except Exception:
                            pass
                        gc.collect()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        continue

                    frame_weight = float(pred_future.shape[0] * pred_future.shape[1])
                    video_metrics = compute_all_metrics(
                        pred_future,
                        gt_future,
                        device,
                        metrics_to_compute=core_metric_names,
                        strict=False,
                        synchronize_cuda=True,
                        enable_lpips_cpu_fallback=True,
                    )
                    for metric_name in core_metric_names:
                        if metric_name in video_metrics:
                            core_metric_sums[metric_name] += (
                                float(video_metrics[metric_name]) * frame_weight
                            )
                            core_metric_weights[metric_name] += frame_weight

                    metric_error_keys = [
                        key for key in video_metrics.keys() if key.endswith("_error")
                    ]
                    if metric_error_keys:
                        video_metric_error_logs.append(
                            f"batch_{i}: "
                            + "; ".join(
                                f"{key}={video_metrics[key]}"
                                for key in metric_error_keys
                            )
                        )

                    if extra_video_metric_names:
                        extra_pred_videos_cpu.append(pred_future.detach().cpu())
                        extra_gt_videos_cpu.append(gt_future.detach().cpu())

                    try:
                        future_action_latent = actions_latent[:, -num_frames:, :]
                        future_rewards = rewards[:, -num_frames:].float()
                        reward_horizon = min(
                            pred_video.shape[1],
                            future_action_latent.shape[1],
                            future_rewards.shape[1],
                        )
                        if reward_horizon <= 0:
                            raise ValueError("reward eval horizon is zero")

                        pred_video_reward = to_minus_1_1_tensor(
                            pred_video[:, :reward_horizon]
                        )
                        pred_video_flat = einops.rearrange(
                            pred_video_reward, "b t c h w -> (b t) c h w"
                        )
                        future_action_latent = future_action_latent[
                            :, :reward_horizon, :
                        ]
                        future_rewards = future_rewards[:, :reward_horizon]
                        future_action_latent_flat = einops.rearrange(
                            future_action_latent, "b t c -> (b t) c"
                        )

                        maybe_sync_cuda()
                        pred_score = model_inner.reward_classifier.predict_score(
                            pred_video_flat, future_action_latent_flat
                        )
                        maybe_sync_cuda()

                        pred_rewards = pred_score.reshape(B_total, reward_horizon)
                        pred_r = pred_rewards.flatten().detach().cpu().float()
                        true_r = future_rewards.flatten().detach().cpu().float()
                        reward_pred_flat_chunks.append(pred_r)
                        reward_true_flat_chunks.append(true_r)

                        pred_event = (pred_rewards > rm_threshold).detach().cpu()
                        true_event = (future_rewards > 0.5).detach().cpu()
                        pred_step = self._first_event_step(
                            pred_event, default_step=reward_horizon + 1
                        )
                        true_step = self._first_event_step(
                            true_event, default_step=reward_horizon + 1
                        )
                        term_abs_error_chunks.append(
                            (pred_step - true_step).abs().float()
                        )
                        term_pred_has_chunks.append(pred_event.any(dim=1))
                        term_true_has_chunks.append(true_event.any(dim=1))
                    except Exception as reward_exc:
                        reward_error_logs.append(
                            f"batch_{i}: {type(reward_exc).__name__}: {reward_exc}"
                        )

                    try:
                        del pred_frames_list, pred_latents, pred_video, pred_video_flat
                        del latents, imgs_flat
                    except Exception:
                        pass
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                if processed_batches == 0:
                    if pipeline_error_logs:
                        job_results["pipeline_stage_failed"] = 1
                        job_results["pipeline_error_log"] = "; ".join(
                            pipeline_error_logs
                        )
                        job_results["error_log"] = job_results["pipeline_error_log"]
                    else:
                        job_results["error_log"] = (
                            "No valid batches found for evaluation"
                        )
                    return job_results

                for metric_name in core_metric_names:
                    if core_metric_weights[metric_name] > 0:
                        job_results[metric_name] = (
                            core_metric_sums[metric_name]
                            / core_metric_weights[metric_name]
                        )

                if extra_video_metric_names and extra_pred_videos_cpu:
                    try:
                        extra_video_metrics = compute_all_metrics(
                            torch.cat(extra_pred_videos_cpu, dim=0).to(device),
                            torch.cat(extra_gt_videos_cpu, dim=0).to(device),
                            device,
                            metrics_to_compute=extra_video_metric_names,
                            strict=False,
                            synchronize_cuda=True,
                            enable_lpips_cpu_fallback=True,
                        )
                        job_results.update(extra_video_metrics)
                        extra_metric_error_keys = [
                            key
                            for key in extra_video_metrics.keys()
                            if key.endswith("_error")
                        ]
                        if extra_metric_error_keys:
                            video_metric_error_logs.append(
                                "; ".join(
                                    f"{key}={extra_video_metrics[key]}"
                                    for key in extra_metric_error_keys
                                )
                            )
                    except Exception as extra_metric_exc:
                        video_metric_error_logs.append(
                            f"extra_metrics: {type(extra_metric_exc).__name__}: {extra_metric_exc}"
                        )

                has_core_video_metric = any(
                    metric_name in job_results for metric_name in core_metric_names
                )
                job_results["video_metrics_stage_failed"] = int(
                    not has_core_video_metric
                )
                job_results["pipeline_stage_failed"] = int(len(pipeline_error_logs) > 0)
                if pipeline_error_logs:
                    job_results["pipeline_error_log"] = "; ".join(pipeline_error_logs)
                if video_metric_error_logs:
                    job_results["video_metric_error_log"] = "; ".join(
                        video_metric_error_logs
                    )

                reward_stage_failed = 0
                if reward_pred_flat_chunks and reward_true_flat_chunks:
                    pred_r = torch.cat(reward_pred_flat_chunks, dim=0).float()
                    true_r = torch.cat(reward_true_flat_chunks, dim=0).float()
                    mse = torch.nn.functional.mse_loss(pred_r, true_r).item()
                    mae = torch.nn.functional.l1_loss(pred_r, true_r).item()

                    pred_centered = pred_r - pred_r.mean()
                    true_centered = true_r - true_r.mean()
                    corr_denom = torch.clamp(
                        pred_centered.norm() * true_centered.norm(), min=1e-8
                    )
                    corr = ((pred_centered * true_centered).sum() / corr_denom).item()

                    abs_error = torch.cat(term_abs_error_chunks, dim=0).float()
                    pred_has = torch.cat(term_pred_has_chunks, dim=0).bool()
                    true_has = torch.cat(term_true_has_chunks, dim=0).bool()
                    negative_mask = ~true_has
                    positive_mask = true_has

                    job_results.update(
                        {
                            "eval_samples": int(eval_sample_count),
                            "num_inference_steps": int(eval_num_inference_steps),
                            "reward_threshold": float(rm_threshold),
                            "reward_MSE": mse,
                            "reward_MAE": mae,
                            "reward_Correlation": corr,
                            "done_step_mae": float(abs_error.mean().item()),
                            "done_step_median_ae": float(abs_error.median().item()),
                            "done_within_1step": float(
                                (abs_error <= 1.0).float().mean().item()
                            ),
                            "done_within_2step": float(
                                (abs_error <= 2.0).float().mean().item()
                            ),
                            "termination_event_accuracy": float(
                                (pred_has == true_has).float().mean().item()
                            ),
                            "pred_terminate_rate": float(
                                pred_has.float().mean().item()
                            ),
                            "true_terminate_rate": float(
                                true_has.float().mean().item()
                            ),
                        }
                    )

                    if bool(positive_mask.any().item()):
                        job_results["done_step_mae_on_gt_done"] = float(
                            abs_error[positive_mask].mean().item()
                        )
                        job_results["termination_false_negative_rate"] = float(
                            ((~pred_has) & positive_mask).float().sum().item()
                            / max(1.0, positive_mask.float().sum().item())
                        )
                    else:
                        job_results["done_step_mae_on_gt_done"] = 0.0
                        job_results["termination_false_negative_rate"] = 0.0

                    if bool(negative_mask.any().item()):
                        job_results["termination_false_positive_rate"] = float(
                            (pred_has & negative_mask).float().sum().item()
                            / max(1.0, negative_mask.float().sum().item())
                        )
                    else:
                        job_results["termination_false_positive_rate"] = 0.0
                else:
                    reward_stage_failed = 1
                    if not reward_error_logs:
                        reward_error_logs.append(
                            "No valid reward batches found for evaluation"
                        )

                if reward_error_logs:
                    job_results["reward_error_log"] = "; ".join(reward_error_logs)

                job_results["reward_stage_failed"] = reward_stage_failed
                if not has_core_video_metric:
                    if pipeline_error_logs:
                        job_results["error_log"] = job_results.get(
                            "pipeline_error_log",
                            "pipeline stage failed",
                        )
                    elif video_metric_error_logs:
                        job_results["error_log"] = job_results.get(
                            "video_metric_error_log",
                            "video metric stage failed",
                        )
                    elif reward_stage_failed:
                        job_results["error_log"] = job_results.get(
                            "reward_error_log",
                            "reward stage failed",
                        )
                elif reward_stage_failed:
                    job_results["error_log"] = job_results.get(
                        "reward_error_log",
                        "reward stage failed",
                    )

                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                return job_results

            for eval_job in eval_jobs:
                prefix = str(eval_job.get("prefix", ""))
                try:
                    job_metrics = run_eval_job(eval_job)
                except Exception as e:
                    warnings.warn(
                        f"WM eval job failed ({eval_job['requested_type']}): {e}"
                    )
                    job_metrics = {"error_log": str(e)}

                for key, value in job_metrics.items():
                    results[f"{prefix}{key}"] = value

            # final cleanup of large objects (GPU and CPU)
            try:
                for name in (
                    "pred_frames_list",
                    "pred_latents",
                    "pred_video",
                    "pred_video_flat",
                    "latents",
                    "imgs",
                    "imgs_flat",
                ):
                    if name in locals():
                        try:
                            del locals()[name]
                        except Exception:
                            pass
            except Exception:
                pass
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            # ensure results contains only python types and no tensors
            def ensure_result_safe(obj):
                if isinstance(obj, dict):
                    return {k: ensure_result_safe(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [ensure_result_safe(x) for x in obj]
                if torch.is_tensor(obj):
                    try:
                        return obj.detach().cpu().item()
                    except Exception:
                        return obj.detach().cpu().numpy().tolist()
                return obj

            results = ensure_result_safe(results)
            self.world_model.train()
            return results

        except Exception as e_outer:
            # top-level catch: never let raw exception propagate to Ray (which would trigger serialization of locals)
            tb = traceback.format_exc()
            warnings.warn(f"[wm eval] uncaught exception: {e_outer}\n{tb}")
            # ensure no heavy objects are referenced in returned dict
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
            return {
                "global_steps": global_steps,
                "error_log": str(e_outer),
                "traceback": tb,
            }

        finally:
            # OPTIONAL: cleanup training shards for this global_steps
            cleanup_info = None
            try:
                print("[wm eval] Clean up tmp evaluating data")
                cleanup_type = "eval_real"
                dl_meta = getattr(self, "_last_wm_dataloader_meta", None)
                if isinstance(dl_meta, dict):
                    cleanup_type = dl_meta.get("actual_type", cleanup_type)
                cleanup_info = self._cleanup_data_shards(
                    global_steps, type_=cleanup_type
                )
            except Exception as e:
                cleanup_info = {"cleaned": False, "error": str(e)}
            finally:
                print(cleanup_info)


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config

        _ensure_dist_process_group(backend="nccl")

        # build device mesh
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        self.role = role
        assert self.role in [
            "actor",
            "rollout",
            "ref",
            "actor_rollout",
            "actor_rollout_ref",
        ]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in [
            "rollout",
            "actor_rollout",
            "actor_rollout_ref",
        ]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get(
                "param_offload", False
            )
            self._is_offload_grad = self.config.actor.fsdp_config.get(
                "grad_offload", False
            )
            self._is_offload_optimizer = self.config.actor.fsdp_config.get(
                "optimizer_offload", False
            )
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get(
                "param_offload", False
            )

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.shape[0]
            self.config.actor.ppo_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.shape[0]

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        log_gpu_memory_usage("Before init from HF AutoModel", logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )
        if self.config.model.use_remove_padding:
            from verl.models.registry import check_model_support_rmpad

            check_model_support_rmpad(actor_model_config.model_type)
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(
            actor_model_config, override_config_kwargs=override_config_kwargs
        )
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from liger_kernel.transformers import AutoLigerKernelForCausalLM

            actor_module = AutoLigerKernelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable()
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage("After init from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        if self._is_ref:
            mixed_precision = None

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module, config=fsdp_config.get("wrap_policy", None)
        )

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f"wrap_policy: {auto_wrap_policy}")

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(
            actor_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
        )

        log_gpu_memory_usage("After Actor FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 0.0),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            actor_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
            )
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        return (
            actor_module_fsdp,
            actor_optimizer,
            actor_lr_scheduler,
            actor_model_config,
        )

    def _build_rollout(self):
        if self.config.rollout.name == "hf":
            from verl.workers.hybrid_engine import BaseShardingManager
            from verl.workers.rollout import HFRollout

            rollout = HFRollout(
                module=self.actor_module_fsdp, config=self.config.rollout
            )
            sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == "vllm":
            from verl.workers.hybrid_engine import FSDPVLLMShardingManager
            from verl.workers.rollout.vllm_rollout import vLLMRollout

            log_gpu_memory_usage("Before building vllm rollout", logger=None)
            rollout = vLLMRollout(
                actor_module=self.actor_module_fsdp,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
            )
            log_gpu_memory_usage("After building vllm rollout", logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
            )
            log_gpu_memory_usage("After building sharding manager", logger=None)

        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                enable_gradient_checkpointing=self.config.model.get(
                    "enable_gradient_checkpointing", False
                ),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
            )

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage(
                    "After offload actor grad during init", logger=logger
                )
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage(
                    "After offload actor optimizer during init", logger=logger
                )
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
            )[0]
            if self._is_offload_param:
                offload_fsdp_param_and_grad(
                    module=self.ref_module_fsdp, offload_grad=self._is_offload_grad
                )

            OmegaConf.set_struct(self.config.ref, True)
            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref, actor_module=self.ref_module_fsdp
            )

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to("cuda")

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.actor_optimizer, device_id=torch.cuda.current_device()
            )

        data.batch = data.batch.cuda()

        log_gpu_memory_usage("Before update policy", logger=logger)

        actor_lr_scale, actor_base_lrs = _prepare_actor_lr_for_update(
            self.actor_optimizer, self.actor_lr_scheduler, data
        )
        metrics = self.actor.update_policy(data=data)

        _finalize_actor_lr_after_update(
            self.actor_optimizer,
            self.actor_lr_scheduler,
            metrics,
            lr_scale=actor_lr_scale,
            base_lrs=actor_base_lrs,
        )

        log_gpu_memory_usage("After update policy", logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_entropy(self, data: DataProto):

        data = data.to("cuda")

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        data.batch = data.batch.cuda()

        log_gpu_memory_usage("Before compute entropy", logger=logger)

        metrics = self.actor.compute_entropy(bacth_data=data)

        log_gpu_memory_usage("After compute entropy", logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to("cuda")
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get("recompute_log_prob", True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        prompts.batch = prompts.batch.cuda()
        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.sharding_manager:
            log_gpu_memory_usage("After entering sharding manager", logger=logger)

            prompts = self.sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.sharding_manager.postprocess_data(output)
            torch.cuda.synchronize()

        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info["micro_batch_size"] = (
                self.config.rollout.log_prob_micro_batch_size
            )
            output.meta_info["temperature"] = self.config.rollout.temperature
            output.meta_info["use_dynamic_bsz"] = (
                self.config.rollout.log_prob_use_dynamic_bsz
            )
            output.meta_info["max_token_len"] = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            old_log_probs = self.actor.compute_log_prob(data=output)
            output.batch["old_log_probs"] = old_log_probs

        output = output.to("cpu")

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        # clear kv cache
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After recompute log prob", logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to("cuda")

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.ref_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={"ref_log_prob": output})

        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.ref_module_fsdp, offload_grad=self._is_offload_grad
            )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        import torch

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        checkpoint_format = normalize_fsdp_checkpoint_format(
            getattr(self.config.model, "checkpoint_format", None)
        )
        if checkpoint_format not in {
            FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT,
            FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT,
            HF_FULL_STATE_DICT_CHECKPOINT_FORMAT,
        }:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Unknown actor checkpoint_format={checkpoint_format!r}; fallback to {FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT}."
                )
            checkpoint_format = FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT
        elif checkpoint_format == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Actor checkpoint_format={checkpoint_format!r} is incompatible with DeviceMesh autosave; upgrade to {FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT}."
                )
            checkpoint_format = resolve_fsdp_lightweight_save_format(checkpoint_format)

        if checkpoint_format == FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
            save_info = _save_fsdp_lightweight_state_dict_checkpoint(
                self.actor.actor_module,
                local_path,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                component="actor",
                base_model_path=getattr(self.config.model, "path", None),
                tokenizer_path=getattr(self.config.model, "tokenizer_path", None),
            )
            if dist.get_rank() == 0:
                print(
                    f"Saved lightweight actor checkpoint to {local_path} ({format_num_bytes(save_info.get('estimated_bytes')) if save_info.get('estimated_bytes') is not None else 'size=unknown'})"
                )
                if hdfs_path is not None:
                    print(f"Uploading actor checkpoint to {hdfs_path}")
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=local_path, dst=hdfs_path)
            if self._is_offload_param:
                offload_fsdp_param_and_grad(
                    module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
                )
            return

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg
        ):
            state_dict = self.actor.actor_module.state_dict()
        if self.rank == 0:
            print(f"Saving actor checkpoint to {local_path}")
            os.makedirs(local_path, exist_ok=True)
            self.actor_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            synced_stats_path = _copy_vla_dataset_statistics(
                local_path,
                self.config.model.path,
                getattr(self.config.model, "tokenizer_path", None),
            )
            if synced_stats_path is not None:
                print(f"Saved dataset statistics to {synced_stats_path}")
            if hdfs_path is not None:
                print(f"Uploading actor checkpoint to {hdfs_path}")
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor

        checkpoint_meta = _load_fsdp_checkpoint_meta(local_path)
        checkpoint_format = (
            str((checkpoint_meta or {}).get("format", "")).strip().lower()
        )
        if checkpoint_format == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
            raise RuntimeError(
                "Legacy fsdp_local_state_dict actor checkpoints are not resumable with the current DeviceMesh runtime."
            )
        if checkpoint_format != FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Actor checkpoint at {local_path} is format={checkpoint_format or 'unknown'}; assume HF/preloaded path and skip explicit shard restore."
                )
            torch.distributed.barrier()
            return {"loaded": False, "format": checkpoint_format or "unknown"}

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        load_info = _load_fsdp_lightweight_state_dict_checkpoint(
            self.actor.actor_module,
            local_path,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            component="actor",
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        torch.cuda.empty_cache()
        return load_info


class CriticWorker(Worker):
    def __init__(self, config):
        super().__init__()

        _ensure_dist_process_group(backend="nccl")
        self.config = config
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size()
        self.config.ppo_micro_batch_size //= torch.distributed.get_world_size()

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.
        # TODO: support loading critic weights from RM. Support using AutoModelForTokenClassification
        from transformers import AutoTokenizer

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(
            tokenizer_path,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )

        from omegaconf import OmegaConf

        override_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from torch import nn
        from transformers import AutoConfig, AutoModelForCausalLM

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=critic_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            critic_module.lm_head = nn.Sequential(
                nn.Linear(critic_model_config.hidden_size, 1, dtype=torch_dtype),
                LambdaLayer(fn=squeeze),
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable()
        if self.rank == 0:
            print_model_size(critic_module)

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module, config=self.config.model.fsdp_config.wrap_policy
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        critic_module = FSDP(
            critic_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            sync_module_states=True,
        )

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup

        critic_lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
        )

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = (
            self._build_critic_model_optimizer(self.config)
        )

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.critic_module, offload_grad=self._is_offload_grad
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(
            config=self.config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
        )
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to("cuda")

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.critic_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        micro_batch_size = self.config.ppo_micro_batch_size
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.critic_module, offload_grad=self._is_offload_grad
            )
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to("cuda")
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.critic_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.critic_optimizer, device_id=torch.cuda.current_device()
            )
        metrics = self.critic.update_critic(data=data)

        self.critic_lr_scheduler.step()
        lr = self.critic_lr_scheduler.get_last_lr()[0]
        metrics["critic/lr(1e-4)"] = lr * 1e4

        output = DataProto(batch=None, meta_info={"metrics": metrics})
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.critic_module, offload_grad=self._is_offload_grad
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.critic_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        checkpoint_format = normalize_fsdp_checkpoint_format(
            getattr(self.config.model, "checkpoint_format", None)
        )
        if checkpoint_format not in {
            FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT,
            FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT,
            HF_FULL_STATE_DICT_CHECKPOINT_FORMAT,
        }:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Unknown critic checkpoint_format={checkpoint_format!r}; fallback to {FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT}."
                )
            checkpoint_format = FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT
        elif checkpoint_format == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Critic checkpoint_format={checkpoint_format!r} is incompatible with DeviceMesh autosave; upgrade to {FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT}."
                )
            checkpoint_format = resolve_fsdp_lightweight_save_format(checkpoint_format)

        if checkpoint_format == FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
            save_info = _save_fsdp_lightweight_state_dict_checkpoint(
                self.critic_module,
                local_path,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                component="critic",
                base_model_path=getattr(self.config.model, "path", None),
                tokenizer_path=getattr(self.config.model, "tokenizer_path", None),
            )
            if dist.get_rank() == 0:
                print(
                    f"Saved lightweight critic checkpoint to {local_path} ({format_num_bytes(save_info.get('estimated_bytes')) if save_info.get('estimated_bytes') is not None else 'size=unknown'})"
                )
                if hdfs_path is not None:
                    print(f"Uploading critic checkpoint to {hdfs_path}")
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=local_path, dst=hdfs_path)
            if self._is_offload_param:
                offload_fsdp_param_and_grad(
                    module=self.critic_module, offload_grad=self._is_offload_grad
                )
            return

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.critic_module, StateDictType.FULL_STATE_DICT, cfg
        ):
            state_dict = self.critic_module.state_dict()
        if self.rank == 0:
            print(f"Saving critic checkpoint to {local_path}")
            os.makedirs(local_path, exist_ok=True)
            self.critic_module._fsdp_wrapped_module.save_pretrained(
                local_path, state_dict=state_dict
            )
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f"Uploading critic checkpoint to {hdfs_path}")
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.critic_module, offload_grad=self._is_offload_grad
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None):
        checkpoint_meta = _load_fsdp_checkpoint_meta(local_path)
        checkpoint_format = (
            str((checkpoint_meta or {}).get("format", "")).strip().lower()
        )
        if checkpoint_format == FSDP_LOCAL_STATE_DICT_CHECKPOINT_FORMAT:
            raise RuntimeError(
                "Legacy fsdp_local_state_dict critic checkpoints are not resumable with the current DeviceMesh runtime."
            )
        if checkpoint_format != FSDP_SHARDED_STATE_DICT_CHECKPOINT_FORMAT:
            if dist.get_rank() == 0:
                print(
                    f"[checkpoint] Critic checkpoint at {local_path} is format={checkpoint_format or 'unknown'}; assume HF/preloaded path and skip explicit shard restore."
                )
            torch.distributed.barrier()
            return {"loaded": False, "format": checkpoint_format or "unknown"}

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.critic_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        load_info = _load_fsdp_lightweight_state_dict_checkpoint(
            self.critic_module,
            local_path,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            component="critic",
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.critic_module, offload_grad=self._is_offload_grad
            )
        torch.cuda.empty_cache()
        return load_info


class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def __init__(self, config):
        super().__init__()

        _ensure_dist_process_group(backend="nccl")
        self.config = config

        self.config.micro_batch_size //= torch.distributed.get_world_size()

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(
                config.model.input_tokenizer
            )
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path,
                trust_remote_code=config.model.get("trust_remote_code", False),
            )
            self.tokenizer = hf_tokenizer(
                local_path,
                trust_remote_code=config.model.get("trust_remote_code", False),
            )

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )
        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_module = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=reward_module, config=self.config.model.fsdp_config
        )

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(
                offload_params=self.config.model.fsdp_config.param_offload
            ),
        )

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.reward_module(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"],
                position_ids=micro_batch["position_ids"],
            )
            rm_score = output.logits  # (batch_size,)
            rm_score = rm_score.squeeze(-1)
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(
            attention_mask, dtype=scores.dtype
        )  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][
                -response_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {
            "input_ids": rm_input_ids,
            "attention_mask": rm_attention_mask,
            "position_ids": rm_position_ids,
        }

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to("cuda")
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)

        rm_data.batch = rm_data.batch.cuda()
        micro_batches = rm_data.batch.split(self.config.micro_batch_size)
        output = []
        for micro_batch in micro_batches:
            rm_score = self._forward_micro_batch(micro_batch)
            output.append(rm_score)
        scores = torch.cat(output, dim=0)  # (batch_size)
        token_level_scores = self._expand_to_token_level(data, scores)
        # Note that this is only the scores, may not be the final rewards used to train RL
        output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
        output = output.to("cpu")
        torch.cuda.empty_cache()
        return output


class PRIMERewardModelWorker(Worker):
    """
    PRIME reward model.
    Can update itself whenever compute_rm_score is called.
    """

    def __init__(self, config):
        super().__init__()

        _ensure_dist_process_group(backend="nccl")
        self.config = config

        world_size = torch.distributed.get_world_size()
        self.config.mini_batch_size //= world_size
        self.config.micro_batch_size //= world_size
        # build device mesh

        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        self._is_offload_param = self.config.prime_model.fsdp_config.get(
            "param_offload", False
        )
        self._is_offload_grad = self.config.prime_model.fsdp_config.get(
            "grad_offload", False
        )
        self._is_offload_optimizer = self.config.prime_model.fsdp_config.get(
            "optimizer_offload", False
        )

    def _build_model_optimizer(self, config, enable_gradient_checkpointing=False):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.prime_model.path)

        if self.config.prime_model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(
                config.prime_model.input_tokenizer
            )
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path,
                trust_remote_code=config.prime_model.get("trust_remote_code", False),
            )
            self.tokenizer = hf_tokenizer(
                local_path,
                trust_remote_code=config.prime_model.get("trust_remote_code", False),
            )

        trust_remote_code = config.prime_model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )
        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        if config.prime_model.use_remove_padding:
            from verl.models.registry import check_model_support_rmpad

            check_model_support_rmpad(model_config.model_type)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from liger_kernel.transformers import AutoLigerKernelForCausalLM

            reward_module = AutoLigerKernelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            reward_module.to(torch.float32)
            if enable_gradient_checkpointing:
                reward_module.gradient_checkpointing_enable()
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        if config.prime_model.get("enable_gradient_checkpointing", False):
            reward_module.gradient_checkpointing_enable()

        if config.prime_model.get("ref_type", "freeze") == "freeze":
            reference_module = AutoLigerKernelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=copy_local_path_from_hdfs(
                    config.prime_model.ref_path
                ),
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            reference_module.to(torch.bfloat16)
            for param in reference_module.parameters():
                param.requires_grad = False
        else:
            reference_module = None

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=reward_module, config=self.config.prime_model.fsdp_config
        )

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=reference_module, config=self.config.prime_model.fsdp_config
        )
        if reference_module is not None:
            reference_module = FSDP(
                reference_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
                device_mesh=self.device_mesh,
                sync_module_states=True,
            )

        self.update_dpo_type = self.config.prime_model.get("update", "none")
        if self.update_dpo_type in ["before", "after"]:

            from torch import optim

            self.reward_optimizer = optim.AdamW(
                reward_module.parameters(),
                lr=config.prime_model.optim.lr,
                betas=config.prime_model.optim.get("betas", (0.9, 0.999)),
                weight_decay=config.prime_model.optim.get("weight_decay", 1e-2),
            )

            total_steps = config.prime_model.optim.get("total_training_steps", 0)
            num_warmup_steps_ratio = config.prime_model.optim.get(
                "lr_warmup_steps_ratio", 0.0
            )
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            from verl.utils.torch_functional import get_constant_schedule_with_warmup

            self.reward_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.reward_optimizer, num_warmup_steps=num_warmup_steps
            )

            # fsdp offload configurations
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.reward_optimizer)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=reward_module, offload_grad=self._is_offload_grad
            )
            if reference_module is not None:
                offload_fsdp_param_and_grad(
                    module=reference_module, offload_grad=self._is_offload_grad
                )

        return reward_module, reference_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPRIME

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.prime_model.get("external_lib", None))
        self.reward_module, self.reference_module = self._build_model_optimizer(
            config=self.config,
            enable_gradient_checkpointing=self.config.prime_model.get(
                "enable_gradient_checkpointing", False
            ),
        )
        self.prm = DataParallelPRIME(
            config=self.config,
            reward_module=self.reward_module,
            reference_module=self.reference_module,
            reward_optimizer=self.reward_optimizer,
            prime_loss_fn=self.config.prime_model.get("loss_type", "ce"),
        )
        torch.cuda.empty_cache()

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][
                -response_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {
            "input_ids": rm_input_ids,
            "attention_mask": rm_attention_mask,
            "position_ids": rm_position_ids,
        }

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        n_samples = data.meta_info["n_samples"]
        beta = self.config.prime_model.get("beta_train", 0.05)
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_data = data

        if self.update_dpo_type != "none":
            if self._is_offload_optimizer:
                load_fsdp_optimizer(
                    optimizer=self.reward_optimizer,
                    device_id=torch.cuda.current_device(),
                )
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.reward_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
            if self.reference_module is not None:
                load_fsdp_param_and_grad(
                    module=self.reference_module,
                    device_id=torch.cuda.current_device(),
                    load_grad=self._is_offload_grad,
                )

        token_level_scores, metrics = self.prm.update_policy(rm_data)

        output = DataProto.from_dict(
            tensors={"rm_scores": token_level_scores}, meta_info={"metrics": metrics}
        )

        if self.update_dpo_type != "none":
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.reward_optimizer)
            self.reward_lr_scheduler.step()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.reward_module, offload_grad=self._is_offload_grad
            )
            if self.reference_module is not None:
                offload_fsdp_param_and_grad(
                    module=self.reference_module, offload_grad=self._is_offload_grad
                )

        output = output.to("cpu")
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.reward_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.reward_module, StateDictType.FULL_STATE_DICT, cfg
        ):
            state_dict = self.reward_module.state_dict()
        if self.rank == 0:
            print(f"Saving reward checkpoint to {local_path}")
            os.makedirs(local_path, exist_ok=True)
            self.reward_module._fsdp_wrapped_module.save_pretrained(
                local_path, state_dict=state_dict
            )
            if hdfs_path is not None:
                print(f"Uploading reward checkpoint to {hdfs_path}")
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.reward_module, offload_grad=self._is_offload_grad
            )
