import os
import random
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds

from typing import Optional, Any


_GLOBAL_PARALLEL_GROUPS = dict()


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for Python, numpy and PyTorch to improve reproducibility.

    Parameters
    ----------
    seed : int
        Base seed to set.
    deterministic : bool
        If True, try to make CUDA / CuDNN algorithms deterministic.
        Note: deterministic mode can be slower and some ops may not have deterministic implementations.
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Python hash seed (makes hash() deterministic across processes)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CuDNN / deterministic behaviour
    # - torch.use_deterministic_algorithms enforces deterministic kernels (PyTorch >=1.8)
    # - fallback to cudnn flags for older versions / compatibility
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        # some torch versions may not support the API or specific ops will raise
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_current_device() -> torch.device:
    """
    Return a torch.device for the current process in a robust way (works for single-GPU,
    multi-GPU single-process, and distributed per-process launches where LOCAL_RANK is set).

    Strategy:
      1. If torch.distributed is initialized, use LOCAL_RANK (env) or torch.distributed.get_rank() to pick device.
      2. Else if torch.cuda.is_available(), prefer the device selected by CUDA_VISIBLE_DEVICES
         / the current CUDA context; return torch.device("cuda", current_device).
      3. Otherwise return CPU.

    Notes:
      - In multi-process multi-GPU launches (torch.distributed.launch / accelerate),
        processes are typically started with LOCAL_RANK env var. We respect that.
      - After obtaining the device index, we call torch.cuda.set_device(idx) to avoid surprises.
    """
    # prefer explicit LOCAL_RANK (set by many launchers: torchrun / accelerate / deepspeed)
    local_rank = None
    if "LOCAL_RANK" in os.environ:
        try:
            local_rank = int(os.environ["LOCAL_RANK"])
        except Exception:
            local_rank = None

    # fallback: some systems use RANK and WORLD_SIZE, but LOCAL_RANK is best for per-process GPU id
    if local_rank is not None:
        if torch.cuda.is_available():
            # If CUDA_VISIBLE_DEVICES maps logical->physical, using local_rank is correct
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        else:
            return torch.device("cpu")

    # if distributed initialized and no LOCAL_RANK env, we can try using rank % n_gpus
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            rank = torch.distributed.get_rank()
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                idx = rank % num_gpus
                torch.cuda.set_device(idx)
                return torch.device("cuda", idx)
        except Exception:
            pass

    # non-distributed: use current cuda device if available
    if torch.cuda.is_available():
        try:
            cur = torch.cuda.current_device()
            # ensure device is set (no-op if already set)
            torch.cuda.set_device(cur)
            return torch.device("cuda", cur)
        except Exception:
            # fallback to simple "cuda"
            return torch.device("cuda")

    # default CPU
    return torch.device("cpu")


def set_data_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["data"] = group


def get_data_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("data", dist.group.WORLD)


def set_sequence_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_sequence_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("sequence", None)


def prepare_dataloader(
    dataset,
    batch_size=None,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group: Optional[Any] = None,
    bucket_config=None,
    num_bucket_build_workers=1,
    prefetch_factor=None,
    cache_pin_memory=False,
    val = False,
    **kwargs,
):
    _kwargs = kwargs.copy()
    # pipeline = simple_ds.ds.with_epoch(simple_ds.epoch_size)
    pipeline = dataset.ds
    if val:
        dataloader = wds.WebLoader(
            pipeline,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        dataloader = wds.WebLoader(
            pipeline,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        ).repeat()
    return (dataloader, None)
