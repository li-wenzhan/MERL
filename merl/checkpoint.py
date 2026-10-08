"""Atomic tensor-only runtime snapshots; model weights are saved separately."""

import random
from pathlib import Path

import numpy as np
import torch


def cpu_tensors(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tensors(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tensors(item) for item in value)
    return value


def rng_state():
    algorithm, keys, position, gaussian, cached = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(), numpy=dict(algorithm=algorithm, keys=torch.from_numpy(keys.astype(np.int64)),
                                                    position=position, gaussian=gaussian, cached=cached))


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["algorithm"], n["keys"].numpy().astype(np.uint32), n["position"], n["gaussian"], n["cached"]))


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(cpu_tensors(payload), temporary)
    temporary.replace(path)
