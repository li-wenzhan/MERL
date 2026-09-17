"""Load LIBERO NumPy initial states without unrestricted pickle deserialization."""

from pathlib import Path

import numpy as np
import torch


def load_init_states(path):
    # Official LIBERO .pruned_init files contain NumPy arrays, not just tensors.
    # Keep the allowlist scoped to this read; never change torch.load globally.
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, bytes,
               type(np.dtype(np.float32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        states = torch.load(path, map_location="cpu", weights_only=True)
    values = np.asarray(states)
    if (values.ndim != 2 or 0 in values.shape or values.dtype.kind != "f"
            or not np.isfinite(values).all()):
        raise ValueError(f"LIBERO initial states must be a finite floating [N, D] array: {path}")
    return states


def load_task_init_states(task_suite, task_id):
    from libero.libero import get_libero_path

    task = task_suite.get_task(task_id)
    root = Path(get_libero_path("init_states")).resolve()
    path = (root / task.problem_folder / task.init_states_file).resolve()
    if not path.is_relative_to(root):
        raise ValueError("LIBERO initial-state file must be inside the configured root")
    return load_init_states(path)
