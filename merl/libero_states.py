"""Load LIBERO NumPy initial states without unrestricted pickle deserialization."""

from pathlib import Path
import pickle
import zipfile

import numpy as np
import torch


class _NumpyStateUnpickler(pickle.Unpickler):
    """Read the NumPy-only ZIP format emitted by LIBERO-PRO's generator."""

    def find_class(self, module, name):
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
                   ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
                   ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported initial-state global: {module}.{name}")
        return allowed[module, name]


def load_init_states(path):
    # Official LIBERO .pruned_init files contain NumPy arrays, not just tensors.
    # Keep the allowlist scoped to this read; never change torch.load globally.
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, bytes,
               type(np.dtype(np.float32)), type(np.dtype(np.float64))]
    generated = False
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if set(archive.namelist()) == {"archive/data.pkl", "archive/version"}:
                if archive.read("archive/version").strip() != b"1":
                    raise ValueError("Unsupported LIBERO-PRO initial-state archive version")
                with archive.open("archive/data.pkl") as stream:
                    states = _NumpyStateUnpickler(stream).load()
                generated = True
    if not generated:
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
