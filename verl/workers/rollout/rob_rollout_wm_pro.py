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
# MERL memory / GLX runtime patch:
# 1. Emit compact WM confidence scalars for scheduler/replay.
# 2. Keep full video tensors optional and worker-local unless explicitly persisted.
# 3. Serialize LIBERO GLX env creation and isolate Xvfb displays per actor.
# import sys
# sys.path.insert(0, "/path/to/SimpleVLA-RL")
# import libero_safe_patch

import ast
import contextlib
import os
import re
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from types import SimpleNamespace

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux training nodes provide fcntl.
    fcntl = None

import cv2
import einops
import torch
import torch.distributed
import torch.distributed as dist
from merl.libero_states import load_task_init_states
from merl.rollout_contract import valid_response_tokens as count_valid_response_tokens
import yaml
from ray import get
from sympy import get_contraction_structure
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
import importlib
import importlib.util

import verl.utils.torch_functional as verl_F
from transformers import GenerationConfig
from verl import DataProto
from verl.utils.libero_path import load_libero_pro_config
from verl.utils.libero_runtime import (
    configure_libero_runtime_env,
    format_libero_runtime_env_summary,
)
from verl.utils.libero_pro_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    invert_gripper_action,
    load_initial_states_by_path,
    normalize_gripper_action,
    quat2axisangle,
)
from verl.utils.libero_utils import resize_image, save_rollout_video
from verl.utils.task_description_contract import normalize_task_descriptions
from verl.utils.torch_functional import get_eos_mask
from verl.utils.tokenizer import hf_processor

from .base import BaseRollout

import gc

# For Libero multiprocessing
import multiprocessing
import queue
import random
import threading
import time
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from codetiming import Timer
from modules.ctrl_world.models.ctrl_world_new import CtrlWorld
from modules.ctrl_world.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from modules.libero_pro import perturbation
from PIL import Image
from verl.utils.vla_utils.openvla_oft.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)

__all__ = ["RobHFRollout"]

# Environment initialization lock for Robotwin
_ENV_INIT_LOCK = threading.Lock()
_DEFAULT_LIBERO_ENV_INIT_LOCK_PATH = "/tmp/merl_libero_glx_env_init.lock"
_NATIVE_GL_CRASH_EXITCODES = {-4, -6, -7, -8, -11}
_LIBERO_GL_FALLBACK_STATE = {"force_fallback": False}

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _normalize_libero_pro_seed(seed_value: Any) -> Optional[Any]:
    if seed_value is None:
        return None
    if isinstance(seed_value, type):
        return None
    if isinstance(seed_value, str) and seed_value.strip().lower() in {
        "",
        "none",
        "null",
    }:
        return None
    return seed_value


def _load_repo_local_perturbation_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "modules"
        / "libero_pro"
        / "perturbation.py"
    )
    if not module_path.is_file():
        return None

    spec = importlib.util.spec_from_file_location(
        "_merl_repo_local_libero_pro_perturbation", str(module_path)
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manual_libero_pro_create_env(
    perturbation_module: Any, configs: Dict[str, Any]
) -> None:
    missing = [
        name
        for name in ("PerturbFlags", "process_bddl_file_mixed", "EvalEnvCreator")
        if not hasattr(perturbation_module, name)
    ]
    if missing:
        raise AttributeError(
            "modules.libero_pro.perturbation is missing both create_env() and "
            f"fallback components: {missing}"
        )

    cfg = dict(configs or {})
    flags = perturbation_module.PerturbFlags(
        use_environment=bool(cfg.get("use_environment", False)),
        use_swap=bool(cfg.get("use_swap", False)),
        use_object=bool(cfg.get("use_object", False)),
        use_language=bool(cfg.get("use_language", False)),
        use_task=bool(cfg.get("use_task", False)),
    )

    perturb_flag = str(cfg.get("perturb_flag", "") or "")
    perturb_key = (
        dict(cfg.get("perturbation_mapping") or {}).get(perturb_flag)
        or cfg.get("perturbation")
        or "temp"
    )
    input_dir = os.path.join(
        str(cfg.get("bddl_files_path", "") or ""),
        str(cfg.get("task_suite_name", "") or ""),
    )
    temp_output_dir = perturbation_module.process_bddl_file_mixed(
        input_dir=input_dir,
        task_suite_name=str(cfg.get("task_suite_name", "") or ""),
        flags=flags,
        configs=dict(cfg.get("ood_task_configs") or {}),
        perturb_key=str(perturb_key),
        seed=_normalize_libero_pro_seed(cfg.get("seed", None)),
    )

    creator = perturbation_module.EvalEnvCreator(
        input_dir=temp_output_dir,
        script_path=str(cfg.get("script_path", "") or ""),
        base_output_dir=str(cfg.get("init_file_dir", "") or ""),
        libero_root=str(cfg.get("libero_pro_root", "") or ""),
    )
    creator.create_env()


def _create_libero_pro_env_assets(configs: Dict[str, Any]) -> None:
    create_env = getattr(perturbation, "create_env", None)
    if callable(create_env):
        create_env(configs=configs)
        return

    try:
        repo_local_perturbation = _load_repo_local_perturbation_module()
    except Exception as exc:
        print(
            "[LIBERO PRO] Failed to load repo-local perturbation fallback: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        repo_local_perturbation = None
    if repo_local_perturbation is not None:
        create_env = getattr(repo_local_perturbation, "create_env", None)
        if callable(create_env):
            print(
                "[LIBERO PRO] Using repo-local perturbation.create_env fallback "
                f"from {getattr(repo_local_perturbation, '__file__', '<unknown>')}",
                flush=True,
            )
            create_env(configs=configs)
            return
        _manual_libero_pro_create_env(repo_local_perturbation, configs)
        return

    _manual_libero_pro_create_env(perturbation, configs)


def _get_libero_pro_benchmark_dict(config):
    from verl.utils.libero_path import ensure_libero_pro_root

    _apply_libero_runtime_config_to_env(config)
    ensure_libero_pro_root(
        evaluation_config_path=getattr(config, "libero_pro_eval_config_path", None)
    )

    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()


def _build_libero_env_worker_config(config):
    runtime_env = configure_libero_runtime_env(force_headless=True)
    worker_backend = _normalize_gl_backend(
        getattr(
            config,
            "env_worker_mujoco_gl",
            os.environ.get(
                "MERL_LIBERO_ENV_BACKEND",
                os.environ.get("MERL_LIBERO_ENV_MUJOCO_GL", "egl"),
            ),
        )
    )
    if worker_backend in {"", "auto"}:
        worker_backend = str(runtime_env["backend"])
    if worker_backend not in {"glx", "egl", "osmesa"}:
        worker_backend = str(runtime_env["backend"])
    worker_pyopengl_platform = "glx" if worker_backend == "glx" else worker_backend
    display = _resolve_libero_worker_display(worker_backend)
    render_cuda_visible_devices = _resolve_libero_render_cuda_visible_devices(
        config, worker_backend
    )
    return SimpleNamespace(
        libero_pro_eval_config_path=getattr(
            config, "libero_pro_eval_config_path", None
        ),
        model_family=str(getattr(config, "model_family", "openvla")),
        num_steps_wait=int(getattr(config, "num_steps_wait", 10)),
        env_init_max_retry=int(
            getattr(
                config,
                "env_init_max_retry",
                os.environ.get("MERL_LIBERO_ENV_INIT_MAX_RETRY", 1),
            )
        ),
        env_init_timeout_s=float(getattr(config, "env_init_timeout_s", 300.0)),
        env_step_timeout_s=float(getattr(config, "env_step_timeout_s", 60.0)),
        env_init_parent_max_retry=int(
            getattr(
                config,
                "env_init_parent_max_retry",
                os.environ.get("MERL_LIBERO_ENV_PARENT_MAX_RETRY", 1),
            )
        ),
        env_mp_start_method=str(runtime_env["start_method"]),
        display=str(display),
        mujoco_gl=str(worker_backend),
        pyopengl_platform=str(worker_pyopengl_platform),
        env_render_cuda_visible_devices=str(render_cuda_visible_devices),
        mujoco_egl_device_id=str(
            _resolve_mujoco_egl_device_id(
                config,
                visible_devices_override=render_cuda_visible_devices,
            )
        ),
        env_init_lock_enable=str(
            getattr(
                config,
                "env_init_lock_enable",
                os.environ.get("MERL_LIBERO_ENV_INIT_LOCK_ENABLE", "auto"),
            )
        ),
        env_init_lock_path=str(_resolve_libero_env_lock_path(config, display)),
        env_init_lock_timeout_s=float(
            getattr(
                config,
                "env_init_lock_timeout_s",
                os.environ.get("MERL_LIBERO_ENV_INIT_LOCK_TIMEOUT_S", 300.0),
            )
        ),
        env_gl_fallback_backend=str(
            getattr(
                config,
                "env_gl_fallback_backend",
                os.environ.get("MERL_LIBERO_GL_FALLBACK", "none"),
            )
            or ""
        ),
        env_force_gl_fallback_after_crash=str(
            getattr(
                config,
                "env_force_gl_fallback_after_crash",
                os.environ.get("MERL_LIBERO_FORCE_FALLBACK_AFTER_CRASH", "true"),
            )
            or "true"
        ),
    )


def _as_bool_flag(value, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return default


def _normalize_gl_backend(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_x_display_num(value: Any, default: int = 99) -> int:
    text = str(value or "").strip()
    if text.startswith(":"):
        text = text[1:]
    text = text.split(".", 1)[0]
    try:
        return int(text)
    except (TypeError, ValueError):
        return int(default)


def _resolve_dist_rank_for_display() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    for key in ("LOCAL_RANK", "RANK", "MERL_ACTOR_RANK"):
        raw_value = os.environ.get(key)
        if raw_value is None:
            continue
        try:
            return int(raw_value)
        except ValueError:
            continue
    return 0


def _resolve_libero_worker_display(backend: str) -> str:
    default_display = str(os.environ.get("DISPLAY") or ":99")
    if _normalize_gl_backend(backend) != "glx":
        return default_display

    mode = str(os.environ.get("MERL_XVFB_DISPLAY_MODE", "shared") or "shared").lower()
    if mode not in {"per_actor", "per_rank"}:
        return default_display

    base_display = _parse_x_display_num(
        os.environ.get("MERL_XVFB_BASE_DISPLAY", default_display)
    )
    display_count = int(os.environ.get("MERL_XVFB_DISPLAY_COUNT", "0") or 0)
    rank = _resolve_dist_rank_for_display()
    if display_count > 0:
        rank = rank % display_count
    return f":{base_display + rank}"


def _split_cuda_visible_devices(raw_value: Any = None) -> List[str]:
    if raw_value is None:
        raw_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    raw_value = str(raw_value or "").strip()
    if not raw_value:
        return []
    return [
        item.strip()
        for item in raw_value.split(",")
        if item.strip() and item.strip().lower() not in {"none", "-1"}
    ]


def _parse_non_negative_int(value: Any) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _resolve_libero_render_cuda_visible_devices(config=None, backend: str = "") -> str:
    raw_value = (
        getattr(config, "env_render_cuda_visible_devices", None)
        if config is not None
        else None
    )
    if raw_value is None:
        raw_value = os.environ.get("MERL_LIBERO_RENDER_CUDA_VISIBLE_DEVICES")
    if raw_value is None and _normalize_gl_backend(backend) == "egl":
        raw_value = os.environ.get("MERL_GLOBAL_CUDA_VISIBLE_DEVICES")
    return str(raw_value or "").strip()


def _resolve_mujoco_egl_device_id(
    config=None, visible_devices_override: Any = None
) -> str:
    raw_value = (
        getattr(config, "mujoco_egl_device_id", None) if config is not None else None
    )
    if raw_value is None:
        raw_value = os.environ.get("MERL_LIBERO_EGL_DEVICE_ID", "auto")
    from verl.utils.libero_runtime import resolve_egl_device_id
    return resolve_egl_device_id(raw_value, visible_devices_override or None)


def _resolve_libero_env_lock_path(config, display: str) -> str:
    base_path = str(
        getattr(
            config,
            "env_init_lock_path",
            os.environ.get(
                "MERL_LIBERO_ENV_INIT_LOCK",
                _DEFAULT_LIBERO_ENV_INIT_LOCK_PATH,
            ),
        )
    )
    scope = (
        str(os.environ.get("MERL_LIBERO_ENV_INIT_LOCK_SCOPE", "global") or "global")
        .strip()
        .lower()
    )
    if scope not in {"display", "per_display"}:
        return base_path

    display_num = _parse_x_display_num(display)
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".lock"
    return f"{root}_{display_num}{ext}"


def _get_libero_primary_backend(config) -> str:
    return (
        _normalize_gl_backend(
            getattr(config, "mujoco_gl", os.environ.get("MUJOCO_GL", "glx"))
        )
        or "glx"
    )


def _get_libero_fallback_backend(config) -> Optional[str]:
    fallback = _normalize_gl_backend(
        getattr(
            config,
            "env_gl_fallback_backend",
            os.environ.get("MERL_LIBERO_GL_FALLBACK", "none"),
        )
    )
    if fallback in {"", "none", "null", "false", "0", "off", "disable", "disabled"}:
        return None
    if fallback == _get_libero_primary_backend(config):
        return None
    return fallback


def _set_libero_backend_on_config(config, backend: str) -> None:
    if config is None:
        return
    backend = _normalize_gl_backend(backend) or "glx"
    pyopengl_platform = "glx" if backend == "glx" else backend
    with contextlib.suppress(Exception):
        setattr(config, "mujoco_gl", backend)
    with contextlib.suppress(Exception):
        setattr(config, "pyopengl_platform", pyopengl_platform)


@contextlib.contextmanager
def _libero_child_start_env(config, backend: str):
    backend = _normalize_gl_backend(backend) or _get_libero_primary_backend(config)
    pyopengl_platform = "glx" if backend == "glx" else backend
    display = str(
        getattr(config, "display", os.environ.get("DISPLAY", ":99"))
        if config is not None
        else os.environ.get("DISPLAY", ":99")
    )
    render_cuda_visible_devices = str(
        getattr(config, "env_render_cuda_visible_devices", "")
        if config is not None
        else ""
    ).strip()
    if backend == "egl" and not render_cuda_visible_devices:
        render_cuda_visible_devices = str(
            os.environ.get("MERL_LIBERO_RENDER_CUDA_VISIBLE_DEVICES")
            or os.environ.get("MERL_GLOBAL_CUDA_VISIBLE_DEVICES")
            or ""
        ).strip()

    updates = {
        "DISPLAY": display,
        "MUJOCO_GL": backend,
        "PYOPENGL_PLATFORM": pyopengl_platform,
        "MERL_LIBERO_ENV_BACKEND": backend,
    }
    removals = []

    if backend == "egl":
        if render_cuda_visible_devices:
            updates["CUDA_VISIBLE_DEVICES"] = render_cuda_visible_devices
        egl_device_id = _resolve_mujoco_egl_device_id(
            config,
            visible_devices_override=render_cuda_visible_devices or None,
        )
        if egl_device_id:
            updates["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_id)
        else:
            removals.append("MUJOCO_EGL_DEVICE_ID")
    else:
        removals.append("MUJOCO_EGL_DEVICE_ID")

    if backend == "glx" and _as_bool_flag(os.environ.get("MERL_GLX_SOFTWARE"), False):
        updates["LIBGL_ALWAYS_SOFTWARE"] = "1"
        updates["LIBGL_DRI3_DISABLE"] = "1"
        updates["__GLX_VENDOR_LIBRARY_NAME"] = "mesa"
    elif backend == "glx":
        removals.extend(
            ["LIBGL_ALWAYS_SOFTWARE", "LIBGL_DRI3_DISABLE", "__GLX_VENDOR_LIBRARY_NAME"]
        )

    old_values = {key: os.environ.get(key) for key in set(updates) | set(removals)}
    try:
        for key in removals:
            os.environ.pop(key, None)
        for key, value in updates.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _apply_libero_runtime_config_to_env(config) -> Dict[str, Any]:
    backend = _get_libero_primary_backend(config)
    pyopengl_platform = _normalize_gl_backend(
        getattr(config, "pyopengl_platform", "glx" if backend == "glx" else backend)
    )
    if not pyopengl_platform:
        pyopengl_platform = "glx" if backend == "glx" else backend

    display = str(getattr(config, "display", os.environ.get("DISPLAY", ":99")) or ":99")
    os.environ["DISPLAY"] = display
    os.environ["MUJOCO_GL"] = backend
    os.environ["PYOPENGL_PLATFORM"] = pyopengl_platform
    render_cuda_visible_devices = str(
        getattr(config, "env_render_cuda_visible_devices", "") or ""
    ).strip()
    if backend == "egl" and render_cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = render_cuda_visible_devices
    if backend == "egl":
        egl_device_id = _resolve_mujoco_egl_device_id(
            config,
            visible_devices_override=render_cuda_visible_devices or None,
        )
        if egl_device_id:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_id)
        else:
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    else:
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    if getattr(config, "env_init_lock_enable", None) is not None:
        os.environ["MERL_LIBERO_ENV_INIT_LOCK_ENABLE"] = str(
            config.env_init_lock_enable
        )
    if getattr(config, "env_init_lock_path", None):
        os.environ["MERL_LIBERO_ENV_INIT_LOCK"] = str(config.env_init_lock_path)
    if getattr(config, "env_init_lock_timeout_s", None) is not None:
        os.environ["MERL_LIBERO_ENV_INIT_LOCK_TIMEOUT_S"] = str(
            config.env_init_lock_timeout_s
        )
    if backend == "glx" and str(
        os.environ.get("MERL_GLX_SOFTWARE", "false")
    ).strip().lower() in {"1", "true", "yes", "on"}:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["LIBGL_DRI3_DISABLE"] = "1"
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "mesa"
    elif backend == "glx":
        os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
        os.environ.pop("LIBGL_DRI3_DISABLE", None)
        os.environ.pop("__GLX_VENDOR_LIBRARY_NAME", None)
    return configure_libero_runtime_env(force_headless=True)


def _is_native_gl_worker_crash(process) -> bool:
    return int(process.exitcode or 0) in _NATIVE_GL_CRASH_EXITCODES


def _libero_env_init_lock_enabled(config) -> bool:
    raw_value = (
        str(
            getattr(
                config,
                "env_init_lock_enable",
                os.environ.get("MERL_LIBERO_ENV_INIT_LOCK_ENABLE", "auto"),
            )
            or "auto"
        )
        .strip()
        .lower()
    )
    if raw_value in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if raw_value in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True

    backend = (
        str(getattr(config, "mujoco_gl", os.environ.get("MUJOCO_GL", "")) or "")
        .strip()
        .lower()
    )
    platform = (
        str(
            getattr(
                config, "pyopengl_platform", os.environ.get("PYOPENGL_PLATFORM", "")
            )
            or ""
        )
        .strip()
        .lower()
    )
    return backend == "glx" or platform == "glx"


class _InterprocessFileLock:
    def __init__(
        self,
        path: str,
        *,
        timeout_s: float = 300.0,
        poll_s: float = 0.1,
        label: str = "file lock",
    ) -> None:
        self.path = path
        self.timeout_s = max(float(timeout_s), 1.0)
        self.poll_s = max(float(poll_s), 0.01)
        self.label = label
        self._file_obj = None

    def __enter__(self):
        if fcntl is None:
            return self

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._file_obj = open(self.path, "a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fcntl.flock(self._file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._file_obj.seek(0)
                self._file_obj.truncate()
                self._file_obj.write(f"pid={os.getpid()} label={self.label}\n")
                self._file_obj.flush()
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out after {self.timeout_s:.1f}s waiting for {self.label}: {self.path}"
                    )
                time.sleep(self.poll_s)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file_obj is None or fcntl is None:
            return
        with contextlib.suppress(Exception):
            fcntl.flock(self._file_obj.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(Exception):
            self._file_obj.close()
        self._file_obj = None


def _create_libero_env_with_runtime_lock(
    task,
    model_family: str,
    *,
    resolution: int,
    config,
    emit_progress: Optional[Callable[..., None]] = None,
    retry: Optional[int] = None,
):
    runtime_env = _apply_libero_runtime_config_to_env(config)

    def _emit(stage: str, **stage_extra) -> None:
        if emit_progress is None:
            return
        extra = {}
        if retry is not None:
            extra["retry"] = retry
        extra.update(stage_extra)
        emit_progress(stage, **extra)

    _emit(
        "env_runtime",
        display=str(os.environ.get("DISPLAY") or ""),
        backend=str(runtime_env.get("backend") or ""),
        egl_device=str(os.environ.get("MUJOCO_EGL_DEVICE_ID") or ""),
        cuda_visible=str(os.environ.get("CUDA_VISIBLE_DEVICES") or ""),
        libgl_software=str(os.environ.get("LIBGL_ALWAYS_SOFTWARE") or ""),
        glx_vendor=str(os.environ.get("__GLX_VENDOR_LIBRARY_NAME") or ""),
    )

    if not _libero_env_init_lock_enabled(config):
        env, task_description = get_libero_env(
            task, model_family, resolution=resolution
        )
        _emit("env_created")
        return env, task_description

    lock_path = str(
        getattr(
            config,
            "env_init_lock_path",
            os.environ.get(
                "MERL_LIBERO_ENV_INIT_LOCK", _DEFAULT_LIBERO_ENV_INIT_LOCK_PATH
            ),
        )
    )
    lock_timeout_s = float(
        getattr(
            config,
            "env_init_lock_timeout_s",
            os.environ.get("MERL_LIBERO_ENV_INIT_LOCK_TIMEOUT_S", 300.0),
        )
    )
    _emit("env_create_wait_lock")
    with _InterprocessFileLock(
        lock_path,
        timeout_s=lock_timeout_s,
        label=f"LIBERO env init pid={os.getpid()}",
    ):
        _emit("env_create_lock_acquired")
        env, task_description = get_libero_env(
            task, model_family, resolution=resolution
        )
    _emit("env_created")
    return env, task_description


def _get_env_mp_context(config=None):
    method = str(
        os.environ.get(
            "MERL_ENV_MP_START_METHOD",
            (
                getattr(config, "env_mp_start_method", "spawn")
                if config is not None
                else "spawn"
            ),
        )
        or "spawn"
    ).lower()
    if method not in {"spawn", "forkserver", "fork"}:
        method = "spawn"
    try:
        return multiprocessing.get_context(method)
    except ValueError:
        return multiprocessing.get_context("spawn")


def _process_exit_summary(process) -> str:
    exitcode = process.exitcode
    if exitcode is None:
        return "still-running"
    if exitcode < 0:
        return f"signal {-exitcode}"
    return f"exitcode {exitcode}"


def _get_worker_message(output_queue, process, timeout: float, context: str):
    deadline = time.monotonic() + timeout
    last_progress = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            progress_suffix = (
                f"; last_progress={last_progress}" if last_progress is not None else ""
            )
            raise RuntimeError(
                f"{context} timed out after {timeout:.1f}s; "
                f"worker status={_process_exit_summary(process)}{progress_suffix}"
            )
        try:
            message = output_queue.get(timeout=min(1.0, remaining))
        except queue.Empty as exc:
            if process.exitcode is not None:
                raise RuntimeError(
                    f"{context} worker exited before sending a queue message "
                    f"({ _process_exit_summary(process) }). This usually indicates a native "
                    "MuJoCo/OpenGL crash; use MERL_ENV_MP_START_METHOD=spawn, keep "
                    "MERL_LIBERO_ENV_BACKEND=egl for LIBERO child workers, and verify "
                    "CUDA_VISIBLE_DEVICES/MUJOCO_EGL_DEVICE_ID are consistent."
                ) from exc
            continue

        if isinstance(message, dict) and message.get("type") == "progress":
            stage = str(message.get("stage") or "unknown")
            if stage != last_progress:
                progress_bits = []
                for key in (
                    "display",
                    "backend",
                    "egl_device",
                    "cuda_visible",
                    "libgl_software",
                    "glx_vendor",
                    "retry",
                    "step",
                    "total",
                ):
                    if key in message:
                        progress_bits.append(f"{key}={message[key]}")
                suffix = f" ({', '.join(progress_bits)})" if progress_bits else ""
                print(f"[{context}] progress: {stage}{suffix}", flush=True)
            last_progress = stage
            deadline = time.monotonic() + timeout
            continue

        return message


def _close_mp_queue(queue_obj) -> None:
    if queue_obj is None:
        return

    with contextlib.suppress(Exception):
        queue_obj.cancel_join_thread()
    with contextlib.suppress(Exception):
        queue_obj.close()


def _shutdown_env_workers(
    input_queues,
    processes,
    output_queues=None,
    join_timeout: float = 5.0,
) -> None:
    for q in input_queues:
        if q is None:
            continue
        with contextlib.suppress(Exception):
            q.put(None)
    for p in processes:
        if p is None:
            continue
        with contextlib.suppress(Exception):
            p.join(timeout=join_timeout)
            if p.is_alive():
                p.terminate()
                p.join(timeout=join_timeout)

    for q in input_queues:
        _close_mp_queue(q)
    for q in output_queues or []:
        _close_mp_queue(q)


def _launch_libero_env_worker_with_retry(
    *,
    mp_ctx,
    target,
    args_factory: Callable[[Any, Any], tuple],
    worker_label: str,
    init_timeout_s: float,
    max_attempts: int,
    worker_config=None,
):
    last_error = None
    primary_backend = _get_libero_primary_backend(worker_config)
    fallback_backend = _get_libero_fallback_backend(worker_config)
    force_fallback_after_crash = _as_bool_flag(
        getattr(
            worker_config,
            "env_force_gl_fallback_after_crash",
            os.environ.get("MERL_LIBERO_FORCE_FALLBACK_AFTER_CRASH", "true"),
        ),
        default=True,
    )
    use_fallback = bool(_LIBERO_GL_FALLBACK_STATE.get("force_fallback"))

    for attempt in range(1, max_attempts + 1):
        active_backend = (
            fallback_backend if (use_fallback and fallback_backend) else primary_backend
        )
        if worker_config is not None:
            _set_libero_backend_on_config(worker_config, active_backend)
        active_display = str(
            getattr(worker_config, "display", os.environ.get("DISPLAY", ""))
            if worker_config is not None
            else os.environ.get("DISPLAY", "")
        )
        active_lock = str(
            getattr(worker_config, "env_init_lock_path", "")
            if worker_config is not None
            else ""
        )
        print(
            f"[{worker_label}] init attempt {attempt}/{max_attempts} start: "
            f"DISPLAY={active_display}, MUJOCO_GL={active_backend}, lock={active_lock}",
            flush=True,
        )
        input_q = mp_ctx.Queue()
        output_q = mp_ctx.Queue()
        process = mp_ctx.Process(target=target, args=args_factory(input_q, output_q))
        process.daemon = True
        with _libero_child_start_env(worker_config, active_backend):
            process.start()

        try:
            init_data = _get_worker_message(
                output_q,
                process,
                timeout=init_timeout_s,
                context=f"{worker_label} init attempt={attempt}",
            )
        except Exception as exc:
            last_error = exc
            native_gl_crash = _is_native_gl_worker_crash(process)
            _shutdown_env_workers([input_q], [process], [output_q], join_timeout=5.0)
            if (
                native_gl_crash
                and fallback_backend
                and active_backend == primary_backend
            ):
                use_fallback = True
                if force_fallback_after_crash:
                    _LIBERO_GL_FALLBACK_STATE["force_fallback"] = True
                print(
                    f"[{worker_label}] native crash with MUJOCO_GL={active_backend}; "
                    f"retrying with MUJOCO_GL={fallback_backend}.",
                    flush=True,
                )
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"{worker_label} init failed after {max_attempts} attempts: {exc}"
                ) from exc
            print(
                f"[{worker_label}] init attempt {attempt}/{max_attempts} failed "
                f"(MUJOCO_GL={active_backend}): {exc}; respawning worker.",
                flush=True,
            )
            continue

        if init_data.get("type") == "error":
            error_text = init_data.get("error") or "unknown error"
            traceback_text = str(init_data.get("traceback") or "").strip()
            rendered = (
                error_text if not traceback_text else f"{error_text}\n{traceback_text}"
            )
            last_error = RuntimeError(rendered)
            _shutdown_env_workers([input_q], [process], [output_q], join_timeout=5.0)
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"{worker_label} init failed after {max_attempts} attempts: {rendered}"
                )
            print(
                f"[{worker_label}] init attempt {attempt}/{max_attempts} returned error "
                f"(MUJOCO_GL={active_backend}): {error_text}; respawning worker.",
                flush=True,
            )
            continue

        return process, input_q, output_q, init_data

    raise RuntimeError(
        f"{worker_label} init failed after {max_attempts} attempts: {last_error}"
    )


def _libero_env_service_worker(config, input_queue, output_queue):
    env = None
    current_task_key = None
    active = False
    complete = False
    finish_step = 0
    max_steps = 0
    is_valid = True
    last_img = None

    def _emit(stage: str, **extra):
        message = {"type": "progress", "stage": stage}
        message.update(extra)
        with contextlib.suppress(Exception):
            output_queue.put(message)

    def _close_env():
        nonlocal env, current_task_key, active
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        env = None
        current_task_key = None
        active = False

    def _build_init_response(command: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal env, current_task_key, active, complete, finish_step, max_steps, is_valid, last_img
        task_name = command["task_name"]
        task_id = int(command["task_id"])
        trial_id = int(command["trial_id"])
        is_valid = bool(command.get("is_valid", True))
        max_steps = int(command.get("max_steps", 512))
        task_description = str(command.get("ood_task_description", "") or "")
        resolution = int(command.get("resolution", 256))

        benchmark_dict = _get_libero_pro_benchmark_dict(config)
        task_suite = benchmark_dict[task_name]()
        task = task_suite.get_task(task_id)
        if not task_description.strip():
            task_description = str(task.language)
        initial_states = load_task_init_states(task_suite, task_id)
        initial_state = initial_states[trial_id]
        task_key = (task_name, task_id)
        _emit("task_loaded", task_name=task_name, task_id=task_id, trial_id=trial_id)

        if env is None or current_task_key != task_key:
            _close_env()
            env, _ = _create_libero_env_with_runtime_lock(
                task,
                config.model_family,
                resolution=resolution,
                config=config,
                emit_progress=_emit,
                retry=1,
            )
            current_task_key = task_key

        _emit(
            "env_reset_start", task_name=task_name, task_id=task_id, trial_id=trial_id
        )
        env.reset()
        _emit("env_reset_done", task_name=task_name, task_id=task_id, trial_id=trial_id)
        obs = env.set_init_state(initial_state)
        _emit(
            "set_init_state_done",
            task_name=task_name,
            task_id=task_id,
            trial_id=trial_id,
        )

        valid_images = []
        dummy_action = get_libero_dummy_action(config.model_family)
        for t in range(int(getattr(config, "num_steps_wait", 10))):
            obs, _, _, _ = env.step(dummy_action)
            if t + 1 == int(getattr(config, "num_steps_wait", 10)) or (t + 1) % 2 == 0:
                _emit(
                    "warmup",
                    task_name=task_name,
                    task_id=task_id,
                    trial_id=trial_id,
                    step=t + 1,
                    total=int(getattr(config, "num_steps_wait", 10)),
                )

        img = obs["agentview_image"][::-1, ::-1].copy()
        last_img = img
        if is_valid:
            valid_images.append(img)
        active = True
        complete = False
        finish_step = 0

        return {
            "type": "init",
            "task_description": task_description,
            "obs": obs,
            "image": img,
            "valid_images": valid_images.copy(),
            "env_images": [img],
            "env_dones": [False],
            "normed_actions": [np.array(dummy_action)],
            "active": True,
            "complete": False,
            "finish_step": 0,
            "task_file_name": f"{task_name}_task_{task_id}_trial_{trial_id}",
        }

    def _build_step_response(action) -> Dict[str, Any]:
        nonlocal active, complete, finish_step, last_img
        if env is None or not active:
            return {
                "type": "terminate",
                "active": False,
                "complete": complete,
                "finish_step": finish_step,
            }

        step_images = []
        env_images = []
        env_dones = []
        normed_actions = []
        obs = None
        try:
            for i in range(len(action)):
                normalized_action = normalize_gripper_action(action[i], binarize=True)
                inverted_action = invert_gripper_action(normalized_action)
                normed_actions.append(inverted_action)
                obs, _, done, _ = env.step(inverted_action.tolist())
                img = obs["agentview_image"][::-1, ::-1].copy()
                last_img = img
                if is_valid:
                    step_images.append(img)
                env_images.append(img)
                env_dones.append(done)
                finish_step += 1
                if done or finish_step >= max_steps:
                    active = False
                    complete = bool(done)
                    break
        except Exception as exc:
            active = False
            complete = False
            return {
                "type": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "active": False,
                "complete": False,
                "finish_step": finish_step,
                "env_images": env_images,
                "env_dones": env_dones,
                "valid_images": step_images if is_valid else [],
            }

        return {
            "type": "step",
            "obs": obs,
            "image": env_images[-1] if env_images else last_img,
            "valid_images": step_images.copy() if is_valid else [],
            "env_images": env_images.copy(),
            "env_dones": env_dones.copy(),
            "normed_actions": normed_actions.copy(),
            "active": active,
            "complete": complete,
            "finish_step": finish_step,
        }

    try:
        _apply_libero_runtime_config_to_env(config)
        from verl.utils.libero_path import ensure_libero_pro_root

        ensure_libero_pro_root(
            evaluation_config_path=getattr(config, "libero_pro_eval_config_path", None)
        )
        output_queue.put({"type": "ready"})
        while True:
            command = input_queue.get()
            if command is None or (
                isinstance(command, dict) and command.get("type") == "close"
            ):
                _close_env()
                output_queue.put({"type": "terminate", "active": False})
                break
            if not isinstance(command, dict):
                output_queue.put(_build_step_response(command))
                continue
            command_type = str(command.get("type") or "").lower()
            if command_type == "reset":
                output_queue.put(_build_init_response(command))
            elif command_type == "step":
                output_queue.put(_build_step_response(command.get("action")))
            else:
                output_queue.put(
                    {
                        "type": "error",
                        "error": f"unknown service command: {command_type}",
                    }
                )
    except Exception as exc:
        with contextlib.suppress(Exception):
            output_queue.put(
                {
                    "type": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "active": False,
                    "complete": False,
                    "finish_step": finish_step,
                }
            )
    finally:
        _close_env()


class _LiberoEnvServiceClient:
    def __init__(
        self,
        *,
        mp_ctx,
        worker_config,
        label: str,
        init_timeout_s: float,
        step_timeout_s: float,
    ):
        self.mp_ctx = mp_ctx
        self.worker_config = worker_config
        self.label = label
        self.init_timeout_s = float(init_timeout_s)
        self.step_timeout_s = float(step_timeout_s)
        self.input_queue = None
        self.output_queue = None
        self.process = None
        self.busy = False
        self.start()

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def start(self) -> None:
        self.input_queue = self.mp_ctx.Queue()
        self.output_queue = self.mp_ctx.Queue()
        self.process = self.mp_ctx.Process(
            target=_libero_env_service_worker,
            args=(self.worker_config, self.input_queue, self.output_queue),
        )
        self.process.daemon = True
        backend = _get_libero_primary_backend(self.worker_config)
        with _libero_child_start_env(self.worker_config, backend):
            self.process.start()
        ready = _get_worker_message(
            self.output_queue,
            self.process,
            timeout=self.init_timeout_s,
            context=f"{self.label} service start",
        )
        if ready.get("type") == "error":
            raise RuntimeError(ready.get("error") or "LIBERO env service start failed")
        if ready.get("type") != "ready":
            raise RuntimeError(f"unexpected LIBERO env service start message: {ready}")

    def reset(
        self,
        *,
        task_name: str,
        task_id: int,
        trial_id: int,
        is_valid: bool,
        global_steps: int,
        max_steps: int,
        ood_task_description: str,
        resolution: int = 256,
    ) -> Dict[str, Any]:
        command = {
            "type": "reset",
            "task_name": task_name,
            "task_id": int(task_id),
            "trial_id": int(trial_id),
            "is_valid": bool(is_valid),
            "global_steps": int(global_steps),
            "max_steps": int(max_steps),
            "ood_task_description": str(ood_task_description or ""),
            "resolution": int(resolution),
        }
        self.input_queue.put(command)
        data = _get_worker_message(
            self.output_queue,
            self.process,
            timeout=self.init_timeout_s,
            context=f"{self.label} reset task={task_name} trial={trial_id}",
        )
        if data.get("type") == "error":
            error_text = data.get("error") or "LIBERO env service reset failed"
            traceback_text = str(data.get("traceback") or "").strip()
            raise RuntimeError(
                error_text if not traceback_text else f"{error_text}\n{traceback_text}"
            )
        return data

    def submit_step(self, action) -> None:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        self.input_queue.put({"type": "step", "action": action})

    def recv_step(self, *, timeout: float, context: str) -> Dict[str, Any]:
        return _get_worker_message(
            self.output_queue,
            self.process,
            timeout=float(timeout),
            context=context,
        )

    def close(self, *, force: bool = False) -> None:
        if not force and self.input_queue is not None:
            with contextlib.suppress(Exception):
                self.input_queue.put({"type": "close"})
                _get_worker_message(
                    self.output_queue,
                    self.process,
                    timeout=5.0,
                    context=f"{self.label} service close",
                )
        _shutdown_env_workers(
            [self.input_queue],
            [self.process],
            [self.output_queue],
            join_timeout=5.0,
        )
        self.input_queue = None
        self.output_queue = None
        self.process = None
        self.busy = False


# ================ Utensils ================
def float01_to_uint8(im: np.ndarray) -> np.ndarray:
    """
    im: float32, [0,1], HWC
    """
    im = np.clip(im, 0.0, 1.0)
    return (im * 255.0).astype(np.uint8)


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.
    """
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(
        tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,)
    )
    new_widths = tf.reshape(
        tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,)
    )

    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (224, 224)
    )

    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Image) -> Image:
    batch_size = 1
    crop_scale = 0.9

    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, crop_scale, batch_size)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    image = Image.fromarray(image.numpy())
    image = image.convert("RGB")
    return image


def get_inner_module(m):
    return m.module if hasattr(m, "module") else m


def encode_img_to_latent(
    model: CtrlWorld, img: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    img: (T, 3, H, W)
    return latents: (T, 4, 32, 32)
    """
    model = get_inner_module(model)
    vae = get_inner_module(model.vae)
    img = img.to(device)
    with torch.no_grad():
        latent = vae.encode(img).latent_dist.sample()
        latent = latent.mul_(vae.config.scaling_factor).cpu()
    return latent


def resize_to_libero_image(img: np.ndarray, resize_size: Union[int, Tuple] = None):
    if not resize_size:
        resize_size = (256, 256)
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img: np.ndarray = resize_image(img, resize_size)
    return img


def extract_images_from_episode(episode):
    """episode: list of transitions (obs, action, reward, done, info)
    Return list of agentview_image arrays (np.uint8) where possible.
    """
    imgs = []
    for tr in episode:
        obs = tr[0]
        if isinstance(obs, dict) and "agentview_image" in obs:
            imgs.append(obs["agentview_image"])
    return imgs


def extract_dones_from_episode(episode):
    dones = []
    for tr in episode:
        dones.append(bool(tr[3]))
    return dones


# ================ Robotwin-specific functions ================
def normalize_proprio(proprio, norm_stats):
    """Normalize proprioception data for Robotwin."""
    if ACTION_PROPRIO_NORMALIZATION_TYPE == "bounds":
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(
            norm_stats["min"]
        )
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == "bounds_q99":
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(
            norm_stats["q01"]
        )
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )
    return normalized_proprio


def get_robotwin2_task(task_name, config):
    """Get robotwin 2.0 task"""
    robotwin2_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "utils", "envs", "robotwin2"
    )
    if robotwin2_path not in sys.path:
        sys.path.append(robotwin2_path)

    robotwin2_utils_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "utils",
        "envs",
        "robotwin2",
        "description",
        "utils",
    )
    if robotwin2_utils_path not in sys.path:
        sys.path.append(robotwin2_utils_path)

    from envs import CONFIGS_PATH

    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit(f"No Task: {task_name}")

    task_config = config.get("twin2_task_config", "demo_randomized")
    config_file = os.path.join(robotwin2_path, f"task_config/{task_config}.yml")

    with open(config_file, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = config.get("twin2_ckpt_setting", "demo_randomized")

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        return embodiment_args

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    args["eval_mode"] = True
    args["eval_video_log"] = False
    args["render_freq"] = 0
    args["instruction_type"] = config.get("twin2_instruction_type", "unseen")

    return env_instance, args


def encode_obs(observation):
    """Post-Process Observation for robotwin 2.0"""
    return observation


# just for robotwin
class RobotwinEnvWrapper:
    """Thread-safe wrapper for Robotwin environment (supports both 1.0 and 2.0)"""

    def __init__(self, task_name, trial_id, trial_seed, config, version="1.0"):
        self.task_name = task_name
        self.trial_id = trial_id
        self.trial_seed = trial_seed
        self.config = config
        self.version = version
        self.env = None
        self.args = None
        self.active = True
        self.complete = False
        self.finish_step = 0
        self.lock = threading.Lock()
        self.instruction = None

    def initialize(self):
        """Initialize the environment"""
        with _ENV_INIT_LOCK:
            with self.lock:
                try:
                    if self.version == "1.0":
                        print(
                            "RobotWin 2.0 fully encompasses RobotWin 1.0, therefore we prioritize support for RobotWin 2.0"
                        )
                        raise ValueError
                    else:  # 2.0
                        self.env, self.args = get_robotwin2_task(
                            self.task_name, self.config
                        )
                        self.env.setup_demo(
                            now_ep_num=self.trial_id,
                            seed=self.trial_seed,
                            is_test=True,
                            **self.args,
                        )
                        episode_info_list = [self.env.get_info()]
                except Exception as e:
                    print(f"****** IN thread: setup_demo ERROR {e} ******", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.env, self.args = get_robotwin2_task(
                        self.task_name, self.config
                    )
                    self.env.setup_demo(
                        now_ep_num=self.trial_id,
                        seed=self.trial_seed,
                        is_test=True,
                        **self.args,
                    )
                    episode_info_list = [self.env.get_info()]

                from generate_episode_instructions import generate_episode_descriptions

                results = generate_episode_descriptions(
                    self.task_name, episode_info_list, 1, seed=self.trial_id
                )
                self.instruction = np.random.choice(
                    results[0][self.args["instruction_type"]]
                )
                self.env.set_instruction(instruction=self.instruction)

    def get_obs(self):
        """Get observation from environment"""
        with self.lock:
            try:
                geted_obs = self.env.get_obs()
                return geted_obs
            except Exception as e:
                print(f"****** IN thread: get_obs ERROR {e} ******", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                geted_obs = self.env.get_obs()
                return geted_obs

    def get_instruction(self):
        """Get instruction for the task"""
        with self.lock:

            return self.env.get_instruction()

    def step(self, action):
        """Execute action in environment"""
        with self.lock:
            try:

                self.env.take_action(action)
                done = self.env.eval_success

            except Exception as e:
                done = False
                error_msg = f"****** action execution ERROR: {type(e).__name__}: {str(e)} ******"
                print(error_msg, flush=True)
                traceback.print_exc()

            try:
                obs = self.env.get_obs()
                obs = encode_obs(obs)
            except Exception as e:
                print(f"****** env.get_obs ERROR {e} ******", flush=True)
                obs = None

            self.finish_step += action.shape[0]

            if done or self.finish_step >= self.env.step_lim:
                self.active = False
                self.complete = done

            return obs, done

    def close(self):
        """Close the environment"""
        with self.lock:
            if self.env is not None:
                try:
                    self.env.close_env(clear_cache=True)
                except Exception as e:
                    print(f"******IN env.close ERROR {e} ******", flush=True)


# ================ Libero-specific functions ================
# pure env
def env_worker(
    task_name,
    task_id,
    trial_id,
    config,
    input_queue,
    output_queue,
    is_valid,
    global_steps,
    max_steps,
    ood_task_description,
):
    """Worker process for Libero environments"""
    try:
        benchmark_dict = _get_libero_pro_benchmark_dict(config)
        task_suite = benchmark_dict[task_name]()
        task = task_suite.get_task(task_id)
        initial_states = load_task_init_states(task_suite, task_id)
        initial_state = initial_states[trial_id]
        task_description = ood_task_description

        env = None
        last_error = None
        max_init_retry = int(getattr(config, "env_init_max_retry", 1))
        for retry_idx in range(max_init_retry):
            try:
                env, _ = _create_libero_env_with_runtime_lock(
                    task,
                    config.model_family,
                    resolution=256,
                    config=config,
                )
                break
            except Exception as error:
                last_error = error
                print(
                    f"*** env initialization failed ({retry_idx + 1}/{max_init_retry}): "
                    f"{type(error).__name__}: {error} ***",
                    flush=True,
                )
                traceback.print_exc()
                if env is not None:
                    try:
                        env.close()
                    except Exception as close_error:
                        print(f"error when close the env: {close_error}")
                torch.cuda.empty_cache()
                gc.collect()
                print("gc collect finish")
        else:
            raise RuntimeError(
                f"Failed to initialize LIBERO_PRO env after {max_init_retry} retries for "
                f"task={task_name}, task_id={task_id}, trial_id={trial_id}"
            ) from last_error

        env.reset()
        obs = env.set_init_state(initial_state)

        t = 0
        valid_images = []
        while t < config.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(config.model_family))
            t += 1

        init_env_images = []
        init_env_dones = []
        if is_valid:
            img = obs["agentview_image"][::-1, ::-1].copy()
            valid_images.append(img)
            init_env_images.append(img)
            init_env_dones.append(False)

        output_queue.put(
            {
                "type": "init",
                "obs": obs,
                "task_description": task_description,
                "valid_images": valid_images.copy(),
                "env_images": init_env_images,
                "env_dones": init_env_dones,
                "task_file_name": f"{task_name}_task_{task_id}_trial_{trial_id}",
                "active": True,
                "complete": False,
                "finish_step": 0,
            }
        )

        active = True
        complete = False
        finish_step = 0

        while True:
            action = input_queue.get()
            if action is None:
                env.close()
                output_queue.put({"type": "terminate"})
                break

            step_images = []
            step_dones = []
            for i in range(len(action)):
                a = action[i]
                normalized_action = normalize_gripper_action(a, binarize=True)
                inverted_action = invert_gripper_action(normalized_action)
                obs, reward, done, info = env.step(inverted_action.tolist())

                if is_valid:
                    img = obs["agentview_image"][::-1, ::-1].copy()
                    step_images.append(img)
                    step_dones.append(bool(done))

                finish_step += 1
                if done or finish_step >= max_steps:
                    active = False
                    complete = done
                    break

            output_data = {
                "type": "step",
                "obs": obs,
                "active": active,
                "complete": complete,
                "finish_step": finish_step,
                "valid_images": step_images.copy() if is_valid else [],
                "env_images": step_images.copy() if is_valid else [],
                "env_dones": step_dones.copy() if is_valid else [],
            }
            output_queue.put(output_data)
    except Exception as error:
        output_queue.put(
            {
                "type": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "task_name": task_name,
                "task_id": task_id,
                "trial_id": trial_id,
            }
        )
        return


# todo: wrong! pure world model mode
def env_worker_wm(
    world_model: CtrlWorld,
    wm_args,
    rm_threshold: float,
    device,
    task_name,
    task_id,
    trial_id,
    config,
    input_queue,
    output_queue,
    is_valid,
    global_steps,
    max_steps,
):
    """Worker process for Libero environments with world model"""

    def emit_progress(stage: str, **extra):
        message = {
            "type": "progress",
            "stage": stage,
            "task_name": task_name,
            "task_id": task_id,
            "trial_id": trial_id,
        }
        message.update(extra)
        try:
            output_queue.put(message)
        except Exception:
            pass

    try:
        benchmark_dict = _get_libero_pro_benchmark_dict(config)
        task_suite = benchmark_dict[task_name]()
        task = task_suite.get_task(task_id)
        initial_states = load_task_init_states(task_suite, task_id)
        initial_state = initial_states[trial_id]
        emit_progress("task_loaded")

        env = None
        last_error = None
        max_init_retry = int(getattr(config, "env_init_max_retry", 1))
        for retry_idx in range(max_init_retry):
            try:
                env, task_description = _create_libero_env_with_runtime_lock(
                    task,
                    config.model_family,
                    resolution=256,
                    config=config,
                    emit_progress=emit_progress,
                    retry=retry_idx + 1,
                )
                break
            except Exception as error:
                last_error = error
                print(
                    f"*** env initialization failed ({retry_idx + 1}/{max_init_retry}): "
                    f"{type(error).__name__}: {error} ***",
                    flush=True,
                )
                traceback.print_exc()
                if env is not None:
                    try:
                        env.close()
                    except Exception as close_error:
                        print(f"error when close the env: {close_error}")
                torch.cuda.empty_cache()
                gc.collect()
                print("gc collect finish")
        else:
            raise RuntimeError(
                f"Failed to initialize WM env after {max_init_retry} retries for "
                f"task={task_name}, task_id={task_id}, trial_id={trial_id}"
            ) from last_error

        env.reset()
        obs = env.set_init_state(initial_state)
        t = 0
        valid_images = []
        while t < config.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(config.model_family))
            t += 1
        img = obs["agentview_image"][::-1, ::-1].copy()
        valid_images.append(img)
        output_queue.put(
            {
                "type": "init",
                "obs": obs,
                "image": img,
                "task_description": task_description,
                "valid_images": valid_images.copy(),
                "env_images": [img],
                "env_dones": [False],
                "task_file_name": f"{task_name}_task_{task_id}_trial_{trial_id}",
                "active": True,
                "complete": False,
                "finish_step": 0,
            }
        )
        H, W = img.shape[:2]
        env.close()

        active = True
        complete = False
        finish_step = 0
        while True:
            try:
                batch_input: Dict[str, Any] = input_queue.get(timeout=3)
                if batch_input is None or batch_input["action"] is None:
                    output_queue.put({"type": "terminate"})
                    break
            except queue.Empty:
                output_queue.put({"type": "terminate"})
                break

            action = batch_input["action"]
            num_future_frames = action.shape[0]
            image: torch.Tensor = batch_input["image"]
            image = image.unsqueeze(0)
            image_latent = encode_img_to_latent(world_model, image, device)
            task_description = batch_input["task_description"]

            hist_images: torch.Tensor = batch_input["hist_images"]
            hist_images_latent = encode_img_to_latent(world_model, hist_images, device)
            hist_images_latent = hist_images_latent.unsqueeze(0)

            action_latent = []
            for t in range(len(action)):
                a_t_np = action[t].numpy()
                normalized_a_t_np = normalize_gripper_action(a_t_np, binarize=True)
                inverted_a_t_np = invert_gripper_action(normalized_a_t_np)
                inverted_a_t = torch.from_numpy(inverted_a_t_np)
                a_t = inverted_a_t.unsqueeze(0).unsqueeze(0)

                a_t_latent = get_inner_module(world_model).action_encoder(
                    a_t,
                    task_description,
                    get_inner_module(world_model).tokenizer,
                    get_inner_module(world_model).text_encoder,
                    wm_args.frame_level_cond,
                )
                action_latent.append(a_t_latent)
            action_latent = torch.cat(action_latent, dim=1)

            step_images: List[np.ndarray] = []
            step_dones: List[bool] = []
            with torch.no_grad():
                pred_images_list, pred_images_latent = (
                    CtrlWorldDiffusionPipeline.__call__(
                        world_model.pipeline,
                        image=image_latent,
                        text=action_latent,
                        width=wm_args.width,
                        height=wm_args.height,
                        num_frames=num_future_frames,
                        history=hist_images_latent,
                        num_inference_steps=wm_args.num_inference_steps,
                        decode_chunk_size=num_future_frames,
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

                pred_images = torch.cat(
                    [
                        torch.from_numpy(pred_imgs_np)
                        for pred_imgs_np in pred_images_list
                    ],
                    dim=0,
                )
                pred_images = einops.rearrange(pred_images, "b_t h w c -> b_t c h w")

                action_latent_flat = einops.rearrange(action_latent, "b t d -> (b t) d")
                pred_scores = get_inner_module(
                    world_model
                ).reward_classifier.predict_score(
                    pred_images,
                    action_latent_flat,
                )

            for t in range(pred_images_list[0].shape[0]):
                pred_img_t: np.ndarray = pred_images_list[0][t]
                pred_img_t = resize_to_libero_image(pred_img_t, (H, W))
                step_images.append(pred_img_t)
                finish_step += 1

                is_success = pred_scores[t] > rm_threshold
                step_dones.append(bool(is_success))
                if is_success or finish_step >= max_steps:
                    active = False
                    complete = is_success
                    break

            output_data = {
                "type": "step",
                "obs": None,
                "image": step_images[-1],
                "active": active,
                "complete": complete,
                "finish_step": finish_step,
                "valid_images": step_images.copy() if is_valid else [],
                "env_images": step_images.copy() if is_valid else [],
                "env_dones": step_dones.copy() if is_valid else [],
            }
            output_queue.put(output_data)
    except Exception as error:
        output_queue.put(
            {
                "type": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "task_name": task_name,
                "task_id": task_id,
                "trial_id": trial_id,
            }
        )
        return


def get_action_latent(
    world_model: CtrlWorld,
    normed_action: List[np.ndarray],
    task_description: str,
    frame_level_cond: bool = False,
) -> torch.Tensor:
    wm = get_inner_module(world_model)
    device = next(wm.parameters()).device
    dtype = next(wm.parameters()).dtype

    action_latent = []
    for t in range(len(normed_action)):
        inverted_a_t_np = normed_action[t]
        inverted_a_t = torch.from_numpy(inverted_a_t_np)  # [action_dim]
        a_t = inverted_a_t.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
        a_t = a_t.to(device=device, dtype=dtype)
        a_t_latent = wm.action_encoder(
            a_t,
            task_description,
            get_inner_module(world_model).tokenizer,
            get_inner_module(world_model).text_encoder,
            frame_level_cond,
        )  # [1, 1, d]
        action_latent.append(a_t_latent)
    action_latent = torch.cat(action_latent, dim=1)  # [1, t, d]
    return action_latent


# todo: wrong! evolving mode
def env_worker_evolving(
    world_model: CtrlWorld,
    wm_args,
    rm_threshold: float,
    device,
    task_name,
    task_id,
    trial_id,
    config,
    input_queue,
    output_queue,
    is_valid,
    global_steps,
    max_steps,
    finished_evolving: bool = False,
):
    """Worker process for Libero environments with world model"""
    benchmark_dict = _get_libero_pro_benchmark_dict(config)
    task_suite = benchmark_dict[task_name]()
    task = task_suite.get_task(task_id)
    initial_states = load_task_init_states(task_suite, task_id)
    initial_state = initial_states[trial_id]

    # 1. initialization stage
    env = None
    while True:
        try:
            env, task_description = _create_libero_env_with_runtime_lock(
                task,
                config.model_family,
                resolution=256,
                config=config,
            )
            break
        except:
            print(f"*** Env initialization failed ***")
            if env is not None:
                try:
                    env.close()
                except Exception as e:
                    print(f"Error when close the env: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            print("gc collect finish")

    env.reset()
    obs = env.set_init_state(initial_state)
    t = 0
    valid_images = []
    while t < config.num_steps_wait:
        dummy_action: List[float] = get_libero_dummy_action(config.model_family)
        obs, _, _, _ = env.step(dummy_action)
        t += 1
    img = obs["agentview_image"][::-1, ::-1]  # np.array, [h, w, c]
    valid_images.append(img)
    output_queue.put(
        {
            "type": "init",
            "obs": obs,
            "image": img,
            "task_description": task_description,
            "task_file_name": f"{task_name}_task_{task_id}_trial_{trial_id}",
            "active": True,
            "complete": False,
            "finish_step": 0,
            "valid_images": valid_images.copy(),
            "normed_actions": [np.array(dummy_action)],
            "env_images": valid_images.copy(),
            "env_dones": [False],
        }
    )
    H, W = img.shape[:2]  # 224, 224
    if finished_evolving:
        env.close()

    # 2. rollout stage
    active = True
    complete = False
    finish_step = 0
    while True:
        try:
            batch_input: Dict[str, Any] = input_queue.get(timeout=3)
            if batch_input is None or batch_input["action"] is None:
                output_queue.put({"type": "terminate"})
                env.close()
                break
        except queue.Empty:
            output_queue.put({"type": "terminate"})
            env.close()
            break

        action = batch_input["action"]  # [action_chunk_size, action_dim]
        num_future_frames = action.shape[0]  # t = action_chunk_size
        image: torch.Tensor = batch_input["image"]  # [c, h, w], (192, 320)
        image = image.unsqueeze(0)  # [1, c, h, w]
        image_latent = encode_img_to_latent(world_model, image, device)  # [1, c, h, w]
        task_description: str = batch_input["task_description"]

        hist_images: torch.Tensor = batch_input["hist_images"]  # [t_h, c, h, w]
        hist_images_latent = encode_img_to_latent(world_model, hist_images, device)
        hist_images_latent = hist_images_latent.unsqueeze(0)  # [1, t_h, c, h, w]

        normed_action = []
        for t in range(len(action)):
            a_t_np = action[t].numpy()
            normalized_a_t_np = normalize_gripper_action(a_t_np, binarize=True)
            inverted_a_t_np = invert_gripper_action(normalized_a_t_np)
            normed_action.append(inverted_a_t_np)
        hist_action: List[np.ndarray] = batch_input["hist_action"]  # [t_h, action_dim]
        all_normed_action = hist_action + normed_action
        all_action_latent = get_action_latent(
            world_model,
            all_normed_action,
            task_description,
            wm_args.frame_level_cond,
        )  # [1, t_h + t, d_action]

        with torch.no_grad():
            # get predicted observation
            pred_images_list, pred_images_latent = CtrlWorldDiffusionPipeline.__call__(
                world_model.pipeline,
                image=image_latent,  # current observation, [1, 4, 24, 40]
                text=all_action_latent,  # ? whole sequence, history + future
                width=wm_args.width,
                height=wm_args.height,
                num_frames=num_future_frames,  # This is the number of future frames to predict
                history=hist_images_latent,  # history observations, [1, t, 4, 24, 40]
                # history=None,
                num_inference_steps=wm_args.num_inference_steps,
                decode_chunk_size=num_future_frames,
                max_guidance_scale=wm_args.guidance_scale,
                fps=wm_args.fps,
                motion_bucket_id=wm_args.motion_bucket_id,
                mask=None,
                output_type="frame",
                return_dict=False,
                frame_level_cond=wm_args.frame_level_cond,
                his_cond_zero=wm_args.his_cond_zero,
            )
            # pred_images_list: list of numpy, bsz * [t, h, w, 3], bsz = 1
            # pred_images_latent: tensor, [bsz, t, c, h, w]

            # get terminated reward
            pred_images = torch.from_numpy(pred_images_list[0])  # [t, h, w, 3]
            pred_images = torch.cat(
                [torch.from_numpy(pred_imgs_np) for pred_imgs_np in pred_images_list],
                dim=0,
            )  # [bsz * t, h, w, 3], bsz = 1
            pred_images = einops.rearrange(pred_images, "b_t h w c -> b_t c h w")

            action_latent = all_action_latent[
                :, -num_future_frames:
            ]  # [1, t, d_action]
            action_latent_flat = einops.rearrange(
                action_latent, "b t d -> (b t) d"
            )  # [bsz * t, d], bsz = 1
            pred_scores = get_inner_module(world_model).reward_classifier.predict_score(
                pred_images,
                action_latent_flat,
            )  # [bsz * t], bsz = 1

        # Storage Results
        step_images: List[np.ndarray] = []  # World Model Observations
        env_step_images: List[np.ndarray] = []  # LIBERO Env Observations
        env_step_dones: List[bool] = []  # LIBERO Env Dones
        for t in range(pred_images_list[0].shape[0]):
            pred_img_t: np.ndarray = pred_images_list[0][t]  # [h, w, 3]
            # resize h * w to H * W
            pred_img_t = resize_to_libero_image(pred_img_t, (H, W))  # [H, W, 3]
            step_images.append(pred_img_t)

            if not finished_evolving:
                a = action[t]
                normalized_action = normalize_gripper_action(a, binarize=True)
                inverted_action = invert_gripper_action(normalized_action)
                obs, reward, done, info = env.step(inverted_action.tolist())
                img = obs["agentview_image"][::-1, ::-1]
                env_step_images.append(img)
                env_step_dones.append(done)

            finish_step += 1
            is_success = pred_scores[t] > rm_threshold
            if is_success or finish_step >= max_steps:
                active = False
                complete = is_success
                break

        output_data = {
            "type": "step",
            "obs": None,
            "image": step_images[-1],  # np.array, [H, W, C]
            "active": active,
            "complete": complete,
            "finish_step": finish_step,
            "valid_images": step_images.copy(),  # as next history obs
            "normed_actions": normed_action,  # as next history action
            "env_images": env_step_images.copy(),
            "env_dones": env_step_dones.copy(),
        }
        output_queue.put(output_data)


def env_worker_evolving_envonly(
    task_name,
    task_id,
    trial_id,
    config,
    input_queue,
    output_queue,
    is_valid,
    global_steps,
    max_steps,
    ood_task_description,
):
    """Pure environment worker for Libero (NO world model, NO diffusers)."""
    import gc

    import numpy as np
    import torch

    try:
        # --------- build task ----------
        benchmark_dict = _get_libero_pro_benchmark_dict(config)
        task_suite = benchmark_dict[task_name]()
        task = task_suite.get_task(task_id)
        initial_states = load_task_init_states(task_suite, task_id)
        initial_state = initial_states[trial_id]
        task_description = ood_task_description

        # --------- init env ----------
        env = None
        max_retry = int(getattr(config, "env_init_max_retry", 1))
        for retry in range(1, max_retry + 1):
            try:
                env, _ = _create_libero_env_with_runtime_lock(
                    task,
                    config.model_family,
                    resolution=256,
                    config=config,
                )
                break
            except Exception as e:
                print(
                    f"[env_worker_evolving] init failed ({retry}/{max_retry}): {e}",
                    flush=True,
                )
                if env is not None:
                    try:
                        env.close()
                    except Exception as e_:
                        print(f"error when close the env: {e_}")
                        pass
                torch.cuda.empty_cache()
                gc.collect()
                if retry >= max_retry:
                    output_queue.put(
                        {
                            "type": "error",
                            "error": f"env init failed after {max_retry} retries: {e}",
                            "traceback": traceback.format_exc(),
                            "active": False,
                            "complete": False,
                            "finish_step": 0,
                            "env_images": [],
                            "env_dones": [],
                            "valid_images": [],
                        }
                    )
                    return
                time.sleep(0.1)

        # --------- reset ----------
        env.reset()
        obs = env.set_init_state(initial_state)

        # --------- warmup ----------
        t = 0
        valid_images = []
        while t < config.num_steps_wait:
            dummy_action = get_libero_dummy_action(config.model_family)
            obs, _, _, _ = env.step(dummy_action)
            t += 1

        img = obs["agentview_image"][::-1, ::-1].copy()
        if is_valid:
            valid_images.append(img)

        # --------- send init ----------
        output_queue.put(
            {
                "type": "init",
                "task_description": task_description,
                "obs": obs,
                "image": img,
                "valid_images": valid_images.copy(),
                "env_images": [img],
                "env_dones": [False],
                "normed_actions": [np.array(dummy_action)],
                "active": True,
                "complete": False,
                "finish_step": 0,
                "task_file_name": f"{task_name}_task_{task_id}_trial_{trial_id}",
            }
        )

        # --------- main loop ----------
        active = True
        complete = False
        finish_step = 0

        while True:
            action = input_queue.get()
            if action is None:
                env.close()
                output_queue.put(
                    {
                        "type": "terminate",
                        "active": False,
                        "complete": complete,
                        "finish_step": finish_step,
                    }
                )
                break

            step_images = []
            env_images = []
            env_dones = []
            normed_actions = []

            try:
                for i in range(len(action)):
                    a = action[i]
                    normalized_action = normalize_gripper_action(a, binarize=True)
                    inverted_action = invert_gripper_action(normalized_action)
                    normed_actions.append(inverted_action)

                    obs, reward, done, info = env.step(inverted_action.tolist())

                    img = obs["agentview_image"][::-1, ::-1].copy()
                    if is_valid:
                        step_images.append(img)

                    env_images.append(img)
                    env_dones.append(done)

                    finish_step += 1
                    if done or finish_step >= max_steps:
                        active = False
                        complete = bool(done)
                        break

            except Exception as e:
                # env step crashed → terminate safely
                print(f"[env_worker_evolving] step crashed: {e}", flush=True)
                active = False
                complete = False

            output_queue.put(
                {
                    "type": "step",
                    "obs": obs,
                    "image": env_images[-1] if len(env_images) > 0 else img,
                    "valid_images": step_images.copy() if is_valid else [],
                    "env_images": env_images.copy(),
                    "env_dones": env_dones.copy(),
                    "normed_actions": normed_actions.copy(),
                    "active": active,
                    "complete": complete,
                    "finish_step": finish_step,
                }
            )

            if not active:
                env.close()
                output_queue.put(
                    {
                        "type": "terminate",
                        "active": False,
                        "complete": complete,
                        "finish_step": finish_step,
                    }
                )
                break

    except Exception as e:
        # --------- final safety net ----------
        print(f"[env_worker_evolving] fatal error: {e}", flush=True)
        output_queue.put(
            {
                "type": "error",
                "error": str(e),
            }
        )


# todo: to be changed
# -------------------------
# 微小增强的 env worker（几乎不变，只加了对异常的更明确终止消息，避免主进程长时间阻塞）
# -------------------------
def env_worker_evolving_envonly_v1(
    task_name,
    task_id,
    trial_id,
    config,
    input_queue,
    output_queue,
    is_valid,
    global_steps,
    max_steps,
    ood_task_description,
):
    """Pure environment worker for Libero (NO world model, NO diffusers).
    只做非常小的强化：任何最终退出路径都保证会 put 一个 'terminate' 或 'error' 消息，
    避免主进程因为子进程崩溃而永远阻塞等待消息。
    （其余逻辑与你的原版保持一致）
    """
    import gc

    def emit_progress(stage: str, **extra):
        message = {
            "type": "progress",
            "stage": stage,
            "task_name": task_name,
            "task_id": task_id,
            "trial_id": trial_id,
        }
        message.update(extra)
        try:
            output_queue.put(message)
        except Exception:
            pass

    try:
        # --------- build task ----------
        benchmark_dict = _get_libero_pro_benchmark_dict(config)
        task_suite = benchmark_dict[task_name]()
        task = task_suite.get_task(task_id)
        initial_states = load_task_init_states(task_suite, task_id)
        initial_state = initial_states[trial_id]
        task_description = ood_task_description
        emit_progress("task_loaded")

        # --------- init env ----------
        env = None
        max_retry = int(getattr(config, "env_init_max_retry", 1))
        for retry in range(1, max_retry + 1):
            try:
                env, _ = _create_libero_env_with_runtime_lock(
                    task,
                    config.model_family,
                    resolution=256,
                    config=config,
                    emit_progress=emit_progress,
                    retry=retry,
                )
                break
            except Exception as e:
                print(
                    f"[env_worker_evolving] init failed ({retry}/{max_retry}): {e}",
                    flush=True,
                )
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                torch.cuda.empty_cache()
                gc.collect()
                if retry >= max_retry:
                    output_queue.put(
                        {
                            "type": "error",
                            "error": f"env init failed after {max_retry} retries: {e}",
                            "active": False,
                            "complete": False,
                            "finish_step": 0,
                            "env_images": [],
                            "env_dones": [],
                            "valid_images": [],
                        }
                    )
                    return
                time.sleep(0.1)

        # --------- reset ----------
        emit_progress("env_reset_start")
        env.reset()
        emit_progress("env_reset_done")
        emit_progress("set_init_state_start")
        obs = env.set_init_state(initial_state)
        emit_progress("set_init_state_done")

        # --------- warmup ----------
        t = 0
        valid_images = []
        while t < config.num_steps_wait:
            dummy_action = get_libero_dummy_action(config.model_family)
            obs, _, _, _ = env.step(dummy_action)
            t += 1
            if t == config.num_steps_wait or t % 2 == 0:
                emit_progress("warmup", step=t, total=config.num_steps_wait)

        img = obs["agentview_image"][::-1, ::-1].copy()
        if is_valid:
            valid_images.append(img)

        # --------- send init ----------
        output_queue.put(
            {
                "type": "init",
                "task_description": task_description,
                "obs": obs,
                "image": img,
                "valid_images": valid_images.copy(),
                "env_images": [img],
                "env_dones": [False],
                "normed_actions": [np.array(dummy_action)],
                "active": True,
                "complete": False,
                "finish_step": 0,
                "task_file_name": f"{task_name}_task_{task_id}_trial_{trial_id}",
            }
        )

        # --------- main loop ----------
        active = True
        complete = False
        finish_step = 0

        while True:
            action = input_queue.get()
            if action is None:
                # graceful terminate
                try:
                    env.close()
                except Exception:
                    pass
                output_queue.put(
                    {
                        "type": "terminate",
                        "active": False,
                        "complete": complete,
                        "finish_step": finish_step,
                    }
                )
                break

            step_images = []
            env_images = []
            env_dones = []
            normed_actions = []

            try:
                for i in range(len(action)):
                    a = action[i]
                    normalized_action = normalize_gripper_action(a, binarize=True)
                    inverted_action = invert_gripper_action(normalized_action)
                    normed_actions.append(inverted_action)

                    obs, reward, done, info = env.step(inverted_action.tolist())

                    img = obs["agentview_image"][::-1, ::-1].copy()
                    if is_valid:
                        step_images.append(img)

                    env_images.append(img)
                    env_dones.append(done)

                    finish_step += 1
                    if done or finish_step >= max_steps:
                        active = False
                        complete = bool(done)
                        break

            except Exception as e:
                # env step crashed → terminate safely, but ensure we send terminate message
                print(f"[env_worker_evolving] step crashed: {e}", flush=True)
                active = False
                complete = False
                try:
                    env.close()
                except Exception:
                    pass

            # always send step (even when crashed) to let main know current state
            try:
                output_queue.put(
                    {
                        "type": "step",
                        "obs": obs,
                        "image": env_images[-1] if len(env_images) > 0 else img,
                        "valid_images": step_images.copy() if is_valid else [],
                        "env_images": env_images.copy(),
                        "env_dones": env_dones.copy(),
                        "normed_actions": normed_actions.copy(),
                        "active": active,
                        "complete": complete,
                        "finish_step": finish_step,
                    }
                )
            except Exception:
                # if even putting step fails, ensure a terminate or error message is sent
                pass

            if not active:
                try:
                    env.close()
                except Exception:
                    pass
                output_queue.put(
                    {
                        "type": "terminate",
                        "active": False,
                        "complete": complete,
                        "finish_step": finish_step,
                    }
                )
                break

    except Exception as e:
        # --------- final safety net ----------
        print(f"[env_worker_evolving] fatal error: {e}", flush=True)
        try:
            output_queue.put(
                {
                    "type": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass


import queue as _queue


# ================ Main Rollout Class ================
class RobWMHFRolloutPro(BaseRollout):  #! tmp：跑通后记得改回RobWMHFRollout
    def __init__(self, module: nn.Module, config, world_model_mapping: Dict = None):
        super().__init__()
        self.config = config
        self.world_model_mapping = world_model_mapping or {}
        self.module = module  # 指的是vla
        self.libero_runtime_env = configure_libero_runtime_env(force_headless=True)
        print(
            format_libero_runtime_env_summary(
                self.libero_runtime_env,
                prefix="[LIBERO_PRO runtime]",
            ),
            flush=True,
        )
        #! libero: 512 -> >= bs * 8**2 * N
        self.max_steps = {
            "libero_spatial": 512,  # 220, tl: 27
            "libero_object": 512,  # 280, tl: 35
            "libero_goal": 512,  # 300, tl: 37
            "libero_10": 512,  # 520, traj_len: 65
            "libero_90": 512,  # 400, tl: 50
            "robotwin2_click_bell": 200,
            "robotwin2_move_can_pot": 200,
            "robotwin2_place_phone_stand": 200,
            "robotwin2_place_a2b_left": 200,
            "robotwin2_place_a2b_right": 200,
            "robotwin2_handover_mic": 200,
            "robotwin2_pick_dual_bottles": 100,
            "robotwin2_lift_pot": 200,
            "robotwin2_put_bottles_dustbin": 800,
            "robotwin2_stack_blocks_two": 400,
            "robotwin2_stack_bowls_two": 400,
            "robotwin2_handover_block": 400,
            "robotwin2_place_empty_cup": 200,
            "robotwin2_shake_bottle": 75,
            "robotwin2_move_stapler_pad": 200,
            "robotwin2_place_container_plate": 150,
            "robotwin2_blocks_ranking_rgb": 600,
            "robotwin2_beat_block_hammer": 200,
            "robotwin2_place_mouse_pad": 200,
            "robotwin2_place_shoe": 250,
            "robotwin2_move_pillbottle_pad": 200,
        }
        self.processor = hf_processor(
            config.pretrained_checkpoint,
            model=getattr(config, "vla", None),
            trust_remote_code=True,
        )
        self.vla_preprocess()

        # Setup execution pool based on task suite
        if "robotwin" in self.config.task_suite_name:
            self.env_thread_pool = ThreadPoolExecutor(max_workers=16)
            self.robotwin_version = self._detect_robotwin_version()

        # Rebuild world model from mapping only when evolving / WM-only modes are enabled.
        self.world_model: Optional[CtrlWorld] = self.world_model_mapping.get(
            "world_model"
        )
        self.rm_threshold: Optional[float] = self.world_model_mapping.get(
            "rm_threshold"
        )
        self.wm_args = self.world_model_mapping.get("wm_args")
        self.device = self.world_model_mapping.get("device")
        self.dtype = self.world_model_mapping.get("dtype")

        if self.world_model is not None and self.device is not None:
            self.world_model.eval()
            self.world_model.to(self.device)
        self._libero_env_services: List[_LiberoEnvServiceClient] = []
        self._libero_env_service_counter = 0

    @staticmethod
    def _select_ood_task_description(
        all_ood_task_descriptions: List[List[str]],
        sample_idx: int,
        n_samples: int,
        task_id: int,
    ) -> str:
        """Select a perturbation task description in prompt order, not filesystem order."""
        if not all_ood_task_descriptions:
            raise RuntimeError("LIBERO_PRO task descriptions are empty.")
        prompt_idx = int(sample_idx) // max(int(n_samples), 1)
        prompt_idx = min(max(prompt_idx, 0), len(all_ood_task_descriptions) - 1)
        descriptions = all_ood_task_descriptions[prompt_idx]
        task_id = int(task_id)
        if task_id < 0 or task_id >= len(descriptions):
            raise IndexError(
                "LIBERO_PRO task description index out of range: "
                f"task_id={task_id}, descriptions={len(descriptions)}, "
                f"prompt_idx={prompt_idx}, n_suites={len(all_ood_task_descriptions)}"
            )
        return str(descriptions[task_id])

    def _require_world_model_runtime(self) -> None:
        required_fields = {
            "world_model": self.world_model,
            "rm_threshold": self.rm_threshold,
            "wm_args": self.wm_args,
            "device": self.device,
            "dtype": self.dtype,
        }
        missing = [name for name, value in required_fields.items() if value is None]
        if missing:
            raise RuntimeError(
                "RobWMHFRolloutPro requires world_model_mapping for WM/evolving rollout paths; "
                f"missing fields: {missing}"
            )

    def _libero_env_service_enabled(self) -> bool:
        return _as_bool_flag(
            getattr(
                self.config,
                "env_service_enable",
                os.environ.get("MERL_LIBERO_ENV_SERVICE_ENABLE", "true"),
            ),
            default=True,
        )

    def _acquire_libero_env_service(
        self,
        *,
        mp_ctx,
        worker_config,
        init_timeout_s: float,
        step_timeout_s: float,
    ) -> _LiberoEnvServiceClient:
        for client in list(self._libero_env_services):
            if not client.alive:
                with contextlib.suppress(Exception):
                    client.close(force=True)
                with contextlib.suppress(ValueError):
                    self._libero_env_services.remove(client)
                continue
            if not client.busy:
                client.busy = True
                return client

        self._libero_env_service_counter += 1
        client = _LiberoEnvServiceClient(
            mp_ctx=mp_ctx,
            worker_config=worker_config,
            label=f"LIBERO_PRO service {self._libero_env_service_counter}",
            init_timeout_s=init_timeout_s,
            step_timeout_s=step_timeout_s,
        )
        client.busy = True
        self._libero_env_services.append(client)
        return client

    def _release_libero_env_service(
        self, client: Optional[_LiberoEnvServiceClient], *, healthy: bool = True
    ) -> None:
        if client is None:
            return
        if healthy and client.alive:
            client.busy = False
            return
        with contextlib.suppress(Exception):
            client.close(force=True)
        with contextlib.suppress(ValueError):
            self._libero_env_services.remove(client)

    def _release_libero_env_handles(
        self, handles: List[Any], *, healthy: bool = True
    ) -> None:
        for handle in handles:
            if isinstance(handle, _LiberoEnvServiceClient):
                self._release_libero_env_service(handle, healthy=healthy)

    def _close_libero_env_services(self) -> None:
        for client in list(self._libero_env_services):
            with contextlib.suppress(Exception):
                client.close(force=True)
        self._libero_env_services.clear()

    def _detect_robotwin_version(self):
        """Detect which version of robotwin to use based on config"""
        if hasattr(self.config, "robotwin_version"):
            return self.config.robotwin_version
        elif "robotwin2" in self.config.task_suite_name:
            return "2.0"
        else:
            print(
                "RobotWin 2.0 fully encompasses RobotWin 1.0, therefore we prioritize support for RobotWin 2.0"
            )
            raise ValueError

        self.rm_model.eval()
        # videos B T H W C
        total_frames = videos.shape[1]
        window_size = 8
        stride = 1
        min_steps = 100
        results = []
        # start_time = time.time()

        for video_idx, video in enumerate(videos):
            clips = []
            for end in range(total_frames, min_steps + window_size - 1, -stride):
                clip = video[end - window_size : end]
                clips.append((clip, end - window_size, end))
            clips = clips[::-1]
            clip_batches = [
                clips[i : i + batch_size] for i in range(0, len(clips), batch_size)
            ]

            finish_step = total_frames - 1
            complete = 0
            for batch_idx, batch in enumerate(clip_batches):
                # current_time = time.time()
                # elapsed_time = current_time - start_time
                # print(f"Rank {dist.get_rank()}: Elapsed time: {elapsed_time:.2f} seconds : video {video_idx}/{len(videos)} batch {batch_idx}/{len(clip_batches)}")
                ranges = [(c[1], c[2]) for c in batch]
                clip_imgs = [c[0] for c in batch]
                # clip_imgs = [[Image.fromarray(frame).convert("RGB") for frame in clip ] for clip in clip_imgs]
                clip_imgs = [[img for img in clip] for clip in clip_imgs]
                # if video_idx == 0 and batch_idx == 0:
                #     print(f"[PID {os.getpid()}] before TF feature_extractor")
                inputs = self.rm_feature_extractor(clip_imgs, return_tensors="pt")[
                    "pixel_values"
                ].to(
                    self.device
                )  # device(type='cuda', index=7)
                # if video_idx == 0 and batch_idx == 0:
                #     print(f"[PID {os.getpid()}] after TF feature_extractor")
                logits = self.rm_model(pixel_values=inputs).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = [1 if p[1] >= self.rm_threshold else 0 for p in probs]

                for (start, end), prob, pred in zip(ranges, probs, preds):
                    if pred == 1 and end - 1 < finish_step:
                        finish_step = end - 1
                        complete = 1
                        break
            results.append({"complete": complete, "finish_step": finish_step})

        complete = torch.from_numpy(np.array([r["complete"] for r in results]))
        finish_step = torch.from_numpy(np.array([r["finish_step"] for r in results]))

        return {"complete": complete, "finish_step": finish_step}

    def vla_preprocess(self):
        if self.config.vla in ["openvla", "openvla-oft"]:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

        if self.config.vla in ["openvla-oft"]:
            if "libero" in self.config.task_suite_name:
                if (
                    self.config.unnorm_key not in self.module.norm_stats
                    and f"{self.config.unnorm_key}_no_noops" in self.module.norm_stats
                ):
                    self.config.unnorm_key = f"{self.config.unnorm_key}_no_noops"
            elif "robotwin" in self.config.task_suite_name:
                self.config.unnorm_key = self.config.unnorm_key.removeprefix(
                    "robotwin_"
                ).removeprefix("robotwin2_")
            print(f"self.config.unnorm_key: {self.config.unnorm_key}")
            print(f"self.module.norm_stats keys: {self.module.norm_stats.keys()}")
            assert (
                self.config.unnorm_key in self.module.norm_stats
            ), f"Action un-norm key {self.config.unnorm_key} not found in VLA `norm_stats`!"

    # pure env mode
    def generate_sequences(self, prompts) -> DataProto:
        print("[rollout] Using pure env mode.")
        batch_size = prompts.batch.batch_size[0]

        if prompts.meta_info.get("n_samples") is None:
            micro_batch_size = (
                self.config.val_micro_batch_size
                if self.config.val_micro_batch_size is not None
                else 1
            )
        else:
            micro_batch_size = self.config.get("micro_batch_size", batch_size)

        print(f"[rollout] micro_batch_size: {micro_batch_size}")
        num_chunks = max(batch_size // micro_batch_size, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        print(f"[rob_rollout] Length of batch_prompts: {len(batch_prompts)}")
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = self._pad_outputs_for_concat(output)
        output = DataProto.concat(output)
        print(f"[rob_rollout] Finished gen a batch.")
        return output

    #! pure world model mode
    def generate_sequences_evolving(self, prompts) -> DataProto:
        self._require_world_model_runtime()
        print("[rob_rollout] Using evovling world model mode.")
        batch_size = prompts.batch.batch_size[0]
        requested_n_samples = int(prompts.meta_info.get("n_samples") or 1)
        serial_samples = _as_bool_flag(
            self.config.get("wm_evolving_serial_samples", True), default=True
        )

        if prompts.meta_info.get("n_samples") is None:
            micro_batch_size = (
                self.config.val_micro_batch_size
                if self.config.val_micro_batch_size is not None
                else 1
            )
        else:
            micro_batch_size = self.config.get("micro_batch_size", batch_size)

        micro_batch_size = max(int(micro_batch_size), 1)
        print(f"[rob_rollout] micro_batch_size: {micro_batch_size}")
        num_chunks = max((batch_size + micro_batch_size - 1) // micro_batch_size, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        print(f"[rob_rollout] Length of batch_prompts: {len(batch_prompts)}")
        output = []
        if requested_n_samples > 1 and serial_samples:
            total_serial_samples = (
                sum(len(prompt_chunk) for prompt_chunk in batch_prompts)
                * requested_n_samples
            )
            serial_sample_idx = 0
            print(
                f"[rob_rollout] Serializing WM evolving n_samples={requested_n_samples}, "
                f"total_serial_samples={total_serial_samples}",
                flush=True,
            )
            for prompt_chunk in batch_prompts:
                for row_idx in range(len(prompt_chunk)):
                    single_prompt = prompt_chunk.slice(slice(row_idx, row_idx + 1))
                    for sample_idx in range(requested_n_samples):
                        serial_sample_idx += 1
                        print(
                            f"[rob_rollout] WM evolving serial sample "
                            f"{serial_sample_idx}/{total_serial_samples}",
                            flush=True,
                        )
                        sample_prompt = single_prompt.slice(slice(0, 1))
                        sample_prompt.non_tensor_batch = {
                            key: np.asarray(value).copy()
                            for key, value in sample_prompt.non_tensor_batch.items()
                        }
                        sample_prompt.meta_info = dict(prompts.meta_info)
                        sample_prompt.meta_info["n_samples"] = 1
                        sample_prompt.meta_info["wm_sample_index"] = sample_idx
                        output.append(self._generate_minibatch_evolving(sample_prompt))
        else:
            output = [self._generate_minibatch_evolving(p) for p in batch_prompts]
        output = self._pad_outputs_for_concat(output)
        output = DataProto.concat(output)
        output.meta_info = dict(prompts.meta_info)
        print(f"[rob_rollout] Finished gen a batch.")
        return output

    def _pad_outputs_for_concat(self, outputs: List[DataProto]) -> List[DataProto]:
        if len(outputs) <= 1:
            return outputs

        tensor_shapes: Dict[str, Tuple[int, ...]] = {}
        tensor_keys = set()
        non_tensor_keys = set()
        for output in outputs:
            if output.batch is None:
                continue
            tensor_keys.update(list(output.batch.keys()))
            non_tensor_keys.update(list(output.non_tensor_batch.keys()))

        for key in tensor_keys:
            values = []
            for output in outputs:
                if output.batch is None or key not in output.batch:
                    continue
                value = output.batch[key]
                if not isinstance(value, torch.Tensor):
                    continue
                values.append(value)
            if not values:
                continue
            target_ndim = int(values[0].ndim)
            if any(int(value.ndim) != target_ndim for value in values):
                raise RuntimeError(
                    f"[rollout] Cannot pad key '{key}' for concat: ndim mismatch "
                    f"{[tuple(value.shape) for value in values]}"
                )
            max_shape = tuple(
                max(int(value.shape[dim]) for value in values)
                for dim in range(1, target_ndim)
            )
            if len(values) != len(outputs) or any(
                tuple(value.shape[1:]) != max_shape for value in values
            ):
                tensor_shapes[key] = max_shape

        needs_non_tensor_fill = any(
            key not in output.non_tensor_batch
            for key in non_tensor_keys
            for output in outputs
        )

        if not tensor_shapes and not needs_non_tensor_fill:
            return outputs

        def _pad_fill_value(key: str, value: torch.Tensor, missing_key: bool = False):
            if missing_key:
                return False if value.dtype == torch.bool else 0
            if key == "is_dummy":
                return True
            if key in {"input_ids", "responses"}:
                return int(self.processor.tokenizer.pad_token_id or 0)
            return 0

        def _pad_value(
            key: str, value: torch.Tensor, target_shape: Tuple[int, ...]
        ) -> torch.Tensor:
            if tuple(value.shape[1:]) == target_shape:
                return value
            padded = torch.full(
                (int(value.shape[0]), *target_shape),
                _pad_fill_value(key, value),
                dtype=value.dtype,
                device=value.device,
            )
            slices = (slice(None),) + tuple(
                slice(0, int(size)) for size in value.shape[1:]
            )
            padded[slices] = value
            return padded

        padded_outputs = []
        template_by_key: Dict[str, torch.Tensor] = {}
        for key, target_shape in tensor_shapes.items():
            for output in outputs:
                if output.batch is not None and key in output.batch:
                    template = output.batch[key]
                    if isinstance(template, torch.Tensor):
                        template_by_key[key] = template
                        break

        for output in outputs:
            if output.batch is None:
                padded_outputs.append(output)
                continue
            source = {}
            for key, target_shape in tensor_shapes.items():
                if key in output.batch:
                    continue
                template = template_by_key.get(key)
                if template is None:
                    continue
                source[key] = torch.full(
                    (len(output), *target_shape),
                    _pad_fill_value(key, template, missing_key=True),
                    dtype=template.dtype,
                    device=template.device,
                )
            for key in output.batch.keys():
                value = output.batch[key]
                if key in tensor_shapes and isinstance(value, torch.Tensor):
                    source[key] = _pad_value(key, value, tensor_shapes[key])
                else:
                    source[key] = value
            non_tensor_batch = dict(output.non_tensor_batch)
            for key in non_tensor_keys:
                if key not in non_tensor_batch:
                    non_tensor_batch[key] = np.array([None] * len(output), dtype=object)
            padded_outputs.append(
                DataProto(
                    batch=TensorDict(source, batch_size=output.batch.batch_size),
                    non_tensor_batch=non_tensor_batch,
                    meta_info=output.meta_info,
                )
            )
        if tensor_shapes:
            print(
                "[rollout] padded variable-length microbatch outputs for concat: "
                + ", ".join(
                    f"{key}->{shape}" for key, shape in sorted(tensor_shapes.items())
                ),
                flush=True,
            )
        return padded_outputs

    def process_input(self, inputs: list, task_descriptions: list):
        """Unified input processing for both Robotwin and Libero"""
        batchdata = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        if self.config.use_proprio and "robotwin" in self.config.task_suite_name:
            batchdata["proprio"] = []

        for i in range(len(inputs)):
            input_data = inputs[i]
            task_description = task_descriptions[i]

            # Process main image
            image = Image.fromarray(input_data["full_image"]).convert("RGB")  # 256
            if self.config.center_crop:
                image = center_crop_image(image)  # 256 -> 224
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            batch_feature = self.processor(prompt, image)

            pixel_values_list = [batch_feature["pixel_values"]]

            # Process additional images (wrist cameras)
            if "robotwin" in self.config.task_suite_name:
                # Robotwin may have multiple wrist images
                for key in input_data:
                    if "wrist" in key and isinstance(input_data[key], np.ndarray):
                        wrist_image = Image.fromarray(input_data[key]).convert("RGB")
                        if self.config.center_crop:
                            wrist_image = center_crop_image(wrist_image)
                        wrist_batch_feature = self.processor(prompt, wrist_image)
                        pixel_values_list.append(wrist_batch_feature["pixel_values"])
            else:
                # Libero has single wrist image
                if "wrist_image" in input_data:
                    raise NotImplementedError
                    wrist_image = Image.fromarray(input_data["wrist_image"]).convert(
                        "RGB"
                    )
                    if self.config.center_crop:
                        wrist_image = center_crop_image(wrist_image)
                    wrist_batch_feature = self.processor(prompt, wrist_image)
                    pixel_values_list.append(wrist_batch_feature["pixel_values"])

            assert (
                len(pixel_values_list) == 1
            ), f"NotImplementedError for len(pixel_values_list): {len(pixel_values_list)}"
            batch_feature["pixel_values"] = torch.cat(pixel_values_list, dim=1)

            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature["attention_mask"]
            pixel_values = batch_feature["pixel_values"]
            pixel_values = pixel_values.to(dtype=torch.bfloat16)

            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (
                        input_ids,
                        torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(
                            input_ids.device
                        ),
                    ),
                    dim=1,
                )
                if self.config.vla in ["openvla-oft"]:
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(
                                attention_mask.device
                            ),
                        ),
                        dim=1,
                    )

            batchdata["input_ids"].append(input_ids)
            batchdata["attention_mask"].append(attention_mask)
            batchdata["pixel_values"].append(pixel_values)

            # Process proprioception for Robotwin
            if self.config.use_proprio and "robotwin" in self.config.task_suite_name:
                proprio = input_data["state"]
                proprio_norm_stats = self.module.norm_stats[self.config.unnorm_key][
                    "proprio"
                ]
                proprio = normalize_proprio(proprio, proprio_norm_stats)
                batchdata["proprio"].append(torch.from_numpy(proprio))

        device = torch.device("cuda")

        # Padding and device placement
        if self.config.vla in ["openvla-oft"]:
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [
                x.transpose(0, 1) for x in batchdata["attention_mask"]
            ]
            batchdata["input_ids"] = (
                pad_sequence(
                    batchdata["input_ids"],
                    batch_first=True,
                    padding_value=self.processor.tokenizer.pad_token_id,
                )
                .squeeze(-1)
                .to(device)
            )
            batchdata["attention_mask"] = (
                pad_sequence(
                    batchdata["attention_mask"], batch_first=True, padding_value=0
                )
                .squeeze(-1)
                .to(device)
            )

            padding_mask = batchdata["input_ids"].ne(
                self.processor.tokenizer.pad_token_id
            )
            assert torch.all(padding_mask == batchdata["attention_mask"].ne(0))
            padding_mask = ~padding_mask
            padding_mask = padding_mask.int()
            sorted_indices = torch.argsort(
                padding_mask, dim=1, descending=True, stable=True
            )
            batchdata["input_ids"] = torch.gather(
                batchdata["input_ids"], 1, sorted_indices
            )
            batchdata["attention_mask"] = torch.gather(
                batchdata["attention_mask"], 1, sorted_indices
            )

            batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"], dim=0).to(
                device
            )  # [T, H, W, C]

            if self.config.use_proprio and "robotwin" in self.config.task_suite_name:
                batchdata["proprio"] = torch.stack(batchdata["proprio"], dim=0).to(
                    device
                )

            assert torch.all(
                batchdata["attention_mask"].ne(0)
                == batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id)
            )
        else:
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)

        return batchdata

    def _generate_minibatch(self, prompts) -> DataProto:
        """Generate minibatch - routes to appropriate implementation based on task suite"""
        try:
            if "robotwin" in self.config.task_suite_name:
                return self._generate_minibatch_robotwin(prompts)
            else:
                return self._generate_minibatch_libero(prompts)
        finally:
            self._cleanup_environments()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _generate_minibatch_wm(self, prompts) -> DataProto:
        self._require_world_model_runtime()
        if "libero" in self.config.task_suite_name:
            return self._generate_minibatch_libero_wm(prompts)
        else:
            raise NotImplementedError

    #! evolving mode
    def _generate_minibatch_evolving(self, prompts) -> DataProto:
        self._require_world_model_runtime()
        if "libero" in self.config.task_suite_name:
            return self._generate_minibatch_libero_evolving(prompts)
        else:
            raise NotImplementedError(f"{self.config.task_suite_name}")

    def _should_save_training_videos(self, global_steps: int) -> bool:
        if not bool(getattr(self.config, "save_training_videos", True)):
            return False

        video_save_interval = int(getattr(self.config, "video_save_interval", 1))
        if video_save_interval <= 0:
            return False

        return int(global_steps) >= 0 and int(global_steps) % video_save_interval == 0

    def _save_debug_rollout_videos(
        self,
        video_records,
        task_records,
        global_steps: int,
        *,
        save_env_tag=None,
        save_wm_tag=None,
    ):
        if not self._should_save_training_videos(global_steps):
            return

        max_env_videos_per_task = max(
            1, int(getattr(self.config, "max_env_videos_per_task", 1))
        )
        max_wm_videos_per_task = max(
            1, int(getattr(self.config, "max_wm_videos_per_task", 1))
        )
        env_counts = defaultdict(int)
        wm_counts = defaultdict(int)

        for idx, task_record in enumerate(task_records):
            if idx >= len(video_records):
                continue

            video_record = video_records[idx]
            task_file_name = task_record.get("task_file_name", f"task_{idx}")
            complete = bool(task_record.get("complete", False))

            if save_wm_tag is not None:
                wm_images = video_record.get("wm_images", [])
                if wm_images and wm_counts[task_file_name] < max_wm_videos_per_task:
                    ordinal = wm_counts[task_file_name]
                    wm_counts[task_file_name] += 1
                    wm_suffix = (
                        f"_{save_wm_tag}"
                        if ordinal == 0
                        else f"_{save_wm_tag}_sample{ordinal}"
                    )
                    save_rollout_video(
                        wm_images,
                        self.config.experiment_name,
                        task_file_name + wm_suffix,
                        global_steps,
                        complete,
                    )

            if save_env_tag is not None:
                env_images = video_record.get("env_images", [])
                if env_images and env_counts[task_file_name] < max_env_videos_per_task:
                    ordinal = env_counts[task_file_name]
                    env_counts[task_file_name] += 1
                    env_suffix = (
                        f"_{save_env_tag}"
                        if ordinal == 0
                        else f"_{save_env_tag}_sample{ordinal}"
                    )
                    env_success = complete if save_env_tag == "real_env" else None
                    save_rollout_video(
                        env_images,
                        self.config.experiment_name,
                        task_file_name + env_suffix,
                        global_steps,
                        env_success,
                    )

    def _generate_minibatch_robotwin(self, prompts):
        """Generate minibatch for Robotwin using threading"""
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get("n_samples", 1)
        task_id = prompts.batch["task_id"].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch["trial_id"].repeat_interleave(n_samples, dim=0)
        trial_seed = prompts.batch["trial_seed"].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(
            prompts.non_tensor_batch["task_suite_name"], n_samples
        )
        max_steps = self.max_steps.get(self.config.task_suite_name, 800)
        batch_size = task_id.size(0)
        is_valid = meta_info.get("n_samples") is None
        global_steps = meta_info.get("global_steps", 0) if is_valid else 0

        # Create environment wrappers
        env_wrappers = []
        for idx in range(batch_size):
            task_name = (
                task_suite_name[idx]
                .removeprefix("robotwin_")
                .removeprefix("robotwin2_")
            )
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            tr_seed = trial_seed[idx][0].item()

            wrapper = RobotwinEnvWrapper(
                task_name, tr_id, tr_seed, self.config, version=self.robotwin_version
            )
            env_wrappers.append(wrapper)

        # Initialize environments in parallel
        init_futures = []
        for wrapper in env_wrappers:
            future = self.env_thread_pool.submit(wrapper.initialize)
            init_futures.append(future)

        for future in as_completed(init_futures, timeout=360):
            try:
                future.result()
            except Exception as e:
                print(f"Environment initialization failed: {e}", flush=True)
                traceback.print_exc()
                raise

        # Collect initial observations
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)

        for idx, wrapper in enumerate(env_wrappers):
            try:
                obs = wrapper.get_obs()
                obs = encode_obs(obs)

                task_description = wrapper.get_instruction()
                task_descriptions.append(task_description)
                inputs.append(
                    self._obs_to_input(
                        obs, is_robotwin=True, robotwin_version=wrapper.version
                    )
                )

                task_file_name = f"{wrapper.task_name}_trial_{wrapper.trial_id}_seed_{wrapper.trial_seed}"
                task_records.append(
                    {
                        "active": wrapper.active,
                        "complete": wrapper.complete,
                        "finish_step": wrapper.finish_step,
                        "task_file_name": task_file_name,
                    }
                )

                if is_valid:
                    img = obs["observation"]["head_camera"]["rgb"]
                    valid_video[task_file_name].append(img)

            except Exception as e:
                print(f"Failed to get initial observation: {e}", flush=True)
                traceback.print_exc()
                raise

        # Main rollout loop
        step = 0
        vla_history = []

        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r["active"]]

            current_inputs = inputs
            current_task_descriptions = task_descriptions

            # Get VLA actions
            vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)

            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]

            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "step": step,
            }
            if vla_output.get("proprio") is not None:
                step_data["proprio"] = vla_output["proprio"]

            vla_history.append(step_data)

            # Execute actions in parallel
            step_futures = []
            for idx in active_indices:
                future = self.env_thread_pool.submit(
                    env_wrappers[idx].step, actions[idx]
                )
                step_futures.append((idx, future))

            # Collect results
            new_inputs = inputs.copy()
            for idx, future in step_futures:
                try:
                    obs, done = future.result(timeout=120)
                    if obs is not None:
                        obs = encode_obs(obs)
                        new_inputs[idx] = self._obs_to_input(
                            obs,
                            is_robotwin=True,
                            robotwin_version=env_wrappers[idx].version,
                        )

                    task_records[idx]["active"] = env_wrappers[idx].active
                    task_records[idx]["complete"] = env_wrappers[idx].complete
                    task_records[idx]["finish_step"] = env_wrappers[idx].finish_step

                    if is_valid and obs is not None:
                        img = obs["observation"]["head_camera"]["rgb"]
                        valid_video[task_records[idx]["task_file_name"]].append(img)

                except Exception as e:
                    print(f"Step execution failed: {e}", flush=True)
                    task_records[idx]["active"] = False
                    task_records[idx]["complete"] = False
                    task_records[idx]["finish_step"] = (
                        step + self.config.action_chunks_len
                    )

            inputs = new_inputs
            step += self.config.action_chunks_len

        # Clean up environments
        cleanup_futures = []
        for wrapper in env_wrappers:
            future = self.env_thread_pool.submit(wrapper.close)
            cleanup_futures.append(future)

        for future in as_completed(cleanup_futures):
            try:
                future.result(timeout=20)
            except Exception as e:
                print(f"Environment cleanup failed: {e}", flush=True)

        torch.cuda.empty_cache()
        gc.collect()

        # Save validation videos
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(
                    r["complete"]
                    for r in task_records
                    if r["task_file_name"] == task_file
                )
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete,
                )

        self.module.train()

        # Prepare output batch
        return self._prepare_output_batch(vla_history, task_records, batch_size)

    def _configured_allowed_task_ids(self) -> Optional[List[int]]:
        raw_ids = getattr(self.config, "allowed_task_ids", None)
        if raw_ids is None:
            return None
        if isinstance(raw_ids, str):
            text = raw_ids.strip()
            if text == "" or text.lower() in {"all", "full", "none", "null"}:
                return None
            raw_ids = (
                ast.literal_eval(text) if text.startswith("[") else text.split(",")
            )
        try:
            ids = sorted({int(str(item).strip()) for item in list(raw_ids)})
        except Exception as exc:
            raise ValueError(
                f"Invalid actor_rollout_ref.rollout.allowed_task_ids={raw_ids!r}"
            ) from exc
        return ids if len(ids) > 0 else None

    @staticmethod
    def sanitize_task_ids(prompts: DataProto, allowed_ids: Optional[List[int]]):
        """Validate task selection; never relabel a requested evaluation task."""
        if not allowed_ids or "task_id" not in prompts.batch:
            return prompts
        task_ids = prompts.batch["task_id"]
        allowed = torch.as_tensor(allowed_ids, dtype=task_ids.dtype, device=task_ids.device)
        if not bool(torch.isin(task_ids, allowed).all().item()):
            raise ValueError("Task IDs outside allowed_task_ids; filter the dataset before dispatch")
        return prompts

    # 1. generate_minibatch: pure env mode
    def _generate_minibatch_libero(self, prompts):
        """Generate minibatch for Libero using multiprocessing"""
        prompts = self.sanitize_task_ids(
            prompts=prompts, allowed_ids=self._configured_allowed_task_ids()
        )
        all_ood_task_descriptions = self._preprocess_libero_pro_task_suite(
            prompts
        )  # 1 * [10]

        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get("n_samples", 1)
        print(f"[env_rollout] n_samples: {n_samples}")

        task_id = prompts.batch["task_id"].repeat_interleave(
            n_samples, dim=0
        )  # [n_samples, 1]
        trial_id = prompts.batch["trial_id"].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(
            prompts.non_tensor_batch["task_suite_name"], n_samples
        )  # [n_samples, 1]
        max_steps = int(
            meta_info.get(
                "max_steps", self.max_steps.get(self.config.task_suite_name, 800)
            )
        )
        batch_size = task_id.size(0)  # n_samples
        print(f"[env_rollout] batch_size: {batch_size}")
        # is_valid = meta_info.get('n_samples') is None
        #! tmp: save video when training
        is_valid = True
        global_steps = meta_info.get("global_steps", 0) if is_valid else 0
        init_timeout_s = float(getattr(self.config, "env_init_timeout_s", 300.0))

        processes = []
        input_queues = []
        output_queues = []
        env_handles = []
        mp_ctx = _get_env_mp_context(self.config)
        worker_config = _build_libero_env_worker_config(self.config)
        init_timeout_s = float(getattr(worker_config, "env_init_timeout_s", 300.0))
        step_timeout_s = float(getattr(worker_config, "env_step_timeout_s", 60.0))
        init_retry_attempts = int(
            getattr(worker_config, "env_init_parent_max_retry", 1)
        )
        use_env_service = self._libero_env_service_enabled()

        inputs = []
        task_descriptions = []
        task_records = []
        video_records = []
        valid_video = defaultdict(list)

        def _short_error(error: Any) -> str:
            lines = str(error).splitlines()
            return lines[0] if lines else error.__class__.__name__

        def _make_failed_init_data(
            task_name: str,
            t_id: int,
            tr_id: int,
            task_description: str,
            error: Exception,
        ) -> Dict[str, Any]:
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            reason = f"env_init_failed:{_short_error(error)}"
            return {
                "type": "init",
                "task_description": str(task_description or ""),
                "obs": None,
                "image": dummy_image,
                "valid_images": [],
                "env_images": [],
                "env_dones": [],
                "active": False,
                "complete": False,
                "finish_step": 0,
                "task_file_name": f"{task_name}_task_{t_id}_trial_{tr_id}",
                "placeholder_reason": reason,
                "is_dummy": True,
            }

        try:
            for idx in range(batch_size):
                task_name = task_suite_name[idx]
                t_id = task_id[idx][0].item()  # 只支持一个task_suite
                tr_id = trial_id[idx][0].item()
                print(f"[env_rollout] t_id: {t_id}, tr_id: {tr_id}")
                ood_task_description = self._select_ood_task_description(
                    all_ood_task_descriptions,
                    sample_idx=idx,
                    n_samples=n_samples,
                    task_id=t_id,
                )

                env_handle = None
                try:
                    if use_env_service:
                        env_handle = self._acquire_libero_env_service(
                            mp_ctx=mp_ctx,
                            worker_config=worker_config,
                            init_timeout_s=init_timeout_s,
                            step_timeout_s=step_timeout_s,
                        )
                        init_data = env_handle.reset(
                            task_name=task_name,
                            task_id=t_id,
                            trial_id=tr_id,
                            is_valid=is_valid,
                            global_steps=global_steps,
                            max_steps=max_steps,
                            ood_task_description=ood_task_description,
                        )
                        process, input_q, output_q = None, None, None
                    else:
                        process, input_q, output_q, init_data = (
                            _launch_libero_env_worker_with_retry(
                                mp_ctx=mp_ctx,
                                target=env_worker_evolving_envonly_v1,
                                args_factory=lambda iq, oq, task_name=task_name, t_id=t_id, tr_id=tr_id, ood_task_description=ood_task_description: (
                                    task_name,
                                    t_id,
                                    tr_id,
                                    worker_config,
                                    iq,
                                    oq,
                                    is_valid,
                                    global_steps,
                                    max_steps,
                                    ood_task_description,
                                ),
                                worker_label=f"LIBERO_PRO env idx={idx} task={task_name} trial={tr_id}",
                                init_timeout_s=init_timeout_s,
                                max_attempts=init_retry_attempts,
                                worker_config=worker_config,
                            )
                        )
                except Exception as exc:
                    print(
                        "[env_rollout] LIBERO_PRO child init failed; "
                        f"marking idx={idx} task={task_name} trial={tr_id} as dummy: {exc}",
                        flush=True,
                    )
                    self._release_libero_env_service(env_handle, healthy=False)
                    env_handle = None
                    process, input_q, output_q = None, None, None
                    init_data = _make_failed_init_data(
                        task_name=task_name,
                        t_id=t_id,
                        tr_id=tr_id,
                        task_description=ood_task_description,
                        error=exc,
                    )
                processes.append(process)
                input_queues.append(input_q)
                output_queues.append(output_q)
                env_handles.append(env_handle)

                assert init_data["type"] == "init"
                task_descriptions.append(init_data["task_description"])
                if init_data.get("obs") is None:
                    inputs.append(
                        {
                            "full_image": init_data["image"],
                            "state": None,
                        }
                    )
                else:
                    inputs.append(
                        self._obs_to_input(init_data["obs"], is_robotwin=False)
                    )
                task_records.append(
                    {
                        "active": init_data["active"],
                        "complete": init_data["complete"],
                        "finish_step": init_data["finish_step"],
                        "task_file_name": init_data["task_file_name"],
                        "placeholder_reason": init_data.get("placeholder_reason", ""),
                        "is_dummy": bool(init_data.get("is_dummy", False)),
                    }
                )
                init_env_images = list(init_data.get("env_images", []))
                init_env_dones = list(init_data.get("env_dones", []))
                video_records.append(
                    {
                        "wm_images": [],
                        "env_images": init_env_images,
                        "env_dones": init_env_dones,
                        "executed_actions": [],
                        "pred_scores": [],
                    }
                )
                if is_valid:
                    valid_video[init_data["task_file_name"]].extend(
                        init_data.get("valid_images") or []
                    )
        except Exception:
            self._release_libero_env_handles(env_handles, healthy=False)
            _shutdown_env_workers(input_queues, processes, output_queues)
            raise

        step = 0
        vla_history = []

        def _mark_step_dummy(step_data: Dict[str, Any], idx: int, reason: str) -> None:
            dummy_flags = step_data.get("is_dummy", None)
            if isinstance(dummy_flags, torch.Tensor) and idx < dummy_flags.numel():
                dummy_flags.view(-1)[idx] = True
            prev_reason = str(step_data.get("placeholder_reason", "") or "")
            if reason and reason not in prev_reason.split(","):
                step_data["placeholder_reason"] = (
                    f"{prev_reason},{reason}" if prev_reason else reason
                )

        def _mark_env_failed(idx: int, step_data: Dict[str, Any], reason: str) -> None:
            task_records[idx]["active"] = False
            task_records[idx]["complete"] = False
            task_records[idx]["placeholder_reason"] = reason
            _mark_step_dummy(step_data, idx, reason)
            if isinstance(env_handles[idx], _LiberoEnvServiceClient):
                self._release_libero_env_service(env_handles[idx], healthy=False)
                env_handles[idx] = None
                return
            with contextlib.suppress(Exception):
                _shutdown_env_workers(
                    [input_queues[idx]],
                    [processes[idx]],
                    [output_queues[idx]],
                    join_timeout=5.0,
                )
            input_queues[idx] = None
            output_queues[idx] = None
            processes[idx] = None

        def _distributed_active_count(local_count: int) -> int:
            if not (dist.is_available() and dist.is_initialized()):
                return int(local_count)
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            count_tensor = torch.tensor([int(local_count)], device=device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            return int(count_tensor.item())

        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r["active"]]
            global_active_count = _distributed_active_count(len(active_indices))

            if global_active_count <= 0:
                if not vla_history:
                    vla_input = self.process_input(inputs, task_descriptions)
                    vla_input.update(meta_info)
                    vla_output = self._generate_one_step(vla_input)
                    actions = vla_output["action"]
                    device = (
                        actions.device
                        if isinstance(actions, torch.Tensor)
                        else vla_output["responses"].device
                    )
                    reasons = [
                        str(r.get("placeholder_reason", "") or "no_active_env")
                        for r in task_records
                    ]
                    vla_history.append(
                        {
                            "responses": vla_output["responses"],
                            "input_ids": vla_output["input_ids"],
                            "attention_mask": vla_output["attention_mask"],
                            "pixel_values": vla_output["pixel_values"],
                            "action": actions,
                            "step": step,
                            "is_dummy": torch.ones(
                                (batch_size,), dtype=torch.bool, device=device
                            ),
                            "placeholder_reason": ",".join(sorted(set(reasons))),
                        }
                    )
                break

            current_inputs = inputs
            current_task_descriptions = task_descriptions

            vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]

            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "step": step,
                "is_dummy": torch.tensor(
                    [not bool(record.get("active", False)) for record in task_records],
                    dtype=torch.bool,
                    device=(
                        actions.device
                        if isinstance(actions, torch.Tensor)
                        else vla_output["responses"].device
                    ),
                ),
            }
            placeholder_reasons = [
                str(record.get("placeholder_reason", "") or "")
                for record in task_records
                if not bool(record.get("active", False))
                and str(record.get("placeholder_reason", "") or "")
            ]
            if placeholder_reasons:
                step_data["placeholder_reason"] = ",".join(
                    sorted(set(placeholder_reasons))
                )
            vla_history.append(step_data)

            if not active_indices:
                step += self.config.action_chunks_len
                print(
                    f"Completed dummy-aligned step {step}/{max_steps}, "
                    f"local active environments: 0, global active environments: {global_active_count}"
                )
                continue

            submitted_indices = []
            for idx in active_indices:
                try:
                    if isinstance(env_handles[idx], _LiberoEnvServiceClient):
                        env_handles[idx].submit_step(actions[idx])
                    else:
                        input_queues[idx].put(actions[idx])
                    submitted_indices.append(idx)
                except Exception as exc:
                    reason = f"env_action_submit_failed:{_short_error(exc)}"
                    print(
                        f"[env_rollout] idx={idx} action submit failed; "
                        f"marking sample as dummy: {exc}",
                        flush=True,
                    )
                    _mark_env_failed(idx, step_data, reason)

            new_inputs = inputs.copy()
            for idx in submitted_indices:
                try:
                    if isinstance(env_handles[idx], _LiberoEnvServiceClient):
                        result = env_handles[idx].recv_step(
                            timeout=step_timeout_s,
                            context=f"LIBERO_PRO env service step idx={idx} step={step}",
                        )
                    else:
                        result = _get_worker_message(
                            output_queues[idx],
                            processes[idx],
                            timeout=step_timeout_s,
                            context=f"LIBERO_PRO env step idx={idx} step={step}",
                        )
                except Exception as exc:
                    reason = f"env_step_failed:{_short_error(exc)}"
                    print(
                        f"[env_rollout] idx={idx} worker step failed; "
                        f"marking sample as dummy: {exc}",
                        flush=True,
                    )
                    _mark_env_failed(idx, step_data, reason)
                    continue
                if result.get("type") == "error":
                    reason = f"env_step_error:{_short_error(result.get('error') or '')}"
                    print(
                        "LIBERO_PRO env rollout sample failed: "
                        f"idx={idx}, error={result.get('error')}\n"
                        f"{result.get('traceback', '')}",
                        flush=True,
                    )
                    _mark_env_failed(idx, step_data, reason)
                    continue
                if result.get("type") == "terminate":
                    task_records[idx]["active"] = False
                    task_records[idx]["complete"] = False
                    task_records[idx]["finish_step"] = min(
                        task_records[idx]["finish_step"]
                        + self.config.action_chunks_len,
                        max_steps,
                    )
                    _mark_step_dummy(step_data, idx, "env_terminated_without_step")
                    continue
                assert result["type"] == "step"
                new_inputs[idx] = self._obs_to_input(result["obs"], is_robotwin=False)
                task_records[idx]["active"] = result["active"]
                task_records[idx]["complete"] = result["complete"]
                task_records[idx]["finish_step"] = result["finish_step"]
                env_images = list(result.get("env_images", []))
                env_dones = list(result.get("env_dones", []))
                video_records[idx]["executed_actions"].extend(result.get("normed_actions", []))
                if env_images:
                    video_records[idx]["env_images"].extend(env_images)
                if env_dones:
                    video_records[idx]["env_dones"].extend(env_dones)
                if is_valid:
                    valid_video[task_records[idx]["task_file_name"]].extend(
                        result.get("valid_images") or []
                    )

            inputs = new_inputs
            step += self.config.action_chunks_len
            print(
                f"Completed step {step}/{max_steps}, active environments: {len(active_indices)}"
            )

        self._release_libero_env_handles(env_handles, healthy=True)
        _shutdown_env_workers(input_queues, processes, output_queues, join_timeout=20)

        torch.cuda.empty_cache()

        presentation_dir = getattr(self.config, "presentation_dir", None)
        if presentation_dir and meta_info.get("validate", False):
            from merl.episode_artifacts import save_episode
            keep = prompts.non_tensor_batch.get("evaluation_keep", np.ones(len(prompts), dtype=bool))
            for index, record in enumerate(task_records):
                if not bool(keep[index // n_samples]):
                    continue
                reason = str(record.get("placeholder_reason", "") or "")
                save_episode(
                    os.path.join(presentation_dir, f"step_{global_steps:06d}"),
                    video_records[index]["env_images"],
                    dict(task_id=int(task_id[index].item()), trial_id=int(trial_id[index].item()),
                         task_suite=str(task_suite_name[index]), instruction=task_descriptions[index],
                         success=bool(record["complete"]), environment_steps=int(record["finish_step"]),
                         valid=not reason and not bool(record.get("is_dummy", False)), failure_reason=reason,
                         max_steps=max_steps, global_step=int(global_steps), observation_source="real_environment",
                         protocol_id=str(getattr(self.config, "presentation_protocol", "")),
                         label=str(getattr(self.config, "presentation_label", self.config.experiment_name))),
                    executed_actions=video_records[index]["executed_actions"],
                )

        if is_valid:
            self._save_debug_rollout_videos(
                video_records,
                task_records,
                global_steps,
                save_env_tag="real_env",
            )

        self.module.train()

        output = self._prepare_output_batch_evolving(
            prompts,
            vla_history,
            task_records,
            task_descriptions,
            video_records,
            batch_size,
            max_steps,
        )
        if meta_info.get("paper_grounded_dir"):
            from merl.paper import save_grounded_trajectory
            import hashlib
            paths = []
            for index, (record, video) in enumerate(zip(task_records, video_records)):
                if record.get("placeholder_reason") or record.get("is_dummy"):
                    raise RuntimeError(f"Incomplete camera-ready grounded trajectory: {record}")
                if len(video["executed_actions"]) != int(record["finish_step"]):
                    raise RuntimeError("Executed-action count disagrees with environment transition count")
                uid = str(output.non_tensor_batch["uid"][index])
                name = hashlib.sha256(f"{uid}/{index}".encode()).hexdigest()[:16]
                path = Path(meta_info["paper_grounded_dir"]) / f"stage_{global_steps:06d}" / f"{name}.npz"
                paths.append(save_grounded_trajectory(
                    path, observations=video["env_images"], executed_actions=video["executed_actions"],
                    instruction=task_descriptions[index], success=record["complete"],
                    task_id=int(task_id[index].item()), trial_id=int(trial_id[index].item()), stage=global_steps))
            output.non_tensor_batch["paper_trajectory_path"] = np.asarray(paths, dtype=object)
        return output

    def _preprocess_img(self, img: np.ndarray) -> torch.Tensor:
        img: torch.Tensor = (
            torch.tensor(img).permute(2, 0, 1).float() / 255.0 * 2 - 1
        )  # [c, H, W]
        # resize H * W to h * w
        img = img.unsqueeze(0)  # [1, c, H, W]
        img = torch.nn.functional.interpolate(
            img,
            size=self.wm_args.img_resizes,  # [192, 320]
            mode="bilinear",
            align_corners=False,
        )  # [1, c, h, w]
        img = img.squeeze(0)  # [c, h, w]
        return img

    def _prepare_world_model_env_input(self, inputs, task_descriptions, actions):
        batches_inputs = []
        for i in range(len(inputs)):
            task_description = task_descriptions[i]
            action = torch.tensor(
                actions[i], dtype=torch.float32
            )  # [action_chunk, action_dim]
            input_data = inputs[i]
            image = np.array(
                Image.fromarray(input_data["full_image"]).convert("RGB")
            )  # [h, w, c]
            image = self._preprocess_img(image)  # returns torch.Tensor [c, h, w] float
            hist_images = []
            for hist_image in input_data.get("hist_images", []):
                hist_images.append(self._preprocess_img(np.array(hist_image)))
            if len(hist_images) > 0:
                hist_images = torch.stack(hist_images)  # [t_h, c, h, w]
            else:
                hist_images = torch.empty((0, *image.shape), dtype=image.dtype)
            hist_action = input_data.get("hist_action", [])
            batches_inputs.append(
                {
                    "task_description": task_description,
                    "image": image,  # torch.Tensor [c,h,w]
                    "action": action,  # torch.Tensor [t, action_dim]
                    "hist_images": hist_images,  # torch.Tensor or empty tensor
                    "hist_action": hist_action,  # list of np arrays
                }
            )
        return batches_inputs

    #! 2. generate_minibatch: pure world model mode
    def _generate_minibatch_libero_wm(self, prompts: DataProto) -> DataProto:
        """Generate minibatch for Libero using multiprocessing"""
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get("n_samples", 1)
        task_id = prompts.batch["task_id"].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch["trial_id"].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(
            prompts.non_tensor_batch["task_suite_name"], n_samples
        )
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        # is_valid = meta_info.get('n_samples') is None
        #! tmp: save video when training
        is_valid = True
        global_steps = meta_info.get("global_steps", 0) if is_valid else 0

        processes = []
        input_queues = []
        output_queues = []

        # Create processes for each environment (batch)
        for idx in range(batch_size):
            task_name = task_suite_name[idx]
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            input_q = Queue()
            output_q = Queue()
            p = Process(
                target=env_worker_wm,
                args=(
                    self.world_model,
                    self.wm_args,
                    self.rm_threshold,
                    self.device,
                    task_name,
                    t_id,
                    tr_id,
                    self.config,
                    input_q,
                    output_q,
                    is_valid,
                    global_steps,
                    max_steps,
                    True,
                ),
            )
            p.start()
            processes.append(p)
            input_queues.append(input_q)
            output_queues.append(output_q)

        # Initialize env, obs
        inputs = []
        task_descriptions = []
        task_records: List[Dict[str, Any]] = []
        video_records: List[List[np.ndarray]] = []
        valid_video: Dict[str, List] = defaultdict(list)  # {task_file_name: images}
        for idx in range(batch_size):
            init_data = _get_worker_message(
                output_queues[idx],
                processes[idx],
                timeout=init_timeout_s,
                context=f"WM env init idx={idx}",
            )
            if init_data.get("type") == "error":
                _shutdown_env_workers(input_queues, processes, output_queues)
                raise RuntimeError(
                    "WM env initialization failed: "
                    f"{init_data.get('error')}\n{init_data.get('traceback', '')}"
                )
            assert init_data["type"] == "init"
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data["obs"], is_robotwin=False))
            task_records.append(
                {
                    "active": init_data["active"],
                    "complete": init_data["complete"],
                    "finish_step": init_data["finish_step"],
                    "task_file_name": init_data["task_file_name"],
                }
            )
            video_records.append(init_data.get("valid_images") or [])
            if is_valid:
                valid_video[init_data["task_file_name"]].extend(
                    init_data.get("valid_images") or []
                )

        # Start rollout with wm
        step = 0
        vla_history = []
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r["active"]]

            current_inputs: List[Dict[str, Any]] = inputs
            current_task_descriptions = task_descriptions

            vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output[
                "action"
            ]  # np.array, [batch_size, action_chunk_size, action_dim]

            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "step": step,
            }
            vla_history.append(step_data)

            batches_inputs = self._prepare_world_model_env_input(
                current_inputs, task_descriptions, actions
            )
            for idx in active_indices:
                # Input batches_inputs into env at the same time, and get the batches_outputs
                input_queues[idx].put(batches_inputs[idx])

            # Prepare next-step inputs from currrent-step outputs
            new_inputs = inputs.copy()
            for idx in active_indices:
                result = _get_worker_message(
                    output_queues[idx],
                    processes[idx],
                    timeout=15,
                    context=f"WM env step idx={idx} step={step}",
                )
                if result.get("type") == "error":
                    _shutdown_env_workers(input_queues, processes)
                    raise RuntimeError(
                        "WM env rollout failed: "
                        f"{result.get('error')}\n{result.get('traceback', '')}"
                    )
                assert result["type"] == "step"
                new_inputs[idx] = self._result_to_input(
                    result, is_robotwin=False
                )  # self._img_to_input
                task_records[idx]["active"] = result["active"]
                task_records[idx]["complete"] = result["complete"]
                task_records[idx]["finish_step"] = result["finish_step"]
                video_records[idx].extend(result.get("valid_images") or [])
                if is_valid:
                    valid_video[task_records[idx]["task_file_name"]].extend(
                        result.get("valid_images") or []
                    )

            inputs = new_inputs
            step += self.config.action_chunks_len
            print(
                f"Completed step {step}/{max_steps}, active environments: {len(active_indices)}"
            )

        # Terminate processes
        for q in input_queues:
            q.put(None)
        for p in processes:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
        torch.cuda.empty_cache()

        # if is_valid:
        if is_valid and global_steps > 0:
            for task_file, images in valid_video.items():
                complete = any(
                    r["complete"]
                    for r in task_records
                    if r["task_file_name"] == task_file
                )
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete,
                )

        self.module.train()
        return self._prepare_output_batch_wm(
            prompts,
            vla_history,
            task_records,
            video_records,
            batch_size,
            max_steps,
        )

    #! 3.v1 generate_minibatch: evolving mode
    def episode_to_vla_entry(self, episode, task_idx=0):
        """
        Convert env-side episode (list of transitions) to a vla-style GT entry dictionary.
        - Tries to call self.action_to_responses_tokens(actions) if exists to produce tokenized 'responses'/'input_ids' etc;
        otherwise returns 'gt_actions' and 'pixel_values' so downstream can use BC or conversion.
        - Keeps raw 'episode' for full fidelity.
        """
        # collect actions and pixel observations
        actions = []
        pixel_values = []
        for tr in episode:
            obs, action, reward, done, info = tr
            actions.append(action)
            if isinstance(obs, dict) and "agentview_image" in obs:
                pixel_values.append(obs["agentview_image"])
            else:
                # fallback blank image sized 256x256 if not present
                pixel_values.append(np.zeros((256, 256, 3), dtype=np.uint8))

        # try to produce tokenized responses if the user has such a helper
        responses = None
        input_ids = None
        attention_mask = None
        try:
            # try instance method first
            if hasattr(self, "action_to_responses_tokens") and callable(
                self.action_to_responses_tokens
            ):
                responses, input_ids, attention_mask = self.action_to_responses_tokens(
                    actions
                )
            # try actor_module method name variants
            elif hasattr(self.actor_module, "action_to_responses_tokens") and callable(
                self.actor_module.action_to_responses_tokens
            ):
                responses, input_ids, attention_mask = (
                    self.actor_module.action_to_responses_tokens(actions)
                )
            elif hasattr(self.actor_module, "actions_to_tokens") and callable(
                self.actor_module.actions_to_tokens
            ):
                responses, input_ids, attention_mask = (
                    self.actor_module.actions_to_tokens(actions)
                )
            else:
                # no tokenization available; leave responses None
                responses = None
        except Exception as e:
            # tokenization failed for some reason; fallback to raw actions
            responses = None

        entry = {
            "is_gt": True,
            "episode": episode,  # raw transitions for full fidelity
            "gt_actions": actions,  # raw action vectors (list)
            "pixel_values": pixel_values,  # list of images (np arrays)
            "responses": responses,  # tokenized responses if available, else None
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return entry

    # -------------------------
    # 修改后的 _generate_minibatch_libero_evolving
    # -------------------------
    def _generate_minibatch_libero_evolving(self, prompts: DataProto) -> DataProto:
        """Generate minibatch for Libero using multiprocessing (evolving WM).
        最小改动说明：
        - 修复 KeyError: 'pixel_values'：在 vla_input 构造中强制保证 pixel_values 存在；
        - 在 vla_input 构造后保证最小 schema（占位 input_ids/attention_mask/pixel_values）并返回 used_ph 标志与 reason；
        - 在 vla_output 收到后保证最小 schema 并返回 vla_minimal 标志与 reason；
        - step 级别合成 is_dummy = used_ph or vla_minimal，并写入 placeholder_reason 字符串；
        - except 兜底时将 last_attempt_state 中的所有 step 标记为 dummy 并追加 reason；
        - 在返回前统计 chosen_state 的 dummy_ratio 并写入 task_records[0]["dummy_ratio"]（便于监控）。
        """
        from multiprocessing import Process, Queue
        import queue as _queue
        import time
        from collections import defaultdict
        import math
        import numpy as np
        import torch
        import torch.distributed as dist
        import traceback

        # === PATCH INPUT: ensure vla_input contains minimal keys expected by _generate_one_step_wm ===
        def _ensure_vla_input_schema(
            vla_input, batch_size, device, reference_pixel_shape=None
        ):
            """
            Guarantee vla_input has minimal keys expected by _generate_one_step_oft_wm.
            reference_pixel_shape: (B, C, H, W) tuple to ensure channel consistency.
            Returns: (vla_input_normalized, used_ph_flag, reason_str)
            """
            used_placeholder = False
            reasons = []
            if not isinstance(vla_input, dict):
                vla_input = {}
                used_placeholder = True
                reasons.append("input_dict_missing")

            # input_ids placeholder
            if "input_ids" not in vla_input or vla_input.get("input_ids") is None:
                try:
                    vla_input["input_ids"] = torch.zeros(
                        (batch_size, 1), dtype=torch.long, device=device
                    )
                except Exception:
                    vla_input["input_ids"] = torch.zeros(
                        (batch_size, 1), dtype=torch.long
                    )
                vla_input["__placeholder_input_ids__"] = True
                used_placeholder = True
                reasons.append("missing_input_ids")

            # attention_mask placeholder
            if (
                "attention_mask" not in vla_input
                or vla_input.get("attention_mask") is None
            ):
                try:
                    vla_input["attention_mask"] = torch.ones_like(
                        vla_input["input_ids"],
                        dtype=torch.long,
                        device=vla_input["input_ids"].device,
                    )
                except Exception:
                    vla_input["attention_mask"] = torch.ones(
                        (batch_size, 1), dtype=torch.long
                    )
                used_placeholder = True
                reasons.append("missing_attention_mask")

            # === FIX: pixel_values placeholder with dynamic channel inference ===
            if "pixel_values" not in vla_input or vla_input.get("pixel_values") is None:
                try:
                    # Determine channel count: prefer reference shape, else default to 6 (for split_sizes=[3,3])
                    if (
                        reference_pixel_shape is not None
                        and len(reference_pixel_shape) == 4
                    ):
                        C = reference_pixel_shape[1]
                        H = reference_pixel_shape[2]
                        W = reference_pixel_shape[3]
                    else:
                        # Default to 6 channels to match split_sizes=[3, 3] error pattern
                        C, H, W = 6, 224, 224
                    vla_input["pixel_values"] = torch.zeros(
                        (batch_size, C, H, W), dtype=torch.float32, device=device
                    )
                except Exception:
                    vla_input["pixel_values"] = torch.zeros(
                        (batch_size, 6, 224, 224), dtype=torch.float32
                    )
                used_placeholder = True
                reasons.append("missing_pixel_values")

            return vla_input, used_placeholder, ",".join(reasons)

        # === PATCH OUTPUT: ensure vla_output schema ===
        def _ensure_vla_output_schema(vla_output, batch_size, device):
            """
            Ensure vla_output contains expected keys. Return (vla_output_norm, minimal_flag, reason_str)
            minimal_flag True 表示某些关键字段缺失（需要上层决定如何处理）
            """
            minimal = False
            reasons = []
            if not isinstance(vla_output, dict):
                vla_output = {}
                minimal = True
                reasons.append("output_not_dict")
            required = [
                "responses",
                "input_ids",
                "attention_mask",
                "pixel_values",
                "action",
            ]
            for k in required:
                if k not in vla_output or vla_output[k] is None:
                    vla_output[k] = None
                    minimal = True
                    reasons.append(f"missing_{k}")
            # Sanity check action type
            if vla_output.get("action") is not None:
                a = vla_output["action"]
                if not isinstance(a, (torch.Tensor, np.ndarray, list)):
                    vla_output["action"] = None
                    minimal = True
                    reasons.append("bad_action_type")
            return vla_output, minimal, ",".join(reasons)

        # === prepare ===
        prompts = self.sanitize_task_ids(
            prompts=prompts, allowed_ids=self._configured_allowed_task_ids()
        )
        all_ood_task_descriptions = self._preprocess_libero_pro_task_suite(prompts)
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get("n_samples", 1)
        print(f"[wm_rollout] n_samples: {n_samples}")
        task_id = prompts.batch["task_id"].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch["trial_id"].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(
            prompts.non_tensor_batch["task_suite_name"], n_samples
        )
        max_steps = int(
            meta_info.get(
                "max_steps", self.max_steps.get(self.config.task_suite_name, 800)
            )
        )
        batch_size = int(task_id.size(0))
        print(f"[wm_rollout] batch_size: {batch_size}")
        assert batch_size > 0, "Local batch is empty!"
        assert batch_size == 1, "WM rollout does not support local multiprocessing."
        is_valid = True
        global_steps = meta_info.get("global_steps", 0) if is_valid else 0
        max_attempts = getattr(self.config, "max_success_attempts", 5)
        fallback_after = getattr(self.config, "fallback_after_attempts", 3)
        action_chunks_len = self.config.action_chunks_len
        steps_per_attempt = int(math.ceil(float(max_steps) / float(action_chunks_len)))
        print(
            f"[wm_rollout] steps_per_attempt: {steps_per_attempt}, max_attempts: {max_attempts}, fallback_after: {fallback_after}"
        )
        best_success_state = None
        last_attempt_state = None
        last_exception = None
        global_success = False
        # store last vla_history for possible fallback; vla_history items KEEP tensors on original device for downstream
        mp_ctx = _get_env_mp_context(self.config)
        worker_config = _build_libero_env_worker_config(self.config)
        init_timeout_s = float(getattr(worker_config, "env_init_timeout_s", 300.0))
        step_timeout_s = float(getattr(worker_config, "env_step_timeout_s", 60.0))
        init_retry_attempts = int(
            getattr(worker_config, "env_init_parent_max_retry", 1)
        )
        use_env_service = self._libero_env_service_enabled()
        for attempt in range(1, max_attempts + 1):
            print(
                f"[wm_rollout] attempt {attempt}/{max_attempts} (global_success={global_success})",
                flush=True,
            )
            do_dummy_attempt = bool(global_success)
            processes = []
            input_queues = []
            output_queues = []
            env_handles = []
            # For this attempt we accumulate vla_history (KEEP device/dtype)
            vla_history = []
            task_descriptions = []
            task_records = []
            video_records = []
            valid_video = defaultdict(list)
            try:
                if not do_dummy_attempt:
                    # --- spawn envs and perform steps_per_attempt forwards ---
                    for idx in range(batch_size):
                        task_name = task_suite_name[idx]
                        t_id = int(task_id[idx][0].item())
                        tr_id = int(trial_id[idx][0].item())
                        ood_task_description = self._select_ood_task_description(
                            all_ood_task_descriptions,
                            sample_idx=idx,
                            n_samples=n_samples,
                            task_id=t_id,
                        )
                        print(
                            f"t_id: {t_id}, ood_task_description: {ood_task_description}"
                        )
                        env_handle = None
                        try:
                            if use_env_service:
                                env_handle = self._acquire_libero_env_service(
                                    mp_ctx=mp_ctx,
                                    worker_config=worker_config,
                                    init_timeout_s=init_timeout_s,
                                    step_timeout_s=step_timeout_s,
                                )
                                init_data = env_handle.reset(
                                    task_name=task_name,
                                    task_id=t_id,
                                    trial_id=tr_id,
                                    is_valid=is_valid,
                                    global_steps=global_steps,
                                    max_steps=max_steps,
                                    ood_task_description=ood_task_description,
                                )
                                process, input_q, output_q = None, None, None
                            else:
                                process, input_q, output_q, init_data = (
                                    _launch_libero_env_worker_with_retry(
                                        mp_ctx=mp_ctx,
                                        target=env_worker_evolving_envonly_v1,
                                        args_factory=lambda iq, oq, task_name=task_name, t_id=t_id, tr_id=tr_id, ood_task_description=ood_task_description: (
                                            task_name,
                                            t_id,
                                            tr_id,
                                            worker_config,
                                            iq,
                                            oq,
                                            is_valid,
                                            global_steps,
                                            max_steps,
                                            ood_task_description,
                                        ),
                                        worker_label=f"LIBERO_PRO evolving idx={idx} task={task_name} trial={tr_id}",
                                        init_timeout_s=init_timeout_s,
                                        max_attempts=init_retry_attempts,
                                        worker_config=worker_config,
                                    )
                                )
                        except Exception as exc:
                            self._release_libero_env_service(env_handle, healthy=False)
                            env_handle = None
                            process, input_q, output_q = None, None, None
                            init_data = {"type": "error", "error": str(exc)}

                        processes.append(process)
                        input_queues.append(input_q)
                        output_queues.append(output_q)
                        env_handles.append(env_handle)

                        if init_data.get("type") == "error":
                            print(
                                f"[rollout] child init error idx={idx}: {init_data.get('error')}",
                                flush=True,
                            )
                            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
                            placeholder_obs = {"agentview_image": dummy_image}
                            task_name = task_suite_name[idx]
                            t_id = int(task_id[idx][0].item())
                            tr_id = int(trial_id[idx][0].item())
                            placeholder_step = {
                                "type": "init",
                                "task_description": "",
                                "obs": placeholder_obs,
                                "image": dummy_image,
                                "valid_images": [],
                                "env_images": [],
                                "env_dones": [],
                                "normed_actions": None,
                                "active": False,
                                "complete": False,
                                "finish_step": 0,
                                "task_file_name": f"{task_name}_task_{t_id}_trial_{tr_id}",
                            }
                            init_data = placeholder_step
                        assert (
                            init_data["type"] == "init"
                        ), f"Unexpected init_data type: {init_data.get('type')}"
                        task_descriptions.append(init_data["task_description"])
                        inputs = [self._result_to_input(init_data, is_robotwin=False)]
                        task_records.append(
                            {
                                "active": bool(init_data.get("active", False)),
                                "complete": bool(init_data.get("complete", False)),
                                "finish_step": int(init_data.get("finish_step", 0)),
                                "task_file_name": init_data.get(
                                    "task_file_name", f"task_{idx}"
                                ),
                            }
                        )
                        video_records.append(
                            {
                                "wm_images": [],
                                "env_images": list(init_data.get("env_images", [])),
                                "env_dones": list(init_data.get("env_dones", [])),
                                "pred_scores": [],
                            }
                        )
                        if is_valid:
                            valid_video[init_data["task_file_name"]].extend(
                                init_data.get("valid_images") or []
                            )
                    # run fixed number of steps
                    step = 0
                    while step < steps_per_attempt:
                        current_inputs = inputs
                        current_task_descriptions = task_descriptions
                        vla_input = self.process_input(
                            current_inputs, current_task_descriptions
                        )
                        vla_input.update(meta_info)
                        # === PATCH INPUT: ensure vla_input has minimal expected keys ===
                        try:
                            vla_input, used_ph, used_ph_reason = (
                                _ensure_vla_input_schema(
                                    vla_input, batch_size, self.device
                                )
                            )
                            if used_ph:
                                print(
                                    f"[wm_rollout] WARNING: using placeholder input_ids/attention_mask/pixel_values in attempt {attempt}, step {step} reason={used_ph_reason}",
                                    flush=True,
                                )
                        except Exception:
                            print(
                                "[wm_rollout] _ensure_vla_input_schema failed:",
                                flush=True,
                            )
                            traceback.print_exc()
                            # === FIX: Manual fallback must include pixel_values ===
                            vla_input = {
                                "input_ids": torch.zeros(
                                    (batch_size, 1),
                                    dtype=torch.long,
                                    device=self.device,
                                ),
                                "attention_mask": torch.ones(
                                    (batch_size, 1),
                                    dtype=torch.long,
                                    device=self.device,
                                ),
                                "pixel_values": torch.zeros(
                                    (batch_size, 6, 224, 224),
                                    dtype=torch.float32,
                                    device=self.device,
                                ),
                            }
                            used_ph = True
                            used_ph_reason = "ensure_schema_exception"
                        # --- ensure deterministic dtype across ranks during forward ---
                        with torch.cuda.amp.autocast(enabled=False):
                            vla_output = self._generate_one_step_wm(vla_input)
                        # === PATCH OUTPUT: ensure schema and mark minimal if necessary ===
                        try:
                            vla_output, vla_minimal, vla_minimal_reason = (
                                _ensure_vla_output_schema(
                                    vla_output, batch_size, self.device
                                )
                            )
                            if vla_minimal:
                                print(
                                    f"[wm_rollout] WARNING: vla_output missing fields in attempt {attempt}, step {step} reason={vla_minimal_reason}",
                                    flush=True,
                                )
                        except Exception:
                            print(
                                "[wm_rollout] _ensure_vla_output_schema failed:",
                                flush=True,
                            )
                            traceback.print_exc()
                            vla_output = {
                                "responses": None,
                                "input_ids": None,
                                "attention_mask": None,
                                "pixel_values": None,
                                "action": None,
                            }
                            vla_minimal = True
                            vla_minimal_reason = "ensure_output_exception"
                        # IMPORTANT: DO NOT move logits/labels/actions to CPU here.
                        # Keep vla_output tensors on their original device/dtype for downstream training computations.
                        actions = vla_output.get("action")
                        # === PATCH: build placeholder_reason & is_dummy per step ===
                        is_dummy_step = bool(used_ph) or bool(vla_minimal)
                        placeholder_reasons = []
                        if used_ph:
                            placeholder_reasons.append(
                                used_ph_reason or "input_placeholder"
                            )
                        if vla_minimal:
                            placeholder_reasons.append(
                                vla_minimal_reason or "output_minimal"
                            )
                        step_data = {
                            "responses": vla_output.get("responses"),
                            "input_ids": vla_output.get("input_ids"),
                            "attention_mask": vla_output.get("attention_mask"),
                            "pixel_values": vla_output.get("pixel_values"),
                            "action": actions,  # 确保 actions 不为 None
                            "step": (attempt - 1) * steps_per_attempt + step,
                            # step-level is_dummy (tensor of shape (batch_size,))
                            "is_dummy": torch.full(
                                (batch_size,),
                                bool(is_dummy_step),
                                dtype=torch.bool,
                                device=self.device,
                            ),
                            "placeholder_reason": (
                                ",".join(placeholder_reasons)
                                if placeholder_reasons
                                else ""
                            ),
                        }
                        vla_history.append(step_data)
                        # === end PATCH ===
                        # prepare WM inputs
                        batches_inputs = self._prepare_world_model_env_input(
                            current_inputs, current_task_descriptions, actions
                        )
                        # send actions to env child for active indices
                        active_indices = [
                            i for i, r in enumerate(task_records) if r["active"]
                        ]
                        for idx in active_indices:
                            try:
                                a_to_send = (
                                    actions[idx] if actions is not None else None
                                )
                                if a_to_send is None:
                                    # 没有有效 action：将该 env 标记为 inactive，避免向子进程发送非法数据
                                    task_records[idx]["active"] = False
                                    continue
                                if isinstance(a_to_send, torch.Tensor):
                                    a_to_send_to_put = a_to_send.detach().cpu().numpy()
                                else:
                                    a_to_send_to_put = a_to_send
                                if isinstance(
                                    env_handles[idx], _LiberoEnvServiceClient
                                ):
                                    env_handles[idx].submit_step(a_to_send_to_put)
                                else:
                                    input_queues[idx].put(a_to_send_to_put)
                            except Exception as e:
                                print(
                                    f"[rollout Error] failed to put action to child {idx}: {e}",
                                    flush=True,
                                )
                                if isinstance(
                                    env_handles[idx], _LiberoEnvServiceClient
                                ):
                                    self._release_libero_env_service(
                                        env_handles[idx], healthy=False
                                    )
                                    env_handles[idx] = None
                                task_records[idx]["active"] = False
                        # compute WM predictions
                        active_batches = [batches_inputs[idx] for idx in active_indices]
                        wm_outs = []
                        if len(active_batches) > 0:
                            wm_outs = (
                                self.compute_wm_predictions_on_main(active_batches)
                                or []
                            )
                        if len(wm_outs) < len(active_batches):
                            wm_outs = list(wm_outs) + [{}] * (
                                len(active_batches) - len(wm_outs)
                            )
                        # collect env outputs
                        new_inputs = inputs.copy()
                        for out_i, idx in enumerate(active_indices):
                            try:
                                if isinstance(
                                    env_handles[idx], _LiberoEnvServiceClient
                                ):
                                    result = env_handles[idx].recv_step(
                                        timeout=step_timeout_s,
                                        context=f"LIBERO_PRO evolving service step idx={idx} step={step}",
                                    )
                                else:
                                    result = _get_worker_message(
                                        output_queues[idx],
                                        processes[idx],
                                        timeout=step_timeout_s,
                                        context=f"LIBERO_PRO evolving step idx={idx} step={step}",
                                    )
                            except RuntimeError as exc:
                                print(
                                    f"[rollout] child {idx} step wait failed: {exc}; marking inactive",
                                    flush=True,
                                )
                                if isinstance(
                                    env_handles[idx], _LiberoEnvServiceClient
                                ):
                                    self._release_libero_env_service(
                                        env_handles[idx], healthy=False
                                    )
                                    env_handles[idx] = None
                                result = {"type": "terminate"}
                            if result.get("type") == "terminate":
                                task_records[idx]["active"] = False
                                task_records[idx]["complete"] = False
                                task_records[idx]["finish_step"] = min(
                                    task_records[idx]["finish_step"]
                                    + action_chunks_len,
                                    max_steps,
                                )
                                env_images = []
                                env_dones = []
                            elif result.get("type") == "error":
                                print(
                                    f"[rollout] child {idx} returned error during step: {result.get('error')}",
                                    flush=True,
                                )
                                if isinstance(
                                    env_handles[idx], _LiberoEnvServiceClient
                                ):
                                    self._release_libero_env_service(
                                        env_handles[idx], healthy=False
                                    )
                                    env_handles[idx] = None
                                task_records[idx]["active"] = False
                                task_records[idx]["complete"] = False
                                env_images = result.get("env_images", [])
                                env_dones = (
                                    result.get("env_dones", [])
                                    if "env_dones" in result
                                    else [False] * len(env_images)
                                )
                                if "obs" in result:
                                    new_inputs[idx] = self._result_to_input(
                                        result, is_robotwin=False
                                    )
                                else:
                                    new_inputs[idx] = self._result_to_input(
                                        result, is_robotwin=False
                                    )
                            else:  # === 正常 "step" ===
                                env_images = result.get("env_images", [])
                                env_dones = result.get("env_dones", [])
                                if "obs" in result:
                                    new_inputs[idx] = self._result_to_input(
                                        result, is_robotwin=False
                                    )
                                else:
                                    new_inputs[idx] = self._result_to_input(
                                        result, is_robotwin=False
                                    )
                            wm_out = wm_outs[out_i] if out_i < len(wm_outs) else {}
                            pred_images_list = wm_out.get("pred_images_list", [])
                            for im in pred_images_list:
                                video_records[idx]["wm_images"].append(
                                    resize_to_libero_image(im, (256, 256))
                                )
                            pred_scores = wm_out.get("pred_scores", [])
                            if pred_scores is not None and len(pred_scores) > 0:
                                video_records[idx].setdefault("pred_scores", []).extend(
                                    np.asarray(pred_scores, dtype=np.float32)
                                    .reshape(-1)
                                    .tolist()
                                )
                            video_records[idx]["env_images"].extend(env_images)
                            video_records[idx]["env_dones"].extend(env_dones)
                            for env_done in env_dones:
                                task_records[idx]["finish_step"] += 1
                                if (
                                    env_done
                                    or task_records[idx]["finish_step"] >= max_steps
                                ):
                                    task_records[idx]["active"] = False
                                    task_records[idx]["complete"] = bool(env_done)
                                    break
                        inputs = new_inputs
                        step += 1
                    # end steps_per_attempt for real attempt
                    # cleanup processes
                    self._release_libero_env_handles(env_handles, healthy=True)
                    _shutdown_env_workers(
                        input_queues,
                        processes,
                        output_queues,
                        join_timeout=20,
                    )
                    torch.cuda.empty_cache()
                    # save last_attempt_state (KEEP tensors on device)
                    last_attempt_state = {
                        "vla_history": vla_history,
                        "task_records": task_records,
                        "task_descriptions": task_descriptions,
                        "video_records": video_records,
                        "batch_size": batch_size,
                        "max_steps": max_steps,
                    }
                    # determine local success
                    local_success = (
                        1
                        if (task_records and task_records[0].get("complete", False))
                        else 0
                    )
                else:
                    # --- dummy attempt: no env spawn, run steps_per_attempt forwards to keep collectives in sync ---
                    # === FIX: Get reference pixel shape from best_success_state if available ===
                    ref_pixel_shape = None
                    if best_success_state is not None:
                        try:
                            prev_hist = best_success_state.get("vla_history", [])
                            if len(prev_hist) > 0:
                                prev_step = prev_hist[0]
                                prev_pv = prev_step.get("pixel_values")
                                if isinstance(prev_pv, torch.Tensor):
                                    ref_pixel_shape = tuple(prev_pv.shape)
                                    print(
                                        f"[wm_rollout] Dummy attempt using reference pixel_shape={ref_pixel_shape} from best_success_state",
                                        flush=True,
                                    )
                        except Exception:
                            pass

                    try:
                        placeholder_obs = {
                            "agentview_image": np.zeros((224, 224, 3), dtype=np.uint8)
                        }
                        vla_input = self.process_input(
                            [
                                self._result_to_input(
                                    {"type": "init", "obs": placeholder_obs},
                                    is_robotwin=False,
                                )
                            ],
                            [
                                self._select_ood_task_description(
                                    all_ood_task_descriptions,
                                    sample_idx=0,
                                    n_samples=n_samples,
                                    task_id=int(task_id[0][0].item()),
                                )
                            ],
                        )
                        vla_input.update(meta_info)
                    except Exception:
                        vla_input = {}
                        vla_input.update(meta_info)
                    for step in range(steps_per_attempt):
                        # === PATCH INPUT (dummy): ensure keys ===
                        try:
                            vla_input, used_ph, used_ph_reason = (
                                _ensure_vla_input_schema(
                                    vla_input, batch_size, self.device, ref_pixel_shape
                                )
                            )
                            if used_ph:
                                print(
                                    f"[wm_rollout] WARNING: using placeholder input_ids/attention_mask/pixel_values (dummy) in attempt {attempt}, step {step} reason={used_ph_reason}",
                                    flush=True,
                                )
                        except Exception:
                            print(
                                "[wm_rollout] _ensure_vla_input_schema failed (dummy):",
                                flush=True,
                            )
                            traceback.print_exc()
                            # === FIX: Manual fallback must include pixel_values with 6 channels ===
                            vla_input = {
                                "input_ids": torch.zeros(
                                    (batch_size, 1),
                                    dtype=torch.long,
                                    device=self.device,
                                ),
                                "attention_mask": torch.ones(
                                    (batch_size, 1),
                                    dtype=torch.long,
                                    device=self.device,
                                ),
                                "pixel_values": torch.zeros(
                                    (batch_size, 6, 224, 224),
                                    dtype=torch.float32,
                                    device=self.device,
                                ),
                            }
                            used_ph = True
                            used_ph_reason = "ensure_schema_exception_dummy"
                        # === FIX: Wrap VLA forward in try-except for dummy attempt stability ===
                        try:
                            with torch.cuda.amp.autocast(enabled=False):
                                vla_output = self._generate_one_step_wm(vla_input)
                        except Exception as e:
                            print(
                                f"[wm_rollout] VLA forward failed in dummy attempt {attempt}, step {step}: {e}",
                                flush=True,
                            )
                            # Create minimal valid output to keep collectives in sync
                            vla_output = {
                                "responses": None,
                                "input_ids": None,
                                "attention_mask": None,
                                "pixel_values": None,
                                "action": None,
                            }
                        try:
                            vla_output, vla_minimal, vla_minimal_reason = (
                                _ensure_vla_output_schema(
                                    vla_output, batch_size, self.device
                                )
                            )
                            if vla_minimal:
                                print(
                                    f"[wm_rollout] WARNING: vla_output missing fields (dummy) in attempt {attempt}, step {step} reason={vla_minimal_reason}",
                                    flush=True,
                                )
                        except Exception:
                            print(
                                "[wm_rollout] _ensure_vla_output_schema failed (dummy):",
                                flush=True,
                            )
                            traceback.print_exc()
                            vla_output = {
                                "responses": None,
                                "input_ids": None,
                                "attention_mask": None,
                                "pixel_values": None,
                                "action": None,
                            }
                            vla_minimal = True
                            vla_minimal_reason = "ensure_output_exception_dummy"
                        # dummy attempt: consider these steps as dummy
                        placeholder_reasons = []
                        if used_ph:
                            placeholder_reasons.append(
                                used_ph_reason or "input_placeholder"
                            )
                        if vla_minimal:
                            placeholder_reasons.append(
                                vla_minimal_reason or "output_minimal"
                            )
                        placeholder_reasons.append("dummy_attempt")
                        vla_history.append(
                            {
                                "responses": vla_output.get("responses"),
                                "input_ids": vla_output.get("input_ids"),
                                "attention_mask": vla_output.get("attention_mask"),
                                "pixel_values": vla_output.get("pixel_values"),
                                "action": vla_output.get("action"),
                                "step": (attempt - 1) * steps_per_attempt + step,
                                "is_dummy": torch.ones(
                                    batch_size, dtype=torch.bool, device=self.device
                                ),
                                "placeholder_reason": ",".join(placeholder_reasons),
                            }
                        )
                        # record last_attempt_state placeholder for dummy attempt
                        last_attempt_state = {
                            "vla_history": vla_history,
                            "task_records": task_records
                            or [
                                {
                                    "active": False,
                                    "complete": False,
                                    "finish_step": 0,
                                    "task_file_name": f"{task_suite_name[0]}_task_{int(task_id[0][0].item())}_trial_{int(trial_id[0][0].item())}",
                                }
                            ],
                            "task_descriptions": task_descriptions
                            or [
                                self._select_ood_task_description(
                                    all_ood_task_descriptions,
                                    sample_idx=0,
                                    n_samples=n_samples,
                                    task_id=int(task_id[0][0].item()),
                                )
                            ],
                            "video_records": video_records
                            or [
                                {
                                    "wm_images": [],
                                    "env_images": [],
                                    "env_dones": [],
                                    "pred_scores": [],
                                }
                            ],
                            "batch_size": batch_size,
                            "max_steps": max_steps,
                        }
                        local_success = 0
                # === synchronize success across ranks ===
                if dist.is_available() and dist.is_initialized():
                    t = torch.tensor(
                        [local_success], device=self.device, dtype=torch.long
                    )
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    any_rank_succeeded = int(t.item()) > 0
                    global_success = bool(any_rank_succeeded)
                else:
                    global_success = bool(local_success)
                # if local success and best_success_state not set, store it (first find)
                if local_success == 1 and best_success_state is None:
                    best_success_state = last_attempt_state
                # --- NEW: if we've reached fallback threshold and no rank succeeded, ensure all ranks break consistently ---
                if attempt >= fallback_after and not global_success:
                    if dist.is_available() and dist.is_initialized():
                        try:
                            br = torch.tensor([1], device=self.device, dtype=torch.long)
                            # if any rank reaches this condition，we want all ranks to break
                            dist.all_reduce(br, op=dist.ReduceOp.MAX)
                            if int(br.item()) == 1:
                                # ensure all ranks see the barrier
                                try:
                                    dist.barrier()
                                except Exception:
                                    pass
                                print(
                                    f"[wm_rollout] reached fallback_after ({fallback_after}) with no success on any rank; using last_attempt_state as fallback.",
                                    flush=True,
                                )
                                break
                        except Exception:
                            # fallback to local break if collectives fail
                            print(
                                "[wm_rollout] dist check failed while synchronizing fallback break; breaking locally.",
                                flush=True,
                            )
                            break
                    else:
                        print(
                            f"[wm_rollout] reached fallback_after ({fallback_after}) with no success on any rank; using last_attempt_state as fallback.",
                            flush=True,
                        )
                        break
                # otherwise continue to next attempt (all ranks will either do real or dummy attempt based on global_success)
                continue
            except Exception as e:
                last_exception = e
                print(
                    f"[wm_rollout] exception during attempt {attempt}: {e}", flush=True
                )
                traceback.print_exc()
                # ensure processes cleaned
                self._release_libero_env_handles(env_handles, healthy=False)
                _shutdown_env_workers(
                    input_queues,
                    processes,
                    output_queues,
                    join_timeout=5.0,
                )
                # synchronize failure (avoid inconsistent global_success)
                local_success = 0
                if dist.is_available() and dist.is_initialized():
                    t = torch.tensor(
                        [local_success], device=self.device, dtype=torch.long
                    )
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    global_success = bool(int(t.item()) > 0)
                else:
                    global_success = False
                # === PATCH EXCEPT: ensure last_attempt_state safe placeholder if None ===
                if last_attempt_state is None:
                    try:
                        last_attempt_state = {
                            "vla_history": [],
                            "task_records": [
                                {
                                    "active": False,
                                    "complete": False,
                                    "finish_step": 0,
                                    "task_file_name": f"{task_suite_name[0]}_task_{int(task_id[0][0].item())}_trial_{int(trial_id[0][0].item())}",
                                }
                            ],
                            "task_descriptions": [
                                self._select_ood_task_description(
                                    all_ood_task_descriptions,
                                    sample_idx=0,
                                    n_samples=n_samples,
                                    task_id=int(task_id[0][0].item()),
                                )
                            ],
                            "video_records": [
                                {
                                    "wm_images": [],
                                    "env_images": [],
                                    "env_dones": [],
                                    "pred_scores": [],
                                }
                            ],
                            "batch_size": batch_size,
                            "max_steps": max_steps,
                        }
                    except Exception:
                        # 最后兜底：构建非常保守的 minimal placeholder
                        last_attempt_state = {
                            "vla_history": [],
                            "task_records": [
                                {
                                    "active": False,
                                    "complete": False,
                                    "finish_step": 0,
                                    "task_file_name": f"task_{0}",
                                }
                            ],
                            "task_descriptions": [""],
                            "video_records": [
                                {
                                    "wm_images": [],
                                    "env_images": [],
                                    "env_dones": [],
                                    "pred_scores": [],
                                }
                            ],
                            "batch_size": batch_size,
                            "max_steps": max_steps,
                        }
                # 强制将已有 vla_history 中的步标记为 dummy 并追加 reason
                try:
                    for s in last_attempt_state.get("vla_history", []):
                        try:
                            # set is_dummy to all-True tensor of same shape
                            if "is_dummy" in s and isinstance(
                                s["is_dummy"], torch.Tensor
                            ):
                                s["is_dummy"] = torch.ones_like(
                                    s["is_dummy"],
                                    dtype=torch.bool,
                                    device=s["is_dummy"].device,
                                )
                            else:
                                s["is_dummy"] = torch.ones(
                                    batch_size, dtype=torch.bool, device=self.device
                                )
                            prev = s.get("placeholder_reason", "")
                            s["placeholder_reason"] = (
                                prev + "," if prev else ""
                            ) + "exception_fallback"
                        except Exception:
                            # best-effort; ignore per-step failures
                            pass
                except Exception:
                    pass
                # === end PATCH EXCEPT ===
                # if reached fallback threshold, break
                if attempt >= fallback_after and not global_success:
                    print(
                        f"[wm_rollout] exception + fallback threshold reached; breaking attempts.",
                        flush=True,
                    )
                    break
                time.sleep(0.2)
                continue
        # === attempts finished or early-broken ===
        chosen_state = (
            best_success_state if best_success_state is not None else last_attempt_state
        )
        if chosen_state is None:
            # fallback minimal placeholder
            dummy_vla_history = []
            dummy_task_records = [
                {
                    "active": False,
                    "complete": False,
                    "finish_step": 0,
                    "task_file_name": f"{task_suite_name[0]}_task_{int(task_id[0][0].item())}_trial_{int(trial_id[0][0].item())}",
                }
            ]
            dummy_task_descriptions = [
                self._select_ood_task_description(
                    all_ood_task_descriptions,
                    sample_idx=0,
                    n_samples=n_samples,
                    task_id=int(task_id[0][0].item()),
                )
            ]
            dummy_video_records = [
                {"wm_images": [], "env_images": [], "env_dones": [], "pred_scores": []}
            ]
            chosen_state = {
                "vla_history": dummy_vla_history,
                "task_records": dummy_task_records,
                "task_descriptions": dummy_task_descriptions,
                "video_records": dummy_video_records,
                "batch_size": batch_size,
                "max_steps": max_steps,
            }
        # mark fallback_used if final chosen is still not complete
        if not chosen_state["task_records"][0].get("complete", False):
            for tr in chosen_state["task_records"]:
                tr["fallback_used"] = True
        # === NEW: compute dummy_ratio for chosen_state and write into task_records[0] for monitoring ===
        try:
            vh = chosen_state.get("vla_history", []) or []
            total_steps = max(1, len(vh))
            dummy_steps = 0
            for s in vh:
                try:
                    is_dummy_tensor = s.get("is_dummy", None)
                    if isinstance(is_dummy_tensor, torch.Tensor):
                        if is_dummy_tensor.numel() == 0:
                            # treat empty as dummy
                            dummy_steps += 1
                        else:
                            if bool(is_dummy_tensor.any().item()):
                                dummy_steps += 1
                    elif isinstance(is_dummy_tensor, (list, tuple)):
                        if any(is_dummy_tensor):
                            dummy_steps += 1
                    else:
                        # if it's a truthy Python value
                        if bool(is_dummy_tensor):
                            dummy_steps += 1
                except Exception:
                    # conservative: count as dummy
                    dummy_steps += 1
            dummy_ratio = float(dummy_steps) / float(total_steps)
            # attach to task_record for monitoring
            try:
                chosen_state["task_records"][0]["dummy_ratio"] = dummy_ratio
            except Exception:
                pass
        except Exception:
            pass
        # === end dummy_ratio ===
        # Save validation videos using CPU copies (non-destructive)
        if is_valid:
            self._save_debug_rollout_videos(
                chosen_state["video_records"],
                chosen_state["task_records"],
                global_steps,
                save_env_tag="env",
                save_wm_tag="wm",
            )
        self.module.train()
        # IMPORTANT: pass vla_history with tensors on original devices (downstream expects that)
        return self._prepare_output_batch_evolving(
            prompts,
            chosen_state["vla_history"],
            chosen_state["task_records"],
            chosen_state["task_descriptions"],
            chosen_state["video_records"],
            chosen_state["batch_size"],
            chosen_state["max_steps"],
        )

    # 1. prepare_output_batch: pure env output
    def _attach_completion_fields(
        self,
        batch: Dict[str, torch.Tensor],
        task_records: List[Dict[str, Any]],
        batch_size: int,
        *,
        env_complete: Optional[torch.Tensor] = None,
        wm_proxy_score: Optional[torch.Tensor] = None,
        pure_wm_proxy: bool = False,
    ) -> None:
        device = batch["responses"].device
        complete = torch.tensor(
            [bool(k["complete"]) for k in task_records],
            dtype=torch.bool,
            device=device,
        )
        batch["complete"] = complete

        if env_complete is None:
            env_complete = torch.zeros_like(complete) if pure_wm_proxy else complete.clone()
        else:
            env_complete = env_complete.to(device=device, dtype=torch.bool).view(batch_size)
        batch["env_complete"] = env_complete

        if wm_proxy_score is None:
            proxy_score = complete.to(dtype=torch.float32) if pure_wm_proxy else torch.zeros(
                (batch_size,), dtype=torch.float32, device=device
            )
        else:
            proxy_score = wm_proxy_score.to(device=device, dtype=torch.float32).view(batch_size)
        threshold = float(self.rm_threshold if self.rm_threshold is not None else 0.5)
        batch["wm_proxy_score"] = proxy_score
        batch["wm_proxy_complete"] = proxy_score >= threshold

    # 1. prepare_output_batch: pure env output
    def _prepare_output_batch_old(
        self, vla_history, task_records, batch_size
    ) -> DataProto:
        """Prepare the output batch from VLA history"""
        batch = {
            "responses": [],
            "input_ids": [],
            "attention_mask": [],
            "pixel_values": [],
        }

        key_names = ["responses", "input_ids", "attention_mask", "pixel_values"]
        if self.config.use_proprio and "robotwin" in self.config.task_suite_name:
            batch["proprio"] = []
            key_names.append("proprio")

        for k in key_names:
            for h in vla_history:
                batch[k].append(h[k])

        for k, v in batch.items():
            batch[k] = torch.stack(v, dim=1)

        self._attach_completion_fields(batch, task_records, batch_size)
        batch["finish_step"] = torch.tensor(
            [k["finish_step"] for k in task_records],
            dtype=torch.int64,
            device=batch["responses"].device,
        )

        output_batch = TensorDict(batch, batch_size=batch_size)
        return DataProto(batch=output_batch)

    # 1. prepare_output_batch: pure env output
    def _prepare_output_batch(self, vla_history, task_records, batch_size) -> DataProto:
        """Prepare the output batch from VLA history"""
        batch = {
            "responses": [],
            "input_ids": [],
            "attention_mask": [],
            "pixel_values": [],
            "is_dummy": [],  # === FIX: 添加 is_dummy ===
        }
        key_names = [
            "responses",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "is_dummy",
        ]
        if self.config.use_proprio and "robotwin" in self.config.task_suite_name:
            batch["proprio"] = []
            key_names.append("proprio")
        for k in key_names:
            for h in vla_history:
                # === FIX: is_dummy 可能不存在，提供默认值 ===
                if k == "is_dummy":
                    batch[k].append(
                        h.get("is_dummy", torch.zeros(batch_size, dtype=torch.bool))
                    )
                else:
                    batch[k].append(h[k])
        for k, v in batch.items():
            batch[k] = torch.stack(v, dim=1)

        # === FIX: 无条件添加 action 字段 ===
        batch_actions = []
        for h in vla_history:
            action = h.get("action", None)
            if action is not None:
                if isinstance(action, torch.Tensor):
                    batch_actions.append(action.detach().cpu().numpy())
                else:
                    batch_actions.append(action)
            else:
                # 创建 placeholder action
                batch_actions.append(
                    np.zeros(
                        (batch_size, self.config.action_chunks_len, ACTION_DIM),
                        dtype=np.float32,
                    )
                )
        batch["action"] = torch.tensor(np.array(batch_actions), dtype=torch.float32)
        # [steps, bsz, action_chunk_size, action_dim]
        batch["action"] = (
            batch["action"]
            .permute(1, 0, 2, 3)
            .reshape(batch_size, -1, batch["action"].shape[-1])
        )  # tensor, [bsz, steps * action_chunk_size, action_dim]

        self._attach_completion_fields(batch, task_records, batch_size)
        batch["finish_step"] = torch.tensor(
            [k["finish_step"] for k in task_records],
            dtype=torch.int64,
            device=batch["responses"].device,
        )

        output_batch = TensorDict(batch, batch_size=batch_size)
        return DataProto(batch=output_batch)

    #! 2. prepare_output_batch: pure world model output
    def _prepare_output_batch_wm(
        self,
        prompts: DataProto,
        vla_history: List[Dict[str, Any]],
        task_records: List[Dict[str, Any]],
        video_records: List[List[np.ndarray]],
        batch_size: int,
        max_steps: int,
    ) -> DataProto:
        return_rollouts = prompts.meta_info.get("return_rollouts", False)

        """Prepare the output batch from VLA history"""
        batch = {
            "responses": [],
            "input_ids": [],
            "attention_mask": [],
            "pixel_values": [],
        }

        key_names = ["responses", "input_ids", "attention_mask", "pixel_values"]
        if self.config.use_proprio and "robotwin" in self.config.task_suite_name:
            batch["proprio"] = []
            key_names.append("proprio")
        for k in key_names:
            for h in vla_history:
                batch[k].append(h[k])
        for k, v in batch.items():
            batch[k] = torch.stack(v, dim=1)

        self._attach_completion_fields(
            batch,
            task_records,
            batch_size,
            pure_wm_proxy=True,
        )
        batch["finish_step"] = torch.tensor(
            [k["finish_step"] for k in task_records],
            dtype=torch.int64,
            device=batch["responses"].device,
        )

        def _copy_prompt_tensor_key(key: str):
            if not hasattr(prompts, "batch") or key not in prompts.batch:
                return
            value = prompts.batch[key]
            if not isinstance(value, torch.Tensor) or value.shape[0] <= 0:
                return
            if value.shape[0] == batch_size:
                copied = value
            elif batch_size % int(value.shape[0]) == 0:
                repeat_factor = batch_size // int(value.shape[0])
                copied = value.repeat_interleave(repeat_factor, dim=0)
            else:
                copied = value[:1].repeat_interleave(batch_size, dim=0)
            batch[key] = copied.detach().to(device=batch["responses"].device)

        for prompt_key in ("task_id", "trial_id", "trial_seed"):
            _copy_prompt_tensor_key(prompt_key)

        valid_response_tokens = count_valid_response_tokens(
            batch["responses"], batch["finish_step"], self.config.action_chunks_len
        )
        batch["valid_response_tokens"] = valid_response_tokens.to(torch.long)
        batch["wm_valid_response_tokens"] = valid_response_tokens.to(torch.long)

        #! Add action and image sequences to the output for training WM
        if return_rollouts:
            batch_actions = (
                []
            )  # List[np.array], each np.array: [bsz, action_chunk_size, action_dim]
            for h in vla_history:
                batch_actions.append(h["action"])
            batch["action"] = torch.tensor(np.array(batch_actions), dtype=torch.float32)
            # [steps, bsz, action_chunk_size, action_dim]
            batch["action"] = (
                batch["action"]
                .permute(1, 0, 2, 3)
                .reshape(batch_size, -1, batch["action"].shape[-1])
            )  # tensor, [bsz, steps * action_chunk_size, action_dim]

            # Get padded videos
            B = len(video_records)
            H, W, C = video_records[0][0].shape
            max_len = max_steps + 1
            padded_videos = torch.zeros((B, max_len, H, W, C), dtype=torch.uint8)
            for i, v in enumerate(video_records):
                v_tensor = torch.from_numpy(np.asarray(v, dtype=np.uint8))
                T = v_tensor.shape[0]
                padded_videos[i, :T] = v_tensor
            batch["video"] = padded_videos  # tensor, [bsz, steps, H, W, C]

        output_batch = TensorDict(batch, batch_size=batch_size)
        output_non_tensor_batch = {}
        if hasattr(prompts, "non_tensor_batch"):
            for key in ("uid", "task_suite_name"):
                value = prompts.non_tensor_batch.get(key, None)
                if value is None:
                    continue
                value = np.asarray(value, dtype=object)
                if len(value) == batch_size:
                    output_non_tensor_batch[key] = value.copy()
                elif len(value) > 0 and batch_size % len(value) == 0:
                    output_non_tensor_batch[key] = np.repeat(
                        value, batch_size // len(value)
                    )
                elif len(value) > 0:
                    output_non_tensor_batch[key] = np.repeat(value[:1], batch_size)
        return DataProto(
            batch=output_batch,
            non_tensor_batch=output_non_tensor_batch,
        )

    #! 3. prepare_output_batch: evolving output
    def _prepare_output_batch_evolving(
        self,
        prompts: DataProto,
        vla_history: List[Dict[str, Any]],
        task_records: List[Dict[str, Any]],
        task_descriptions: List[str],
        video_records: List[Dict[str, List[np.ndarray]]],
        batch_size: int,
        max_steps: int,
    ) -> DataProto:
        return_rollouts = prompts.meta_info.get("return_rollouts", False)
        is_wm_rollout = bool(prompts.meta_info.get("use_wm", False))
        """Prepare the output batch from VLA history"""
        batch = {
            "responses": [],
            "input_ids": [],
            "attention_mask": [],
            "pixel_values": [],
            "is_dummy": [],
        }
        key_names = [
            "responses",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "is_dummy",
        ]
        if self.config.use_proprio and "robotwin" in self.config.task_suite_name:
            batch["proprio"] = []
            key_names.append("proprio")
        for k in key_names:
            for h in vla_history:
                batch[k].append(h[k])
        for k, v in batch.items():
            batch[k] = torch.stack(v, dim=1)

        # === FIX: 确保 action 字段始终存在（无条件添加）===
        batch_actions = []
        for h in vla_history:
            action = h.get("action", None)
            if action is not None:
                if isinstance(action, torch.Tensor):
                    batch_actions.append(action.detach().cpu().numpy())
                else:
                    batch_actions.append(action)
            else:
                # 创建 placeholder action
                batch_actions.append(
                    np.zeros(
                        (batch_size, self.config.action_chunks_len, ACTION_DIM),
                        dtype=np.float32,
                    )
                )
        batch["action"] = torch.tensor(np.array(batch_actions), dtype=torch.float32)
        # [steps, bsz, action_chunk_size, action_dim]
        batch["action"] = (
            batch["action"]
            .permute(1, 0, 2, 3)
            .reshape(batch_size, -1, batch["action"].shape[-1])
        )  # tensor, [bsz, steps * action_chunk_size, action_dim]

        self._attach_completion_fields(batch, task_records, batch_size)
        batch["finish_step"] = torch.tensor(
            [k["finish_step"] for k in task_records],
            dtype=torch.int64,
            device=batch["responses"].device,
        )

        def _copy_prompt_tensor_key(key: str):
            if not hasattr(prompts, "batch") or key not in prompts.batch:
                return
            value = prompts.batch[key]
            if not isinstance(value, torch.Tensor) or value.shape[0] <= 0:
                return
            if value.shape[0] == batch_size:
                copied = value
            elif batch_size % int(value.shape[0]) == 0:
                repeat_factor = batch_size // int(value.shape[0])
                copied = value.repeat_interleave(repeat_factor, dim=0)
            else:
                copied = value[:1].repeat_interleave(batch_size, dim=0)
            batch[key] = copied.detach().to(device=batch["responses"].device)

        for prompt_key in ("task_id", "trial_id", "trial_seed"):
            _copy_prompt_tensor_key(prompt_key)

        is_dummy = batch.get("is_dummy", None)
        if is_dummy is not None:
            dummy_by_step = is_dummy.to(dtype=torch.bool).view(batch_size, -1)
            dummy_step_count = dummy_by_step.to(torch.long).sum(dim=1)
        else:
            dummy_step_count = torch.zeros(
                batch_size, dtype=torch.long, device=batch["responses"].device
            )
        valid_response_tokens = count_valid_response_tokens(
            batch["responses"], batch["finish_step"], self.config.action_chunks_len,
            dummy_chunks=is_dummy,
        )
        batch["wm_dummy_step_count"] = dummy_step_count
        batch["wm_placeholder_step_count"] = dummy_step_count.clone()
        batch["valid_response_tokens"] = valid_response_tokens.to(torch.long)
        batch["wm_valid_response_tokens"] = valid_response_tokens.to(torch.long)

        def _sampled_obs_mse(video_record: Dict[str, List[np.ndarray]]) -> float:
            wm_images = list(video_record.get("wm_images", []))
            env_images = list(video_record.get("env_images", []))
            limit = min(len(wm_images), len(env_images), max_steps)
            if limit <= 0:
                return 1.0
            sample_count = min(limit, 8)
            frame_indices = np.linspace(0, limit - 1, sample_count, dtype=np.int64)
            err_sum = 0.0
            used = 0
            for frame_idx in frame_indices:
                wm_frame = np.asarray(wm_images[int(frame_idx)])
                env_frame = np.asarray(env_images[int(frame_idx)])
                if wm_frame.ndim != 3 or env_frame.ndim != 3:
                    continue
                h = min(wm_frame.shape[0], env_frame.shape[0])
                w = min(wm_frame.shape[1], env_frame.shape[1])
                c = min(wm_frame.shape[2], env_frame.shape[2])
                if h <= 0 or w <= 0 or c <= 0:
                    continue
                diff = (
                    wm_frame[:h, :w, :c].astype(np.float32)
                    - env_frame[:h, :w, :c].astype(np.float32)
                ) / 255.0
                err_sum += float(np.mean(diff * diff))
                used += 1
            return float(err_sum / max(used, 1)) if used > 0 else 1.0

        def _done_abs_error(
            task_record: Dict[str, Any], video_record: Dict[str, List[np.ndarray]]
        ) -> float:
            finish_step = int(task_record.get("finish_step", 0))
            dones = np.asarray(video_record.get("env_dones", []), dtype=np.float32)
            dones = dones.reshape(-1) if dones.size > 0 else dones
            if dones.size > 0 and bool(np.any(dones > 0.5)):
                actual_finish = int(np.argmax(dones > 0.5)) + 1
            else:
                actual_finish = max(
                    len(video_record.get("env_images", [])),
                    len(video_record.get("wm_images", [])),
                    finish_step,
                )
            denom = max(int(max_steps), 1)
            return float(min(abs(actual_finish - finish_step) / denom, 1.0))

        response_shape = tuple(batch["responses"].shape)
        response_flat_length = int(batch["responses"][0].numel())
        action_token_len = batch["responses"].shape[-1] // self.config.action_chunks_len
        rm_scores = torch.zeros_like(batch["responses"], dtype=torch.float32)
        rm_scores_flat = rm_scores.view(batch_size, -1)
        wm_proxy_scores = torch.zeros(
            (batch_size,), dtype=torch.float32, device=batch["responses"].device
        )
        for idx, (task_record, video_record) in enumerate(
            zip(task_records, video_records)
        ):
            pred_scores = np.asarray(
                video_record.get("pred_scores", []), dtype=np.float32
            ).reshape(-1)
            if pred_scores.size == 0:
                continue
            valid_steps = int(batch["valid_response_tokens"][idx].item()) // action_token_len
            valid_steps = min(valid_steps, int(pred_scores.size))
            if valid_steps <= 0:
                continue
            # Keep WM reward on the same scale as verifier outcome reward by
            # mapping one sequence-level score onto the terminal action token.
            seq_rm_score = float(np.max(pred_scores[:valid_steps]))
            wm_proxy_scores[idx] = seq_rm_score
            token_idx = min(
                valid_steps * action_token_len - 1, response_flat_length - 1
            )
            if token_idx >= 0:
                rm_scores_flat[idx, token_idx] = seq_rm_score
        rm_scores = rm_scores_flat.view(response_shape)
        batch["rm_scores"] = rm_scores
        threshold = float(self.rm_threshold if self.rm_threshold is not None else 0.5)
        batch["wm_proxy_score"] = wm_proxy_scores
        batch["wm_proxy_complete"] = wm_proxy_scores >= threshold
        batch["wm_obs_error"] = torch.tensor(
            [_sampled_obs_mse(v_record) for v_record in video_records],
            dtype=torch.float32,
            device=batch["responses"].device,
        )
        batch["wm_done_error"] = torch.tensor(
            [
                _done_abs_error(task_record, v_record)
                for task_record, v_record in zip(task_records, video_records)
            ],
            dtype=torch.float32,
            device=batch["responses"].device,
        )
        batch["wm_pred_valid"] = torch.tensor(
            [bool(v_record.get("wm_images", [])) for v_record in video_records],
            dtype=torch.bool,
            device=batch["responses"].device,
        )

        # return_rollouts 逻辑保留（仅用于 video）
        if return_rollouts:
            # Get padded wm videos
            B = len(video_records)
            # === FIX: 添加空列表防御检查 ===
            try:
                first_wm_frame = None
                first_env_frame = None
                for v_record in video_records:
                    wm_images = v_record.get("wm_images", [])
                    env_images = v_record.get("env_images", [])
                    if first_wm_frame is None and wm_images:
                        first_wm_frame = wm_images[0]
                    if first_env_frame is None and env_images:
                        first_env_frame = env_images[0]
                    if first_wm_frame is not None and first_env_frame is not None:
                        break

                if first_wm_frame is not None:
                    H, W, C = first_wm_frame.shape
                elif first_env_frame is not None:
                    H, W, C = first_env_frame.shape
                    if is_wm_rollout:
                        print(
                            "[wm_rollout] WARNING: wm_images empty for current batch; "
                            "using env_images only for output shape and keeping WM video zero-padded.",
                            flush=True,
                        )
                else:
                    H, W, C = 256, 256, 3
            except Exception as e:
                print(
                    f"[wm_rollout] WARNING: Failed to get wm_images shape: {e}, using default",
                    flush=True,
                )
                H, W, C = 256, 256, 3
            max_len = max_steps
            padded_wm_videos = torch.zeros((B, max_len, H, W, C), dtype=torch.uint8)
            for i, v_record in enumerate(video_records):
                v_list = v_record.get("wm_images", [])
                if v_list and len(v_list) > 0:
                    v_tensor = torch.from_numpy(np.asarray(v_list, dtype=np.uint8))
                    T = min(v_tensor.shape[0], max_len)
                    padded_wm_videos[i, :T] = v_tensor[:max_len]
            batch["video"] = padded_wm_videos

            # Get padded env videos
            try:
                if (
                    video_records
                    and len(video_records) > 0
                    and video_records[0].get("env_images")
                    and len(video_records[0]["env_images"]) > 0
                ):
                    H, W, C = video_records[0]["env_images"][0].shape
                else:
                    H, W, C = 256, 256, 3
            except Exception:
                H, W, C = 256, 256, 3
            padded_env_videos = torch.zeros((B, max_len, H, W, C), dtype=torch.uint8)
            for i, v_record in enumerate(video_records):
                v_list = v_record.get("env_images", [])
                if v_list and len(v_list) > 0:
                    v_tensor = torch.from_numpy(np.asarray(v_list, dtype=np.uint8))
                    T = min(v_tensor.shape[0], max_len)
                    padded_env_videos[i, :T] = v_tensor[:max_len]
            batch["env_video"] = padded_env_videos

            # Get padded env dones
            batch_env_dones = []
            for i, v_record in enumerate(video_records):
                dones_list = v_record.get("env_dones", [])
                if len(dones_list) == 0:
                    dones = torch.zeros((0,), dtype=torch.int64)
                else:
                    dones = torch.tensor(dones_list, dtype=torch.int64)
                cur_len = dones.shape[0]
                if cur_len < max_len:
                    pad_len = max_len - cur_len
                    pad = torch.zeros((pad_len,), dtype=torch.int64)
                    dones = torch.cat([dones, pad], dim=0)
                elif cur_len > max_len:
                    dones = dones[:max_len]
                batch_env_dones.append(dones)
            batch["env_dones"] = torch.stack(batch_env_dones, dim=0)

        output_batch = TensorDict(batch, batch_size=batch_size)
        output_non_tensor_batch = {}
        prompt_uids = None
        if hasattr(prompts, "non_tensor_batch"):
            prompt_uids = prompts.non_tensor_batch.get("uid", None)
        if prompt_uids is not None:
            prompt_uids = np.asarray(prompt_uids, dtype=object)
            if len(prompt_uids) == batch_size:
                output_non_tensor_batch["uid"] = prompt_uids.copy()
            elif len(prompt_uids) > 0 and batch_size % len(prompt_uids) == 0:
                repeat_factor = batch_size // len(prompt_uids)
                output_non_tensor_batch["uid"] = np.repeat(prompt_uids, repeat_factor)
        if hasattr(prompts, "non_tensor_batch"):
            prompt_task_suite = prompts.non_tensor_batch.get("task_suite_name", None)
            if prompt_task_suite is not None:
                prompt_task_suite = np.asarray(prompt_task_suite, dtype=object)
                if len(prompt_task_suite) == batch_size:
                    output_non_tensor_batch["task_suite_name"] = (
                        prompt_task_suite.copy()
                    )
                elif (
                    len(prompt_task_suite) > 0
                    and batch_size % len(prompt_task_suite) == 0
                ):
                    repeat_factor = batch_size // len(prompt_task_suite)
                    output_non_tensor_batch["task_suite_name"] = np.repeat(
                        prompt_task_suite, repeat_factor
                    )
                elif len(prompt_task_suite) > 0:
                    output_non_tensor_batch["task_suite_name"] = np.repeat(
                        prompt_task_suite[:1], batch_size
                    )

        placeholder_reasons = [[] for _ in range(batch_size)]
        for h in vla_history:
            reason = str(h.get("placeholder_reason", "") or "").strip()
            if not reason:
                continue
            dummy_flags = h.get("is_dummy", None)
            if isinstance(dummy_flags, torch.Tensor):
                flags = dummy_flags.detach().cpu().view(-1).bool().tolist()
            else:
                flags = [bool(dummy_flags)] * batch_size
            if len(flags) == 1 and batch_size > 1:
                flags = flags * batch_size
            for idx, flag in enumerate(flags[:batch_size]):
                if flag:
                    placeholder_reasons[idx].append(reason)
        output_non_tensor_batch["placeholder_reason"] = np.array(
            [
                ";".join(sorted(set(reasons))) if len(reasons) > 0 else ""
                for reasons in placeholder_reasons
            ],
            dtype=object,
        )
        task_descriptions = normalize_task_descriptions(
            task_descriptions,
            batch_size,
            context="rob_rollout_wm_pro.output_task_descriptions",
        )
        output_non_tensor_batch["task_descriptions"] = np.array(
            task_descriptions, dtype=object
        )
        return DataProto(
            batch=output_batch,
            non_tensor_batch=output_non_tensor_batch,
            meta_info={},
        )

    @torch.no_grad()
    def _generate_one_step(self, prompts: dict):
        """Generate one step of actions"""
        if self.config.vla == "openvla-oft":
            return self._generate_one_step_oft(prompts)
        elif self.config.vla == "openvla":
            return self._generate_one_step_openvla(prompts)
        else:
            raise ValueError(f"Unknown VLA type: {self.config.vla}")

    @torch.no_grad()
    def _generate_one_step_wm(self, prompts: dict):
        """Generate one step of actions"""
        if self.config.vla == "openvla-oft":
            return self._generate_one_step_oft_wm(prompts)
        elif self.config.vla == "openvla":
            return self._generate_one_step_openvla(prompts)
        else:
            raise ValueError(f"Unknown VLA type: {self.config.vla}")

    def _generate_one_step_oft(self, prompts: dict):
        """Generate one step for OpenVLA-OFT"""
        idx = prompts["input_ids"]
        attention_mask = prompts["attention_mask"]
        pixel_values = prompts["pixel_values"]
        proprio = prompts.get("proprio", None)

        param_ctx = contextlib.nullcontext()
        do_sample = prompts.get("do_sample", self.config.do_sample)
        temperature = prompts.get("temperature", self.config.temperature)

        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(
                self.module, writeback=False, recurse=False
            )

        with param_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                actions, response = self.module.generate_action_verl(
                    input_ids=idx,
                    pixel_values=pixel_values,
                    proprio=proprio,
                    attention_mask=attention_mask,
                    padding_idx=self.processor.tokenizer.pad_token_id,
                    do_sample=do_sample,
                    unnorm_key=self.config.unnorm_key,
                    temperature=temperature,
                )

        assert self.processor.tokenizer.pad_token_id is not None

        idx = verl_F.pad_sequence_to_length(
            idx,
            max_seq_len=self.config.max_prompt_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            left_pad=True,
        )

        attention_mask = verl_F.pad_sequence_to_length(
            attention_mask,
            max_seq_len=self.config.max_prompt_length,
            pad_token_id=0,
            left_pad=True,
        )

        batch = {
            "responses": response,
            "input_ids": idx,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "action": actions,
        }
        if proprio is not None:
            batch["proprio"] = proprio

        return batch

    def _generate_one_step_oft_wm(self, prompts: dict):
        idx = prompts["input_ids"]
        attention_mask = prompts["attention_mask"]
        pixel_values = prompts["pixel_values"]
        proprio = prompts.get("proprio", None)

        do_sample = prompts.get("do_sample", self.config.do_sample)
        temperature = prompts.get("temperature", self.config.temperature)

        # Ensure params are available if module is FSDP (same pattern as non-WM path)
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(
                self.module, writeback=False, recurse=False
            )

        # Ensure inputs are on same device as model params (defensive)
        try:
            model_device = next(self.module.parameters()).device
            if idx.device != model_device:
                idx = idx.to(device=model_device)
            if attention_mask.device != model_device:
                attention_mask = attention_mask.to(device=model_device)
            if (
                isinstance(pixel_values, torch.Tensor)
                and pixel_values.device != model_device
            ):
                pixel_values = pixel_values.to(device=model_device)
            if (
                proprio is not None
                and isinstance(proprio, torch.Tensor)
                and proprio.device != model_device
            ):
                proprio = proprio.to(device=model_device)
        except StopIteration:
            # model has no params? fallback — but this would be unusual
            pass

        with param_ctx:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    actions, response = self.module.generate_action_verl(
                        input_ids=idx,
                        pixel_values=pixel_values,
                        proprio=proprio,
                        attention_mask=attention_mask,
                        padding_idx=self.processor.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        unnorm_key=self.config.unnorm_key,
                        temperature=temperature,
                    )

        # pad outputs like original code expects
        idx = verl_F.pad_sequence_to_length(
            idx,
            max_seq_len=self.config.max_prompt_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            left_pad=True,
        )
        attention_mask = verl_F.pad_sequence_to_length(
            attention_mask,
            max_seq_len=self.config.max_prompt_length,
            pad_token_id=0,
            left_pad=True,
        )

        batch = {
            "responses": response,
            "input_ids": idx,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "action": actions,
        }
        if proprio is not None:
            batch["proprio"] = proprio

        return batch

    def _generate_one_step_openvla(self, prompts: dict):
        """Generate one step for OpenVLA"""
        idx = prompts["input_ids"]
        attention_mask = prompts["attention_mask"]
        pixel_values = prompts["pixel_values"]

        eos_token_id = prompts["eos_token_id"]
        pad_token_id = prompts["pad_token_id"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)
        param_ctx = contextlib.nullcontext()

        do_sample = prompts.get("do_sample", self.config.do_sample)
        response_length = self.module.get_action_dim(self.config.unnorm_key)
        top_p = prompts.get("top_p", self.config.get("top_p", 1.0))
        top_k = prompts.get("top_k", self.config.get("top_k", 0))
        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)

        temperature = prompts.get("temperature", self.config.temperature)
        generation_config = GenerationConfig(
            temperature=temperature, top_p=top_p, top_k=top_k
        )

        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(
                self.module, writeback=False, recurse=False
            )

        with param_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=idx,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                )

        seq = output.sequences
        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:]

        response_attention_mask = get_eos_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Extract and unnormalize actions
        predicted_action_token_ids = response.detach().cpu().numpy()
        discretized_actions = self.module.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.module.bin_centers.shape[0] - 1
        )
        normalized_actions = self.module.bin_centers[discretized_actions]

        action_norm_stats = self.module.get_action_stats(self.config.unnorm_key)
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        actions = np.expand_dims(actions, axis=1)

        prompt = verl_F.pad_sequence_to_length(
            prompt,
            max_seq_len=self.config.max_prompt_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            left_pad=True,
        )
        seq = verl_F.pad_sequence_to_length(
            seq,
            max_seq_len=self.config.max_prompt_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            left_pad=True,
        )
        attention_mask = verl_F.pad_sequence_to_length(
            attention_mask,
            max_seq_len=self.config.max_prompt_length,
            pad_token_id=0,
            left_pad=True,
        )

        batch = {
            "prompts": prompt,
            "responses": response,
            "input_ids": seq,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "action": actions,
        }

        return batch

    def _obs_to_input(self, obs, is_robotwin=False, robotwin_version="1.0"):
        """Convert observation to model input format"""
        if not is_robotwin:
            # Libero
            state = np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            )

            if self.config.num_images_in_input > 1:
                return {
                    "full_image": get_libero_image(obs, 224),
                    "wrist_image": get_libero_wrist_image(obs, 224),
                    "state": state,
                }
            else:
                return {"full_image": get_libero_image(obs, 224), "state": state}
        else:
            # Robotwin
            if robotwin_version == "1.0":
                state = obs["joint_action"]
                state[6] /= 0.045
                state[13] /= 0.045
            else:  # 2.0
                state = obs["joint_action"]["vector"]

            if self.config.num_images_in_input == 3:
                return {
                    "full_image": obs["observation"]["head_camera"]["rgb"],
                    "left_wrist": obs["observation"]["left_camera"]["rgb"],
                    "right_wrist": obs["observation"]["right_camera"]["rgb"],
                    "state": state,
                }
            else:
                return {
                    "full_image": obs["observation"]["head_camera"]["rgb"],
                    "state": state,
                }

    def _img_to_input(
        self, img: np.ndarray, is_robotwin: bool = False, robotwin_version: str = "1.0"
    ) -> Dict[str, Any]:
        """Convert current observation image to next model input format"""
        assert is_robotwin == False, "Robotwin is not supported yet."
        if not is_robotwin:
            # Libero
            assert (
                self.config.num_images_in_input == 1
            ), "Only the main view image is supported for Libero."
            return {"full_image": resize_to_libero_image(img, 256), "state": None}

    def _result_to_input(
        self,
        result: Dict[str, Any],
        is_robotwin: bool = False,
    ) -> Dict[str, Any]:
        """Convert current result to next model input format"""
        assert is_robotwin == False, "Robotwin is not supported yet."
        if not is_robotwin:
            # Libero
            assert (
                self.config.num_images_in_input == 1
            ), "Only the main view image is supported for Libero."
            full_image: np.ndarray = resize_to_libero_image(result["image"], (256, 256))
            hist_images: List[np.ndarray] = []
            valid_images = result.get("valid_images") or []
            for t in range(len(valid_images)):
                hist_image_t = resize_to_libero_image(valid_images[t], (256, 256))
                hist_images.append(hist_image_t)
            hist_action: List[np.ndarray] = result.get("normed_actions") or []
            return {
                "full_image": full_image,
                "hist_images": hist_images,
                "hist_action": hist_action,
                "state": None,
            }

    def __del__(self):
        """Cleanup resources on deletion"""
        with contextlib.suppress(Exception):
            self._close_libero_env_services()
        if hasattr(self, "env_thread_pool"):
            self.env_thread_pool.shutdown(wait=False)

    def _cleanup_environments(self):
        """Clean up all environmental resources"""
        with contextlib.suppress(Exception):
            self._close_libero_env_services()
        if hasattr(self, "envs") and self.envs:
            for env in self.envs:
                try:
                    if hasattr(env, "close"):
                        env.close()
                except Exception as e:
                    print(f"Error closing environment: {e}")
            self.envs = []

        try:
            import mujoco

            if hasattr(mujoco, "renderer") and hasattr(mujoco.renderer, "_contexts"):
                contexts = mujoco.renderer._contexts
                for ctx in list(contexts.values()):
                    try:
                        if hasattr(ctx, "free"):
                            ctx.free()
                    except:
                        pass
                contexts.clear()
                print(f"Cleaned {len(contexts)} MuJoCo contexts")
        except ImportError:
            pass

    def compute_wm_predictions_on_main(self, batches_inputs: List[Dict[str, Any]]):
        """
        batches_inputs: list of per-sample dict as produced by _prepare_world_model_env_input,
        each dict contains:
            - 'task_description': str
            - 'image': torch.Tensor [C, h, w]   (float in -1..1, or uint8 scaled earlier)
            - 'hist_images': torch.Tensor [t_h, C, h, w]
            - 'action': torch.Tensor [action_chunk_size, action_dim]  (numpy->tensor already)
        Returns:
        - wm_outs: list, length B, each item a dict {
            'pred_images_list': list[np.ndarray] (len = action_chunk_size, shape [h,w,3]),
            'pred_images_latent': torch.Tensor [t, c, h_lat, w_lat],
            'pred_scores': np.ndarray [t]   # cpu numpy
            }
        NOTE: This implementation calls WM per-sample in a Python loop for simplicity and to avoid
        batch-dimension mismatches with the pipeline. If needed for speed, implement batching here.
        """
        valid_batches = []
        valid_indices = []
        for i, bi in enumerate(batches_inputs):
            if bi is None:
                continue
            if bi["hist_images"].numel() == 0:
                continue
            if bi["hist_images"].shape[1] == 0:  # num_frames == 0
                continue
            valid_batches.append(bi)
            valid_indices.append(i)
        if len(valid_batches) == 0:
            return []

        wm = self.world_model  # main-process model (on self.device)
        wm_args = self.wm_args
        device = self.device
        valid_wm_outs = []

        # ensure model on device and eval
        wm.to(device)
        wm.eval()

        with torch.no_grad():
            # optional autocast if bm dtype supports it
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for bi in valid_batches:
                # prepare image tensors like child previously did
                # bi['image'] is [C,h,w] or [1,C,h,w]; adapt to required shape
                image = bi["image"]
                if image.dim() == 3:
                    image = image.unsqueeze(0)  # [1, C, h, w]
                image = image.to(device=device, dtype=self.dtype)

                hist_images = bi.get("hist_images", None)
                if hist_images is None:
                    history_latent = None
                else:
                    # hist_images: [t_h, C, h, w] -> [1, t_h, C, h, w]
                    if hist_images.dim() == 4:
                        hist_images_in = hist_images.unsqueeze(0).to(
                            device=device, dtype=self.dtype
                        )  # [1, t_h, C, h, w]
                    else:
                        hist_images_in = hist_images.to(device=device, dtype=self.dtype)
                # encode to latent (use your helper)
                image_latent = encode_img_to_latent(
                    wm, image, device
                )  # returns CPU latents in your helper; adapt as needed
                image_latent = image_latent.to(device=device, dtype=self.dtype)
                if hist_images is not None:
                    hist_latent = encode_img_to_latent(
                        wm, hist_images_in.squeeze(0), device
                    )
                    hist_latent = hist_latent.unsqueeze(0).to(
                        device=device, dtype=self.dtype
                    )  # [1, t_h, ...]
                else:
                    hist_latent = None

                # prepare action_latent: combine history actions + future actions
                action_tensor = bi["action"]  # [t, action_dim] (torch tensor)
                # normalize/invert as in original code
                normed_action = []
                for t in range(action_tensor.shape[0]):
                    a_t_np = action_tensor[t].cpu().numpy()
                    normalized_a_t_np = normalize_gripper_action(a_t_np, binarize=True)
                    inverted_a_t_np = invert_gripper_action(normalized_a_t_np)
                    normed_action.append(inverted_a_t_np)
                hist_action = bi.get("hist_action", [])
                all_normed_action = hist_action + normed_action
                all_action_latent = get_action_latent(
                    wm,
                    all_normed_action,
                    bi["task_description"],
                    wm_args.frame_level_cond,
                )  # [1, t_h + t, d]
                # print(f"all_normed_action len: {len(all_normed_action)}")
                # print(f"all_normed_action[0].shape: {all_normed_action[0].shape}")

                # call pipeline (per-sample)
                # print(f"image_latent shape: {image_latent.shape}")
                # print(f"hist_latent shape: {hist_latent.shape}")
                # print(f"all_action_latent shape: {all_action_latent.shape}")
                with torch.no_grad():
                    pred_images_list, pred_images_latent = (
                        CtrlWorldDiffusionPipeline.__call__(
                            wm.pipeline,
                            image=image_latent,
                            text=all_action_latent,
                            width=wm_args.width,
                            height=wm_args.height,
                            num_frames=action_tensor.shape[0],
                            history=(hist_latent if hist_latent is not None else None),
                            num_inference_steps=wm_args.num_inference_steps,
                            decode_chunk_size=action_tensor.shape[0],
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

                    # print(f"len pred_images_list: {len(pred_images_list)}")
                    # print(f"pred_images_list[0].shape: {pred_images_list[0].shape}")
                    # print(f"min/max: {pred_images_list[0].min()}/{pred_images_list[0].max()}")

                    # convert pred_images (list of np arrays) to a torch tensor for scoring
                    # pred_images_list is list (bsz) * [t, h, w, 3], but here bsz==1 -> pred_images_list[0]
                    # pred_images_np = pred_images_list[0]  # [t, h, w, 3] numpy
                    pred_images_np = float01_to_uint8(
                        pred_images_list[0]
                    )  # [t, h, w, 3], 0~255, uint8 for viz

                    # convert to torch tensor shape [t, c, h, w]
                    pred_images_tensor = (
                        torch.from_numpy(pred_images_list[0])
                        .permute(0, 3, 1, 2)
                        .to(device=device, dtype=self.dtype)
                    )

                    # build action latent for last t frames
                    action_latent = all_action_latent[
                        :, -action_tensor.shape[0] :
                    ]  # [1, t, d]
                    action_latent_flat = einops.rearrange(
                        action_latent, "b t d -> (b t) d"
                    ).to(device=device, dtype=self.dtype)

                    # predict score with reward_classifier
                    pred_scores_t = get_inner_module(
                        wm
                    ).reward_classifier.predict_score(
                        pred_images_tensor, action_latent_flat
                    )  # expects [bsz * t]
                    # pred_scores_np = pred_scores_t.detach().cpu().numpy()
                    pred_scores_np = (
                        pred_scores_t.detach().to(torch.float32).cpu().numpy()
                    )

                valid_wm_outs.append(
                    {
                        "pred_images_list": pred_images_np,  # numpy array [t, h, w, 3]
                        "pred_images_latent": pred_images_latent,  # tensor on device (bsz,t,c,h,w)
                        "pred_scores": pred_scores_np,  # numpy [t]
                    }
                )

        wm_outputs = [None] * len(batches_inputs)
        for out, idx in zip(valid_wm_outs, valid_indices):
            wm_outputs[idx] = out
        return wm_outputs

    # For Libero-pro
    def _preprocess_libero_pro_task_suite(self, prompts):
        # print(type(prompts.non_tensor_batch["task_suite_name"]))
        # print(prompts.non_tensor_batch["task_suite_name"].shape)
        # print(prompts.non_tensor_batch["task_suite_name"][0])
        # print(type(prompts.non_tensor_batch["task_suite_name"][0]))

        all_ood_task_descriptions = []
        raw_task_suite_names = prompts.non_tensor_batch["task_suite_name"]
        num_task_suite = int(raw_task_suite_names.shape[0])

        def collect_suite_members(dir_path: str, suffix: str) -> set[str]:
            directory = Path(dir_path)
            if not directory.is_dir():
                return set()

            members = set()
            for file_path in directory.iterdir():
                if not file_path.is_file() or not file_path.name.endswith(suffix):
                    continue
                members.add(file_path.name[: -len(suffix)])
            return members

        def collect_expected_suite_members(
            task_suite_name: str,
        ) -> Tuple[set[str], set[str]]:
            benchmark_dict = _get_libero_pro_benchmark_dict(self.config)
            if task_suite_name not in benchmark_dict:
                raise KeyError(
                    f"Unknown LIBERO_PRO task suite '{task_suite_name}'. "
                    f"Available suites include: {sorted(list(benchmark_dict.keys()))[:12]}"
                )
            task_suite = benchmark_dict[task_suite_name]()

            expected_bddl_members = set()
            expected_init_members = set()
            for task_index in range(task_suite.get_num_tasks()):
                task = task_suite.get_task(task_index)
                expected_bddl_members.add(Path(task.bddl_file).stem)
                expected_init_members.add(
                    task.init_states_file.removesuffix(".pruned_init")
                )

            return expected_bddl_members, expected_init_members

        def read_language_from_bddl(bddl_path: Path, fallback: str = "") -> str:
            if not bddl_path.is_file():
                return str(fallback or "")
            content = bddl_path.read_text(encoding="utf-8")
            match = re.search(r"\(:language\s*(.*?)\)", content, re.IGNORECASE | re.DOTALL)
            if match:
                return " ".join(match.group(1).strip().split())
            if ":language" in content:
                return " ".join(content.split(":language", 1)[1].split(")", 1)[0].strip().split())
            return str(fallback or "")

        def build_ordered_task_descriptions(
            task_suite_name: str,
            bddl_dir: str,
        ) -> List[str]:
            benchmark_dict = _get_libero_pro_benchmark_dict(self.config)
            if task_suite_name not in benchmark_dict:
                raise KeyError(
                    f"Unknown LIBERO_PRO task suite '{task_suite_name}' while reading ordered descriptions."
                )
            task_suite = benchmark_dict[task_suite_name]()
            bddl_root = Path(bddl_dir)
            descriptions: List[str] = []
            for task_index in range(task_suite.get_num_tasks()):
                task = task_suite.get_task(task_index)
                bddl_name = Path(task.bddl_file).name
                bddl_path = bddl_root / bddl_name
                if not bddl_path.is_file():
                    matches = sorted(bddl_root.glob(f"{Path(task.bddl_file).stem}.bddl"))
                    if matches:
                        bddl_path = matches[0]
                language = read_language_from_bddl(
                    bddl_path,
                    fallback=getattr(task, "language", ""),
                )
                if not language:
                    raise RuntimeError(
                        "LIBERO_PRO generated task has no language description: "
                        f"suite={task_suite_name}, task_index={task_index}, bddl={bddl_path}"
                    )
                descriptions.append(language)
            return descriptions

        def validate_suite_assets(
            bddl_dir: str,
            init_dir: str,
            generated_task_suite_name: Optional[str] = None,
            expected_log: Optional[str] = None,
        ) -> Tuple[bool, str]:
            issues = []

            if not os.path.isdir(bddl_dir):
                issues.append(f"missing bddl dir: {bddl_dir}")
            if not os.path.isdir(init_dir):
                issues.append(f"missing init dir: {init_dir}")

            if expected_log is not None:
                log_path = os.path.join(bddl_dir, "log.txt")
                if not os.path.isfile(log_path):
                    issues.append(f"missing log file: {log_path}")
                else:
                    with open(log_path, "r", encoding="utf-8") as log_file:
                        log_contents = log_file.read().strip()
                    if log_contents != expected_log:
                        issues.append(
                            f"stale log state: expected {expected_log}, got {log_contents}"
                        )

            bddl_members = collect_suite_members(bddl_dir, ".bddl")
            init_members = collect_suite_members(init_dir, ".pruned_init")

            expected_bddl_members = bddl_members
            expected_init_members = init_members
            if generated_task_suite_name:
                try:
                    expected_bddl_members, expected_init_members = (
                        collect_expected_suite_members(generated_task_suite_name)
                    )
                except KeyError as exc:
                    issues.append(str(exc))

            if not bddl_members:
                issues.append(f"no bddl files found under {bddl_dir}")

            missing_bddl_members = sorted(expected_bddl_members - bddl_members)
            if missing_bddl_members:
                preview = ", ".join(missing_bddl_members[:3])
                if len(missing_bddl_members) > 3:
                    preview += f", ... ({len(missing_bddl_members)} total)"
                issues.append(f"missing benchmark bddl files for: {preview}")

            missing_init_members = sorted(expected_init_members - init_members)
            if missing_init_members:
                preview = ", ".join(missing_init_members[:3])
                if len(missing_init_members) > 3:
                    preview += f", ... ({len(missing_init_members)} total)"
                issues.append(f"missing init files for: {preview}")

            return len(issues) == 0, "; ".join(issues)

        def regenerate_suite_assets(
            bddl_dir: str,
            init_dir: str,
            generation_cfg: Dict[str, Any],
            reason: str,
            generated_task_suite_name: Optional[str] = None,
            expected_log: Optional[str] = None,
        ) -> None:
            print(
                f"[LIBERO PRO] Regenerating suite assets because {reason}",
                flush=True,
            )

            for dir_path in (bddl_dir, init_dir):
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                elif os.path.exists(dir_path):
                    os.remove(dir_path)

            if expected_log is not None:
                os.makedirs(bddl_dir, exist_ok=True)
                with open(
                    os.path.join(bddl_dir, "log.txt"),
                    "w",
                    encoding="utf-8",
                ) as log_file:
                    log_file.write(expected_log)

            _create_libero_pro_env_assets(configs=generation_cfg)

            assets_ok, assets_reason = validate_suite_assets(
                bddl_dir=bddl_dir,
                init_dir=init_dir,
                generated_task_suite_name=generated_task_suite_name,
                expected_log=expected_log,
            )
            if not assets_ok:
                raise RuntimeError(
                    "LIBERO_PRO generated incomplete suite assets. "
                    f"Target bddl dir: {bddl_dir}; target init dir: {init_dir}; "
                    f"validation error: {assets_reason}"
                )

        def append_suffix_once(task_suite_name: str, suffix: str) -> str:
            suffix = str(suffix or "").strip().strip("_")
            task_suite_name = str(task_suite_name or "").strip()
            if not suffix:
                return task_suite_name
            token = f"_{suffix}"
            if task_suite_name.endswith(token):
                return task_suite_name
            return f"{task_suite_name}{token}"

        for idx in range(num_task_suite):
            # _generate_minibatch_libero
            evaluation_config_path = self.config.libero_pro_eval_config_path
            evaluation_cfg = load_libero_pro_config(evaluation_config_path)

            raw_task_suite_name = raw_task_suite_names[idx]
            print(f"[LIBERO PRO] Old task suite name: {raw_task_suite_name}")
            # evaluation_cfg["bddl_files_path"] = os.path.join(
            #     evaluation_cfg.get("bddl_files_path", ""),
            #     raw_task_suite_name
            # )
            evaluation_cfg["task_suite_name"] = raw_task_suite_name

            use_swap = evaluation_cfg.get("use_swap", False)
            use_object = evaluation_cfg.get("use_object", False)
            use_language = evaluation_cfg.get("use_language", False)
            use_task = evaluation_cfg.get("use_task", False)
            use_environment = evaluation_cfg.get("use_environment", False)

            # rename task suite name
            # Step 1: Check if only one of the use_xxx flags is True
            if sum([use_swap, use_object, use_language, use_task, use_environment]) > 1:
                print(
                    "[LIBERO PRO]: Step 1-1: Check if only one of the use_xxx flags is True"
                )
                # If more than one flag is True, use the temp environment
                generated_task_suite_name = append_suffix_once(
                    raw_task_suite_name, "temp"
                )
                bddl_file_path = os.path.join(
                    evaluation_cfg.get("bddl_files_path", ""),
                    generated_task_suite_name,
                )
                init_file_path = os.path.join(
                    evaluation_cfg.get("init_file_dir", ""),
                    generated_task_suite_name,
                )
                expected_log = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"
                assets_ok, assets_reason = validate_suite_assets(
                    bddl_dir=bddl_file_path,
                    init_dir=init_file_path,
                    generated_task_suite_name=generated_task_suite_name,
                    expected_log=expected_log,
                )
                if not assets_ok:
                    regenerate_suite_assets(
                        bddl_dir=bddl_file_path,
                        init_dir=init_file_path,
                        generation_cfg=evaluation_cfg,
                        reason=assets_reason,
                        generated_task_suite_name=generated_task_suite_name,
                        expected_log=expected_log,
                    )
                # Update task_suite_name with "_temp" suffix
                prompts.non_tensor_batch["task_suite_name"][
                    idx
                ] = generated_task_suite_name
            # Step 2: Handle the case when only one use_xxx flag is True
            else:  # <= 1
                print(
                    "[LIBERO PRO]: Step 1-2: Handle the case when only one use_xxx flag is True"
                )
                if use_swap:
                    perturb_key = "use_swap"
                elif use_object:
                    perturb_key = "use_object"
                elif use_language:
                    perturb_key = "use_language"
                elif use_task:
                    perturb_key = "use_task"
                elif use_environment:
                    perturb_key = "use_environment"
                else:
                    raise ValueError("Must use one perturb type.")
                evaluation_cfg["perturb_flag"] = perturb_key

                #! env 就是 temp，暂时缺少 env 的，要自己生成
                perturb_suffix = evaluation_cfg.get("perturbation_mapping", {}).get(
                    perturb_key, ""
                )
                generated_task_suite_name = append_suffix_once(
                    raw_task_suite_name, perturb_suffix
                )
                bddl_file_path = os.path.join(
                    evaluation_cfg.get("bddl_files_path", ""),
                    generated_task_suite_name,
                )
                init_file_path = os.path.join(
                    evaluation_cfg.get("init_file_dir", ""),
                    generated_task_suite_name,
                )
                evaluation_cfg["perturbation"] = perturb_suffix
                assets_ok, assets_reason = validate_suite_assets(
                    bddl_dir=bddl_file_path,
                    init_dir=init_file_path,
                    generated_task_suite_name=generated_task_suite_name,
                )
                if not assets_ok:
                    regenerate_suite_assets(
                        bddl_dir=bddl_file_path,
                        init_dir=init_file_path,
                        generation_cfg=evaluation_cfg,
                        reason=assets_reason,
                        generated_task_suite_name=generated_task_suite_name,
                    )
                prompts.non_tensor_batch["task_suite_name"][
                    idx
                ] = generated_task_suite_name

            print(
                f"[LIBERO PRO] New task suite name: {prompts.non_tensor_batch['task_suite_name'][idx]}"
            )

            # get task info
            print("[LIBERO PRO]: Step 2: Get task info")
            # bddl_file_path = evaluation_cfg.get("bddl_files_path", "").rsplit('/', 1)[0] + prompts.non_tensor_batch["task_suite_name"][idx]
            bddl_file_path = os.path.join(
                evaluation_cfg.get("bddl_files_path", ""),
                prompts.non_tensor_batch["task_suite_name"][idx],
            )
            print(f"bddl_file_path: {bddl_file_path}")
            ood_task_description = build_ordered_task_descriptions(
                str(prompts.non_tensor_batch["task_suite_name"][idx]),
                bddl_file_path,
            )
            print(f"[LIBERO PRO] ood_task_description: {ood_task_description}")

            all_ood_task_descriptions.append(ood_task_description)

        return all_ood_task_descriptions
