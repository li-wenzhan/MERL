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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

# import sys
# sys.path.insert(0, "/path/to/SimpleVLA-RL")
# import libero_safe_patch

import json
import faulthandler
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import warnings
from functools import partial

import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayTrainer
from verl.trainer.ppo.dataproto_filter import DataProtoFilter
from verl.utils.reward_score import countdown, gsm8k, logic, math, multiply

warnings.filterwarnings(
    "ignore",
    message="Batch mode enable graph is only supported with num_graph_seeds==1",
)


class RobRewardManager:
    """The reward manager."""

    # TODO: we are requiring a reward manager to be much more stronger than this. so this is fully refactored!
    def __init__(self, num_examine, config) -> None:
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config = config

    def verify(self, data):
        completion_key = "env_complete" if "env_complete" in data.batch else "complete"
        completes = data.batch[completion_key].tolist()
        batch_size = data.batch["responses"].size(0)
        if len(completes) != batch_size:
            print(
                "[reward_manager] WARNING: completion length mismatch "
                f"({len(completes)} vs batch_size={batch_size}); "
                "truncating/padding with failures instead of interrupting training.",
                flush=True,
            )
            completes = list(completes)[:batch_size]
            if len(completes) < batch_size:
                completes.extend([False] * (batch_size - len(completes)))
        score = [float(item) for item in completes]
        format = [1.0 for _ in range(len(completes))]

        data.batch["acc"] = torch.tensor(
            score, dtype=torch.float32, device=data.batch["responses"].device
        )
        data.batch["format_correctness"] = torch.tensor(
            format, dtype=torch.float32, device=data.batch["responses"].device
        )

        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}

        reward_metrics["all"] = data.batch["acc"].mean().item()
        reward_metrics["completion_key_is_env"] = (
            1.0 if completion_key == "env_complete" else 0.0
        )
        format_metrics["all"] = data.batch["format_correctness"].mean().item()
        reward_format_metrics["all"] = data.batch["acc"].mean().item()

        return score, reward_metrics, format_metrics, reward_format_metrics

    def _normalize_rm_scores(
        self,
        data: DataProto,
        verifier_reward: torch.Tensor,
        valid_response_length: torch.Tensor,
    ) -> torch.Tensor:
        responses = data.batch["responses"]
        rm_scores = data.batch["rm_scores"].to(torch.float32)
        target_shape = verifier_reward.shape
        batch_size, target_flat_len = target_shape
        action_token_len = max(
            1, int(self.config.actor_rollout_ref.model.action_token_len)
        )
        action_chunks_len = max(
            1, int(self.config.actor_rollout_ref.model.action_chunks_len)
        )

        if tuple(rm_scores.shape) == tuple(responses.shape):
            return rm_scores.reshape(target_shape)
        if rm_scores.numel() == verifier_reward.numel():
            return rm_scores.reshape(target_shape)

        if rm_scores.shape[0] != batch_size:
            if rm_scores.numel() % batch_size != 0:
                print(
                    "[reward_manager] WARNING: incompatible rm_scores shape "
                    f"{tuple(rm_scores.shape)} for responses shape {tuple(responses.shape)}; "
                    "dropping malformed WM rewards.",
                    flush=True,
                )
                return torch.zeros_like(verifier_reward)
            rm_scores = rm_scores.reshape(batch_size, -1)
        else:
            rm_scores = rm_scores.reshape(batch_size, -1)

        normalized = torch.zeros_like(verifier_reward)
        action_tensor = data.batch.get("action", None)
        max_action_slots = (
            int(action_tensor.shape[1])
            if action_tensor is not None and action_tensor.dim() >= 2
            else max(1, target_flat_len // action_token_len)
        )
        response_steps = int(responses.shape[1]) if responses.dim() > 2 else 1
        used_legacy_projection = False

        for sample_idx in range(batch_size):
            sample_scores = rm_scores[sample_idx]
            sample_len = int(sample_scores.numel())
            if sample_len <= 0:
                continue

            sample_valid_tokens = int(valid_response_length[sample_idx].item())
            sample_valid_tokens = max(0, min(sample_valid_tokens, target_flat_len))
            if sample_valid_tokens <= 0:
                continue

            sample_valid_actions = min(
                max_action_slots,
                max(0, sample_valid_tokens // action_token_len),
            )

            if sample_len == sample_valid_tokens:
                normalized[sample_idx, :sample_valid_tokens] = sample_scores[
                    :sample_valid_tokens
                ]
                continue

            if sample_len == 1:
                normalized[sample_idx, sample_valid_tokens - 1] = sample_scores[0]
                used_legacy_projection = True
                continue

            if responses.dim() > 2 and sample_len == response_steps:
                valid_response_steps = min(
                    response_steps,
                    (sample_valid_actions + action_chunks_len - 1) // action_chunks_len,
                )
                for step_idx in range(valid_response_steps):
                    end_action = min(
                        (step_idx + 1) * action_chunks_len,
                        sample_valid_actions,
                    )
                    token_idx = min(
                        end_action * action_token_len - 1,
                        target_flat_len - 1,
                    )
                    if token_idx >= 0:
                        normalized[sample_idx, token_idx] = sample_scores[step_idx]
                used_legacy_projection = True
                continue

            if sample_len <= sample_valid_actions:
                valid_action_scores = min(sample_len, sample_valid_actions)
                token_indices = (
                    (torch.arange(valid_action_scores, device=sample_scores.device) + 1)
                    * action_token_len
                    - 1
                ).clamp(max=target_flat_len - 1)
                normalized[sample_idx, token_indices.long()] = sample_scores[
                    :valid_action_scores
                ]
                used_legacy_projection = True
                continue

            if sample_valid_tokens % sample_len == 0:
                bucket = sample_valid_tokens // sample_len
                token_indices = (
                    (torch.arange(sample_len, device=sample_scores.device) + 1) * bucket
                    - 1
                ).clamp(max=target_flat_len - 1)
                normalized[sample_idx, token_indices.long()] = sample_scores
                used_legacy_projection = True
                continue

            normalized[sample_idx, sample_valid_tokens - 1] = sample_scores.max()
            used_legacy_projection = True

        if used_legacy_projection:
            print(
                "[reward_manager] Normalized legacy rm_scores shape "
                f"{tuple(data.batch['rm_scores'].shape)} to token-level shape {target_shape}.",
                flush=True,
            )

        return normalized

    def _get_valid_response_length(self, data: DataProto) -> torch.Tensor:
        batch = data.batch
        batch_size = batch["responses"].shape[0]
        device = batch["responses"].device
        action_token_len = max(
            1, int(self.config.actor_rollout_ref.model.action_token_len)
        )
        return DataProtoFilter._per_sample_valid_response_tokens(
            batch=batch,
            B=batch_size,
            device=device,
            action_token_len=action_token_len,
        )

    def _world_model_cfg(self):
        return getattr(
            getattr(self.config, "actor_rollout_ref", None),
            "world_model",
            None,
        )

    def _train_mode(self) -> str:
        return str(
            getattr(getattr(self.config, "trainer", None), "train_mode", "MERL")
            or "MERL"
        ).upper()

    def _world_model_bool(self, key: str, default: bool) -> bool:
        wm_cfg = self._world_model_cfg()
        try:
            value = getattr(wm_cfg, key)
        except Exception:
            try:
                value = wm_cfg.get(key, default) if wm_cfg is not None else default
            except Exception:
                value = default
        return bool(value)

    def _use_wm_reward_proxy(self) -> bool:
        # MERL trains the reward model, but by default does not route its proxy
        # score into actor/critic objectives. MBRL keeps the historical behavior.
        train_mode = self._train_mode()
        default = train_mode != "MERL"
        return self._world_model_bool("use_wm_reward_proxy", default)

    def _sample_field(
        self,
        data: DataProto,
        key: str,
        *,
        default: float,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        batch_size = data.batch["responses"].shape[0]
        device = data.batch["responses"].device
        if key not in data.batch:
            return torch.full((batch_size,), default, device=device, dtype=dtype)
        value = data.batch[key]
        if value.dim() > 1:
            value = value.reshape(batch_size, -1)[:, 0]
        return value.to(device=device, dtype=dtype).reshape(batch_size)

    def __call__(self, data: DataProto):

        # aggregate all available reward tensors

        reward_tensor_dict = {}
        reward_metrics = {}
        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )  # batch * 64 * 56
        verifier_reward = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_tensor = reward_tensor.reshape((reward_tensor.shape[0], -1))
        verifier_reward = verifier_reward.reshape((verifier_reward.shape[0], -1))

        is_wm = data.batch.get("is_wm", None)
        if is_wm is None:
            is_wm = torch.zeros(
                (verifier_reward.shape[0],),
                dtype=torch.bool,
                device=verifier_reward.device,
            )
        else:
            if is_wm.dim() > 1:
                is_wm = is_wm.view(is_wm.shape[0], -1)[:, 0]
            is_wm = is_wm.to(device=verifier_reward.device, dtype=torch.bool)

        valid_response_length = self._get_valid_response_length(data)
        train_mode = self._train_mode()
        use_wm_reward_proxy = self._use_wm_reward_proxy()
        wm_real_anchor_reward = self._world_model_bool(
            "wm_real_anchor_reward",
            train_mode == "MERL" and not use_wm_reward_proxy,
        )
        anchor_reward = self._sample_field(
            data, "anchor_reward", default=0.0, dtype=torch.float32
        )
        has_anchor_reward = (
            self._sample_field(data, "has_anchor_reward", default=0.0) > 0.5
        )

        if "acc" in data.batch:
            # the separated rewards have been logged; now we add format correctness back for reward shaping
            # verifier_score = data.batch['acc'].cpu().numpy().tolist() + (0.0 * data.batch['format_correctness'].cpu().numpy()).tolist()
            verifier_score = data.batch["acc"].cpu().numpy().tolist()
        else:
            verifier_score, verifier_metrics, format_metrics, reward_format_metrics = (
                self.verify(data)
            )
            reward_metrics.update(verifier_metrics)

        verifier_score_t = torch.as_tensor(
            verifier_score,
            dtype=torch.float32,
            device=verifier_reward.device,
        ).reshape(verifier_reward.shape[0])
        use_anchor_mask = is_wm & has_anchor_reward & bool(wm_real_anchor_reward)
        terminal_reward = torch.where(use_anchor_mask, anchor_reward, verifier_score_t)
        for i in range(verifier_reward.shape[0]):
            end_idx = int(valid_response_length[i].item()) - 1
            if end_idx < 0:
                continue
            if bool(is_wm[i].item()) and not bool(use_anchor_mask[i].item()):
                continue
            verifier_reward[i, end_idx] += terminal_reward[i]

        reward_tensor_dict["gt_scores"] = verifier_reward
        reward_metrics["wm_real_anchor_reward_enabled"] = (
            1.0 if bool(wm_real_anchor_reward) else 0.0
        )
        if bool(is_wm.any().item()):
            wm_has_anchor = has_anchor_reward[is_wm]
            reward_metrics["wm_anchor_reward_coverage"] = (
                wm_has_anchor.float().mean().item()
                if wm_has_anchor.numel() > 0
                else 0.0
            )
            anchored_values = anchor_reward[is_wm & has_anchor_reward]
            reward_metrics["wm_anchor_reward_mean"] = (
                anchored_values.float().mean().item()
                if anchored_values.numel() > 0
                else 0.0
            )
        else:
            reward_metrics["wm_anchor_reward_coverage"] = 0.0
            reward_metrics["wm_anchor_reward_mean"] = 0.0

        if "rm_scores" in data.batch.keys():
            raw_rm_scores = self._normalize_rm_scores(
                data=data,
                verifier_reward=verifier_reward,
                valid_response_length=valid_response_length,
            )
            raw_rm_scores = raw_rm_scores * is_wm.unsqueeze(1).to(raw_rm_scores.dtype)
            reward_metrics["reward_model_raw"] = (
                raw_rm_scores.sum(dim=1).mean().item()
            )
            if bool(is_wm.any().item()):
                reward_metrics["reward_model_wm_raw_mean"] = (
                    raw_rm_scores.sum(dim=1)[is_wm].mean().item()
                )
            else:
                reward_metrics["reward_model_wm_raw_mean"] = 0.0

            reward_metrics["reward_model_disabled"] = (
                0.0 if use_wm_reward_proxy else 1.0
            )
            rm_scores = (
                raw_rm_scores if use_wm_reward_proxy else torch.zeros_like(raw_rm_scores)
            )
            if not use_wm_reward_proxy:
                data.batch["rm_scores"] = torch.zeros_like(
                    data.batch["rm_scores"], dtype=torch.float32
                )
            reward_tensor_dict["rm_scores"] = rm_scores
            reward_metrics["reward_model"] = rm_scores.sum(dim=1).mean().item()
            if bool(is_wm.any().item()):
                reward_metrics["reward_model_wm_mean"] = (
                    rm_scores.sum(dim=1)[is_wm].mean().item()
                )
            else:
                reward_metrics["reward_model_wm_mean"] = 0.0
            if use_wm_reward_proxy:
                reward_tensor += rm_scores
        else:
            reward_metrics["reward_model_raw"] = 0.0
            reward_metrics["reward_model_wm_raw_mean"] = 0.0
            reward_metrics["reward_model_disabled"] = (
                0.0 if use_wm_reward_proxy else 1.0
            )
            reward_metrics["reward_model"] = 0.0
            reward_metrics["reward_model_wm_mean"] = 0.0

        if self.config.verifier.reward_coef != 0:

            reward_metrics["verifier"] = (
                reward_tensor_dict["gt_scores"].sum(dim=1).mean().item()
            )
            reward_tensor += (
                self.config.verifier.reward_coef * reward_tensor_dict["gt_scores"]
            )

        reward_tensor_dict["all"] = reward_tensor
        reward_metrics["reward_all"] = reward_tensor.sum(dim=-1).mean(dim=0).item()
        reward_metrics["all"] = reward_metrics["reward_all"]

        return reward_tensor_dict, reward_metrics


import hydra
import ray


def _str_to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default

    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_ray_runtime_env(runtime_env_path):
    loaded = {}
    if runtime_env_path and os.path.isfile(str(runtime_env_path)):
        with open(str(runtime_env_path), "r") as f:
            loaded = json.load(f) or {}

    env_vars = {
        str(key): str(value)
        for key, value in (loaded.get("env_vars", {}) or {}).items()
    }
    return loaded, env_vars


def _resolve_ray_tmpdir(ray_tmpdir: str) -> str:
    ray_tmpdir = str(ray_tmpdir or "").strip()
    if not ray_tmpdir:
        return ""

    ray_tmpdir = os.path.abspath(os.path.expanduser(ray_tmpdir))
    socket_probe = os.path.join(
        ray_tmpdir,
        "session_YYYY-MM-DD_HH-MM-SS_000000_000000",
        "sockets",
        "plasma_store",
    )
    if os.name != "nt" and len(socket_probe) >= 100:
        uid = os.getuid() if hasattr(os, "getuid") else "u"
        fallback_tmpdir = os.path.abspath(
            os.environ.get("MERL_RAY_SHORT_TMPDIR", f"/tmp/merl_ray_{uid}")
        )
        print(
            "[main_ppo] WARNING: Ray tmpdir is too long for Unix socket paths; "
            f"fallback from '{ray_tmpdir}' to '{fallback_tmpdir}'.",
            flush=True,
        )
        ray_tmpdir = fallback_tmpdir
        socket_probe = os.path.join(
            ray_tmpdir,
            "session_YYYY-MM-DD_HH-MM-SS_000000_000000",
            "sockets",
            "plasma_store",
        )

    if os.name != "nt" and len(socket_probe) >= 100:
        raise OSError(
            "Ray tmpdir is too long for Unix socket paths even after fallback: "
            f"tmpdir='{ray_tmpdir}', estimated_socket='{socket_probe}'. "
            "Set MERL_RAY_SHORT_TMPDIR to a short path such as /tmp/merl_ray."
        )

    os.makedirs(ray_tmpdir, exist_ok=True)
    os.environ["RAY_TMPDIR"] = ray_tmpdir
    return ray_tmpdir


def _as_optional_number(value, cast_fn=float):
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return cast_fn(raw)
    except Exception:
        return None


def _start_ray_init_process_watchdog(timeout_s: float):
    if os.name == "nt" or timeout_s <= 0:
        return None
    parent_pid = os.getpid()
    timeout_msg = (
        f"[main_ppo] FATAL: ray.init did not return within {float(timeout_s):.1f}s; "
        "external watchdog terminating the stuck driver.\n"
    )
    code = (
        "import os, signal, sys, time\n"
        f"time.sleep({float(timeout_s)!r})\n"
        f"sys.stderr.write({timeout_msg!r})\n"
        "sys.stderr.flush()\n"
        "try:\n"
        f"    os.kill({parent_pid}, signal.SIGTERM)\n"
        "except ProcessLookupError:\n"
        "    pass\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=None,
        start_new_session=True,
    )


def _stop_ray_init_process_watchdog(watchdog):
    if watchdog is None or watchdog.poll() is not None:
        return
    watchdog.terminate()
    try:
        watchdog.wait(timeout=2)
    except subprocess.TimeoutExpired:
        watchdog.kill()


def _ray_init_with_timeout(ray_init_kwargs, timeout_s: float):
    timeout_s = float(timeout_s or 0)
    if timeout_s <= 0:
        return ray.init(**ray_init_kwargs)

    timeout_event = threading.Event()
    process_watchdog = _start_ray_init_process_watchdog(timeout_s + 5.0)

    def _hard_timeout():
        if timeout_event.wait(timeout_s):
            return
        print(
            f"[main_ppo] FATAL: ray.init did not return within {timeout_s:.1f}s; "
            "terminating to avoid an indefinite Ray startup hang.",
            flush=True,
        )
        faulthandler.dump_traceback(all_threads=True)
        os._exit(124)

    watchdog = threading.Thread(
        target=_hard_timeout,
        name="ray-init-hard-timeout",
        daemon=True,
    )
    watchdog.start()

    if os.name == "nt" or not hasattr(signal, "SIGALRM"):
        try:
            return ray.init(**ray_init_kwargs)
        finally:
            timeout_event.set()
            _stop_ray_init_process_watchdog(process_watchdog)

    def _handle_timeout(signum, frame):
        raise TimeoutError(
            f"ray.init did not return within {timeout_s:.1f}s; "
            "check Ray gcs/raylet logs under the configured Ray tmpdir."
        )

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return ray.init(**ray_init_kwargs)
    finally:
        timeout_event.set()
        _stop_ray_init_process_watchdog(process_watchdog)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _build_ray_init_kwargs(config):
    runtime_env_path = getattr(config.trainer, "runtime_env", "")
    loaded_runtime_env, env_vars = _load_ray_runtime_env(runtime_env_path)
    if not env_vars:
        env_vars = {
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        }
    if "MERL_GLOBAL_CUDA_VISIBLE_DEVICES" in os.environ:
        env_vars.setdefault(
            "MERL_GLOBAL_CUDA_VISIBLE_DEVICES",
            os.environ["MERL_GLOBAL_CUDA_VISIBLE_DEVICES"],
        )
    if "MERL_LIBERO_RENDER_CUDA_VISIBLE_DEVICES" in os.environ:
        env_vars.setdefault(
            "MERL_LIBERO_RENDER_CUDA_VISIBLE_DEVICES",
            os.environ["MERL_LIBERO_RENDER_CUDA_VISIBLE_DEVICES"],
        )
    if "MERL_LIBERO_ENV_SERVICE_ENABLE" in os.environ:
        env_vars.setdefault(
            "MERL_LIBERO_ENV_SERVICE_ENABLE",
            os.environ["MERL_LIBERO_ENV_SERVICE_ENABLE"],
        )

    for key, value in env_vars.items():
        os.environ[key] = value

    ray_address = str(
        getattr(
            config.trainer,
            "ray_address",
            os.environ.get("MERL_RAY_ADDRESS", os.environ.get("RAY_ADDRESS", "")),
        )
        or ""
    ).strip()
    force_local_ray = _str_to_bool(
        os.environ.get("MERL_RAY_FORCE_LOCAL", "true"),
        default=True,
    )
    use_local_ray = (
        not ray_address and force_local_ray
    ) or ray_address.lower() == "local"

    if use_local_ray:
        os.environ.pop("RAY_ADDRESS", None)
        os.environ.pop("MERL_RAY_ADDRESS", None)

    runtime_env_mode = (
        str(
            getattr(
                config.trainer,
                "ray_runtime_env_mode",
                os.environ.get("MERL_RAY_RUNTIME_ENV_MODE", "env_vars_only"),
            )
            or "env_vars_only"
        )
        .strip()
        .lower()
    )

    if runtime_env_mode in {"none", "disabled", "off"}:
        runtime_env = None
    elif runtime_env_mode in {"full", "raw"}:
        runtime_env = (
            loaded_runtime_env if loaded_runtime_env else {"env_vars": env_vars}
        )
    else:
        runtime_env = {"env_vars": env_vars}

    ray_tmpdir = str(
        getattr(config.trainer, "ray_tmpdir", os.environ.get("MERL_RAY_TMPDIR", ""))
        or ""
    ).strip()
    ray_tmpdir = _resolve_ray_tmpdir(ray_tmpdir)

    include_dashboard = _str_to_bool(
        getattr(
            config.trainer,
            "ray_include_dashboard",
            os.environ.get("MERL_RAY_INCLUDE_DASHBOARD", "false"),
        ),
        default=False,
    )

    ray_init_kwargs = {
        "log_to_driver": True,
    }
    if use_local_ray:
        ray_init_kwargs["address"] = "local"
    elif ray_address:
        ray_init_kwargs["address"] = ray_address
    if use_local_ray or not ray_address:
        ray_node_ip = str(
            getattr(
                config.trainer,
                "ray_node_ip_address",
                os.environ.get(
                    "MERL_RAY_NODE_IP",
                    os.environ.get("RAY_NODE_IP_ADDRESS", "127.0.0.1"),
                ),
            )
            or ""
        ).strip()
        if ray_node_ip:
            ray_init_kwargs["_node_ip_address"] = ray_node_ip
        ray_init_kwargs["include_dashboard"] = include_dashboard
        if ray_tmpdir:
            ray_init_kwargs["_temp_dir"] = ray_tmpdir

        ray_num_gpus = _as_optional_number(
            getattr(
                config.trainer, "ray_num_gpus", os.environ.get("MERL_RAY_NUM_GPUS", "")
            )
        )
        if ray_num_gpus is not None:
            ray_init_kwargs["num_gpus"] = ray_num_gpus

        ray_num_cpus = _as_optional_number(
            getattr(
                config.trainer, "ray_num_cpus", os.environ.get("MERL_RAY_NUM_CPUS", "")
            ),
            cast_fn=int,
        )
        if ray_num_cpus is not None and ray_num_cpus > 0:
            ray_init_kwargs["num_cpus"] = ray_num_cpus

    if runtime_env is not None:
        ray_init_kwargs["runtime_env"] = runtime_env

    ignored_runtime_env_keys = []
    if runtime_env_mode not in {"full", "raw"}:
        ignored_runtime_env_keys = sorted(set(loaded_runtime_env.keys()) - {"env_vars"})

    return ray_init_kwargs, {
        "runtime_env_path": str(runtime_env_path),
        "address": "local" if use_local_ray else (ray_address or "<python-local>"),
        "force_local": force_local_ray,
        "runtime_env_mode": runtime_env_mode,
        "env_var_keys": sorted(env_vars.keys()),
        "tmpdir": ray_tmpdir or "<ray-default>",
        "include_dashboard": include_dashboard,
        "ignored_runtime_env_keys": ignored_runtime_env_keys,
    }


@ray.remote(num_cpus=0)
def _ray_startup_probe():
    return {
        "pid": os.getpid(),
        "time": time.time(),
        "cwd": os.getcwd(),
        "display": os.environ.get("DISPLAY", ""),
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM", ""),
        "merl_env_mp_start_method": os.environ.get("MERL_ENV_MP_START_METHOD", ""),
        "merl_global_cuda_visible_devices": os.environ.get(
            "MERL_GLOBAL_CUDA_VISIBLE_DEVICES", ""
        ),
        "merl_render_cuda_visible_devices": os.environ.get(
            "MERL_LIBERO_RENDER_CUDA_VISIBLE_DEVICES", ""
        ),
        "merl_libero_egl_device_id": os.environ.get("MERL_LIBERO_EGL_DEVICE_ID", ""),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID", ""),
    }


def _as_non_negative_int(value) -> int:
    try:
        return max(0, int(float(value)))
    except Exception:
        return 0


def _get_alive_ray_node_rows(ray_nodes=None):
    rows = []
    for node in ray_nodes if ray_nodes is not None else ray.nodes():
        if not bool(node.get("Alive", False)):
            continue
        resources = node.get("Resources", {}) or {}
        gpu_count = _as_non_negative_int(resources.get("GPU", 0))
        cpu_count = _as_non_negative_int(resources.get("CPU", 0))
        rows.append(
            {
                "node_id": str(node.get("NodeID", "unknown")),
                "gpu": gpu_count,
                "cpu": cpu_count,
                "slots": min(gpu_count, cpu_count),
            }
        )
    return rows


def _get_available_ray_slot_count() -> int:
    try:
        resources = ray.available_resources() or {}
    except Exception:
        return 0

    available_gpu = _as_non_negative_int(resources.get("GPU", 0))
    available_cpu = _as_non_negative_int(resources.get("CPU", 0))
    return min(available_gpu, available_cpu)


def _build_process_layout(node_rows, world_size: int):
    remaining = int(world_size)
    layout = []
    chosen_rows = []
    for row in sorted(
        [item for item in node_rows if int(item.get("slots", 0)) > 0],
        key=lambda item: (int(item["slots"]), int(item["gpu"]), int(item["cpu"])),
        reverse=True,
    ):
        if remaining <= 0:
            break
        take = min(int(row["slots"]), remaining)
        if take <= 0:
            continue
        layout.append(take)
        chosen_rows.append({**row, "used": take})
        remaining -= take

    if remaining != 0:
        raise RuntimeError(
            f"failed to assign actor world_size={world_size}; remaining={remaining}, node_rows={node_rows}"
        )
    return layout, chosen_rows


def _validate_runtime_world_size(config, world_size: int):
    checks = {
        "data.train_batch_size": _as_non_negative_int(
            getattr(config.data, "train_batch_size", 0)
        ),
        "data.val_batch_size": _as_non_negative_int(
            getattr(config.data, "val_batch_size", 0)
        ),
        "actor.ppo_mini_batch_size": _as_non_negative_int(
            getattr(config.actor_rollout_ref.actor, "ppo_mini_batch_size", 0)
        ),
        "rollout.log_prob_micro_batch_size": _as_non_negative_int(
            getattr(config.actor_rollout_ref.rollout, "log_prob_micro_batch_size", 0)
        ),
    }

    if world_size <= 0:
        return False, "world_size must be > 0"

    for label, value in checks.items():
        if value <= 0:
            return False, f"{label} must be > 0, got {value}"
        if value % world_size != 0:
            return False, f"{label}={value} is not divisible by world_size={world_size}"

    return True, ""


def _resolve_actor_process_on_nodes(config):
    desired_world_size = _as_non_negative_int(
        getattr(config.trainer, "n_gpus_per_node", 0)
    ) * _as_non_negative_int(getattr(config.trainer, "nnodes", 0))
    train_mode = str(getattr(config.trainer, "train_mode", "MERL")).upper()
    reserve_wm_slot = int(
        bool(getattr(config.actor_rollout_ref.world_model, "enable", False))
        and (not bool(getattr(config.trainer, "val_only", False)))
        and train_mode != "MFRL"
    )

    node_rows = _get_alive_ray_node_rows()
    total_slots = sum(int(row["slots"]) for row in node_rows)
    max_actor_world_size = max(0, total_slots - reserve_wm_slot)
    available_slots = _get_available_ray_slot_count()
    if available_slots > 0:
        max_actor_world_size = min(
            max_actor_world_size,
            max(0, available_slots - reserve_wm_slot),
        )

    if total_slots <= 0:
        raise RuntimeError(
            "Ray reports no alive nodes with both CPU and GPU resources; "
            f"node_rows={node_rows}"
        )

    actual_world_size = 0
    reject_reasons = []
    for candidate in range(min(desired_world_size, max_actor_world_size), 0, -1):
        is_valid, reason = _validate_runtime_world_size(config, candidate)
        if is_valid:
            actual_world_size = candidate
            break
        reject_reasons.append(reason)

    if actual_world_size <= 0:
        raise RuntimeError(
            "Unable to find a feasible actor world size for the current Ray cluster. "
            f"desired_world_size={desired_world_size}, total_slots={total_slots}, "
            f"available_slots={available_slots}, reserve_wm_slot={reserve_wm_slot}, node_rows={node_rows}, "
            f"reject_reasons={reject_reasons}"
        )

    process_on_nodes, chosen_rows = _build_process_layout(node_rows, actual_world_size)
    return {
        "process_on_nodes": process_on_nodes,
        "desired_world_size": desired_world_size,
        "actual_world_size": actual_world_size,
        "reserve_wm_slot": reserve_wm_slot,
        "total_slots": total_slots,
        "available_slots": available_slots,
        "node_rows": node_rows,
        "chosen_rows": chosen_rows,
    }


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    # 1. Keep local Ray startup deterministic across resume/fresh debug runs.
    if ray.is_initialized():
        ray.shutdown()

    if not ray.is_initialized():
        ray_init_kwargs, ray_init_summary = _build_ray_init_kwargs(config)
        ray_init_timeout_s = float(
            getattr(
                config.trainer,
                "ray_init_timeout_s",
                os.environ.get("MERL_RAY_INIT_TIMEOUT_S", 180),
            )
        )
        print(f"[main_ppo] ray.init begin: {ray_init_summary}", flush=True)
        _ray_init_with_timeout(ray_init_kwargs, ray_init_timeout_s)

    print(
        "[main_ppo] ray.init done | "
        f"cluster_resources={ray.cluster_resources()} | "
        f"available_resources={ray.available_resources()}",
        flush=True,
    )

    probe_timeout_s = float(
        getattr(config.trainer, "ray_startup_probe_timeout_s", 180.0)
    )
    print(
        f"[main_ppo] submitting Ray startup probe (timeout={probe_timeout_s:.1f}s)",
        flush=True,
    )
    try:
        probe_result = ray.get(_ray_startup_probe.remote(), timeout=probe_timeout_s)
    except ray.exceptions.GetTimeoutError as exc:
        raise RuntimeError(
            "Ray started, but a zero-CPU startup probe could not run within "
            f"{probe_timeout_s:.1f}s. This is earlier than LIBERO/GLX env creation; "
            "check Ray worker startup/runtime_env logs, stale Ray sessions, and available CPU resources."
        ) from exc
    print(f"[main_ppo] Ray startup probe OK: {probe_result}", flush=True)

    print("[main_ppo] submitting main_task", flush=True)
    main_ref = main_task.remote(config)
    print("[main_ppo] waiting for main_task", flush=True)
    ray.get(main_ref)


def _configure_policy_evaluation(config):
    """Keep policy evaluation real-only regardless of the checkpoint's mode label."""
    if not bool(getattr(config.trainer, "val_only", False)):
        return False
    wm_cfg = getattr(config.actor_rollout_ref, "world_model", None)
    if wm_cfg is not None:
        for key in ("enable", "fine_tune", "fixed_eval_enabled"):
            if hasattr(wm_cfg, key):
                setattr(wm_cfg, key, False)
    return True


@ray.remote(num_cpus=0)
def main_task(config):
    print("[main_task] started; importing trainer dependencies", flush=True)
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf, open_dict
    from transformers import AutoTokenizer
    from verl.utils.fs import copy_local_path_from_hdfs

    print("[main_task] imports done; resolving config", flush=True)
    pprint(
        OmegaConf.to_container(config, resolve=True)
    )  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    policy_evaluation = _configure_policy_evaluation(config)

    actor_layout = _resolve_actor_process_on_nodes(config)
    if actor_layout["actual_world_size"] != actor_layout["desired_world_size"]:
        print(
            "[main_ppo] Adjusted actor world size to match Ray topology: "
            f"desired={actor_layout['desired_world_size']}, actual={actor_layout['actual_world_size']}, "
            f"process_on_nodes={actor_layout['process_on_nodes']}, available_slots={actor_layout['available_slots']}, "
            f"reserve_wm_slot={actor_layout['reserve_wm_slot']}",
            flush=True,
        )
    else:
        print(
            "[main_ppo] Actor world size resolved from Ray topology: "
            f"world_size={actor_layout['actual_world_size']}, process_on_nodes={actor_layout['process_on_nodes']}, "
            f"available_slots={actor_layout['available_slots']}, reserve_wm_slot={actor_layout['reserve_wm_slot']}",
            flush=True,
        )

    with open_dict(config.actor_rollout_ref.actor):
        config.actor_rollout_ref.actor.ppo_micro_batch_size = int(
            actor_layout["actual_world_size"]
        )

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import (
            ActorRolloutRefWorker,
            CriticWorker,
            RobActorRolloutRefWorker,
        )

        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import (
            ActorRolloutRefWorker,
            CriticWorker,
            RobActorRolloutRefWorker,
        )

        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    train_mode = str(getattr(config.trainer, "train_mode", "MERL")).upper()
    if train_mode not in ("MBRL", "MFRL", "MERL", "ONLINE_MBRL", "STATIC_TRUST"):
        print(f"[main_ppo] Unknown train_mode='{train_mode}', falling back to 'MERL'")
        train_mode = "MERL"
    config.trainer.train_mode = train_mode

    strict_mode_assert = bool(getattr(config.trainer, "strict_mode_assert", False))
    wm_cfg = getattr(config.actor_rollout_ref, "world_model", None)
    if wm_cfg is not None:
        with open_dict(wm_cfg):
            if "enable" not in wm_cfg:
                wm_cfg.enable = False
            if "fine_tune" not in wm_cfg:
                wm_cfg.fine_tune = False

    if strict_mode_assert and not policy_evaluation:
        project_name = str(getattr(config.trainer, "project_name", "")).upper()
        adv_estimator = str(getattr(config.algorithm, "adv_estimator", "")).lower()
        n_samples = int(getattr(config.data, "n_samples", 1))

        if project_name and project_name != train_mode:
            raise ValueError(
                f"[main_ppo] strict_mode_assert: project_name='{project_name}' does not match train_mode='{train_mode}'."
            )

        if adv_estimator == "grpo" and n_samples <= 1:
            raise ValueError(
                "[main_ppo] strict_mode_assert: GRPO requires data.n_samples > 1."
            )

        if wm_cfg is not None:
            expected_enable = train_mode != "MFRL"
            expected_fine_tune = train_mode in ("MERL", "ONLINE_MBRL")
            actual_enable = bool(getattr(wm_cfg, "enable", False))
            actual_fine_tune = bool(getattr(wm_cfg, "fine_tune", False))
            if actual_enable != expected_enable:
                raise ValueError(
                    "[main_ppo] strict_mode_assert: "
                    f"world_model.enable={actual_enable} but train_mode='{train_mode}' expects {expected_enable}."
                )
            if actual_fine_tune != expected_fine_tune:
                raise ValueError(
                    "[main_ppo] strict_mode_assert: "
                    f"world_model.fine_tune={actual_fine_tune} but train_mode='{train_mode}' expects {expected_fine_tune}."
                )

    if wm_cfg is not None and not policy_evaluation:
        if train_mode == "MFRL":
            if bool(getattr(wm_cfg, "enable", False)) or bool(
                getattr(wm_cfg, "fine_tune", False)
            ):
                print(
                    "[main_ppo] MFRL mode detected; disabling world model worker and WM updates."
                )
            wm_cfg.enable = False
            wm_cfg.fine_tune = False
        elif train_mode in ("MBRL", "STATIC_TRUST"):
            if not bool(getattr(wm_cfg, "enable", False)):
                print(
                    "[main_ppo] MBRL mode detected; enabling world model worker for imagined rollouts."
                )
            if bool(getattr(wm_cfg, "fine_tune", False)):
                print("[main_ppo] MBRL mode detected; freezing world model updates.")
            wm_cfg.enable = True
            wm_cfg.fine_tune = False
        elif train_mode == "ONLINE_MBRL":
            from merl.modes import validate_online_mbrl
            validate_online_mbrl(wm_cfg)
            wm_cfg.enable = True
            wm_cfg.fine_tune = True

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    if config.actor_rollout_ref.world_model.enable or config.trainer.get("engine") == "merl":
        print("Using World Model Actor Rollout Ref Worker.")
        from verl.workers.fsdp_workers import RobWMActorRolloutRefWorker

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(
                RobWMActorRolloutRefWorker
            ),  #! wm actor worker
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(RobWMActorRolloutRefWorker),  #! wm actor worker
        }
    else:
        print("Using Actor Rollout Ref Worker.")
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(RobActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(RobActorRolloutRefWorker),
        }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: actor_layout["process_on_nodes"],
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable and config.reward_model.rm_coef != 0.0:
        if config.reward_model.rm_type == "normal":
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        elif config.reward_model.rm_type == "prime":
            from verl.workers.fsdp_workers import PRIMERewardModelWorker

            role_worker_mapping[Role.RewardModel] = ray.remote(PRIMERewardModelWorker)
        else:
            raise NotImplementedError
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RobRewardManager(
        num_examine=0, config=config
    )  # note: verifier is called both inside reward_fn and outside.

    # Note that we always use function-based RM for validation
    val_reward_fn = RobRewardManager(num_examine=1, config=config)

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    print("[startup] constructing trainer", flush=True)
    trainer = RayTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    print("[startup] trainer constructed; initializing distributed workers", flush=True)
    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        trainer.init_workers()
    finally:
        faulthandler.cancel_dump_traceback_later()
    print("[startup] all workers initialized; entering training/evaluation", flush=True)
    if config.trainer.get("actor_checkpoint"):
        from merl.trainer import _require_workers
        _require_workers(trainer.actor_rollout_wg.load_checkpoint(config.trainer.actor_checkpoint), "loaded")
    if config.trainer.get("engine") == "merl" and not policy_evaluation and not config.trainer.get("rollout_before_train", False):
        from merl.trainer import fit
        fit(trainer)
    elif train_mode == "MFRL" or policy_evaluation:
        trainer.fit()
    else:
        trainer.fit_wm_v5()


if __name__ == "__main__":
    main()
