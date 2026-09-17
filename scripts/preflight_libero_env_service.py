#!/usr/bin/env python
# LIBERO-PRO env-service preflight.
# 1. Start one long-lived env service in the same child-runtime path as rollout.
# 2. Reset a real LIBERO task and validate the first rendered observation.
# 3. Fail before Ray/model startup if MuJoCo/OpenGL is misconfigured.

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def _require_file(label: str, path: str) -> str:
    normalized = str(Path(path).expanduser().resolve())
    if not Path(normalized).is_file():
        raise SystemExit(f"[preflight] missing {label}: {normalized}")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/evaluation_config.yaml")
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--model-family", default="openvla")
    parser.add_argument("--num-steps-wait", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--step-count", type=int, default=1,
                        help="Number of zero-motion actions to exercise after reset")
    parser.add_argument("--output-dir", help="New directory for rendered smoke-test evidence")
    args = parser.parse_args()
    if args.step_count < 0 or args.step_count > args.max_steps - args.num_steps_wait:
        parser.error("step-count must fit within max-steps minus num-steps-wait")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=False)

    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    cfg_path = _require_file("LIBERO_PRO_EVAL_CONFIG_PATH", args.config)
    os.environ.setdefault("MERL_LIBERO_ENV_SERVICE_ENABLE", "true")

    from verl.workers.rollout.rob_rollout_wm_pro import (
        _LiberoEnvServiceClient,
        _build_libero_env_worker_config,
        _get_env_mp_context,
        _get_libero_primary_backend,
        _resolve_mujoco_egl_device_id,
    )

    rollout_config = SimpleNamespace(
        libero_pro_eval_config_path=cfg_path,
        model_family=str(args.model_family),
        num_steps_wait=int(args.num_steps_wait),
        env_init_timeout_s=float(args.timeout_s),
        env_step_timeout_s=60.0,
        env_init_parent_max_retry=1,
    )
    worker_config = _build_libero_env_worker_config(rollout_config)
    if _get_libero_primary_backend(worker_config) == "egl":
        render_devices = str(getattr(worker_config, "env_render_cuda_visible_devices", ""))
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            for item in [x.strip() for x in render_devices.split(",") if x.strip()] or [""]:
                if item:
                    os.environ["CUDA_VISIBLE_DEVICES"] = item
                resolved = _resolve_mujoco_egl_device_id(
                    SimpleNamespace(mujoco_egl_device_id="auto"),
                    visible_devices_override=render_devices,
                )
                if resolved != "0":
                    raise RuntimeError(
                        "auto EGL device must resolve to local MuJoCo device 0, "
                        f"got {resolved!r} with simulated CUDA_VISIBLE_DEVICES={item!r}"
                    )
        finally:
            if original_cuda_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
    mp_ctx = _get_env_mp_context(worker_config)
    client = None
    try:
        client = _LiberoEnvServiceClient(
            mp_ctx=mp_ctx,
            worker_config=worker_config,
            label="LIBERO_PRO preflight service",
            init_timeout_s=float(args.timeout_s),
            step_timeout_s=60.0,
        )
        init_data = client.reset(
            task_name=str(args.task_suite),
            task_id=int(args.task_id),
            trial_id=int(args.trial_id),
            is_valid=True,
            global_steps=0,
            max_steps=int(args.max_steps),
            ood_task_description="",
        )
        if init_data.get("type") != "init":
            raise RuntimeError(f"unexpected init message: {init_data}")
        image = init_data.get("image")
        image_arr = np.asarray(image)
        if image_arr.ndim != 3 or image_arr.shape[-1] != 3 or image_arr.size == 0:
            raise RuntimeError(f"invalid rendered image shape: {image_arr.shape}")
        if not bool(init_data.get("active", False)):
            raise RuntimeError("env service reset returned inactive state")

        frames = [image_arr]
        step_data = None
        if args.step_count:
            actions = np.zeros((args.step_count, 7), dtype=np.float32)
            actions[:, -1] = 1.0
            client.submit_step(actions)
            step_data = client.recv_step(timeout=60.0, context="preflight zero-motion step")
            if step_data.get("type") != "step":
                raise RuntimeError(f"env step failed: {step_data}")
            returned_frames = step_data.get("env_images", [])
            if not returned_frames or len(returned_frames) > args.step_count:
                raise RuntimeError("env step returned an invalid frame count")
            if len(returned_frames) != args.step_count and not step_data.get("complete", False):
                raise RuntimeError("env step stopped early without success")
            for frame in returned_frames:
                if np.asarray(frame).shape != image_arr.shape:
                    raise RuntimeError("env step changed rendered image shape")
            frames.extend(returned_frames)

        if output_dir:
            from PIL import Image
            for index, frame in enumerate(frames):
                Image.fromarray(np.asarray(frame)).save(output_dir / f"frame_{index:03d}.png")
            payload = {
                "purpose": "environment_smoke_only_not_policy_evaluation",
                "config": cfg_path, "task_suite": args.task_suite,
                "task_id": args.task_id, "trial_id": args.trial_id,
                "backend": _get_libero_primary_backend(worker_config),
                "action_count": len(frames) - 1,
                "task_description": init_data.get("task_description", ""),
                "success": bool(step_data and step_data.get("complete", False)),
            }
            (output_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")

        print(
            "[preflight] LIBERO-PRO env service OK "
            f"(backend={_get_libero_primary_backend(worker_config)}, "
            f"cuda_visible={getattr(worker_config, 'env_render_cuda_visible_devices', '')}, "
            f"egl_device={getattr(worker_config, 'mujoco_egl_device_id', '')}, "
            f"task={args.task_suite}:{args.task_id}, trial={args.trial_id}, "
            f"image_shape={tuple(image_arr.shape)}, actions={len(frames) - 1})"
        )
    except Exception as exc:
        raise SystemExit(f"[preflight] LIBERO-PRO env service failed: {exc}") from exc
    finally:
        if client is not None:
            client.close(force=True)


if __name__ == "__main__":
    main()
