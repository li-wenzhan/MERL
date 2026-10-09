#!/usr/bin/env python

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


def _require_file(label: str, path: str) -> str:
    normalized = str(Path(path).expanduser().resolve())
    if not Path(normalized).is_file():
        raise SystemExit(f"[preflight] missing {label}: {normalized}")
    return normalized


def _load_wm_args(config_path: str):
    spec = importlib.util.spec_from_file_location("merl_wm_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[preflight] failed to import world-model config: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "wm_args"):
        raise SystemExit(f"[preflight] config does not define wm_args(): {config_path}")
    return module.wm_args()


def forward_smoke(model, cfg, image_path, instruction):
    """Exercise the GPU pipeline with a frame repeated across the history."""
    import numpy as np
    from PIL import Image
    from modules.ctrl_world.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

    model.eval().to(device="cuda", dtype=torch.bfloat16)
    pixels = np.array(Image.open(image_path).convert("RGB").resize((cfg.width, cfg.height)))
    image = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.bfloat16) / 127.5 - 1
    with torch.inference_mode():
        torch.manual_seed(0)
        latent = model.vae.encode(image).latent_dist.sample() * model.vae.config.scaling_factor
        history = latent.unsqueeze(1).repeat(1, cfg.num_history, 1, 1, 1)
        actions = torch.zeros(1, cfg.num_history + cfg.num_frames, cfg.action_dim,
                              device="cuda", dtype=torch.bfloat16)
        encoded = model.action_encoder(actions, [instruction], model.tokenizer, model.text_encoder, True)
        frames, latents = CtrlWorldDiffusionPipeline.__call__(
            model.pipeline, image=latent, text=encoded, width=cfg.width, height=cfg.height,
            num_frames=cfg.num_frames, history=history, num_inference_steps=2,
            decode_chunk_size=2, max_guidance_scale=cfg.guidance_scale,
            fps=cfg.fps, motion_bucket_id=cfg.motion_bucket_id, mask=None, output_type="frame",
            return_dict=False, frame_level_cond=True, his_cond_zero=cfg.his_cond_zero,
        )
        array = np.asarray(frames[0])
        if array.shape != (cfg.num_frames, cfg.height, cfg.width, 3) or not np.isfinite(array).all():
            raise RuntimeError("Invalid predicted frame shape or nonfinite pixels")
        images = torch.as_tensor(array, device="cuda", dtype=torch.bfloat16).permute(0, 3, 1, 2)
        if images.max() > 1:
            images = images / 255
        reward = model.reward_classifier.predict_score(images, encoded[0, -cfg.num_frames:])
        if not bool((torch.isfinite(reward) & (reward >= 0) & (reward <= 1)).all()):
            raise RuntimeError("Reward proxy must produce finite probabilities")
    return {"kind": "synthetic_repeated_history_pipeline",
            "frame_shape": list(array.shape), "latent_shape": list(latents.shape),
            "reward_shape": list(reward.shape), "finite": True, "inference_steps": 2,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/wm_online_config.py")
    parser.add_argument("--checkpoint", help="Override and require the trained simulator checkpoint")
    parser.add_argument("--load-model", action="store_true", help="Strictly load all simulator parameters on CPU")
    parser.add_argument("--image", type=Path, help="Also exercise GPU prediction on a real frame with synthetic history")
    parser.add_argument("--instruction", help="Task instruction matching --image")
    parser.add_argument("--output", type=Path, help="Write a new JSON forward-smoke result")
    args = parser.parse_args()
    if args.image and not args.instruction:
        parser.error("--image requires --instruction")
    if args.image:
        _require_file("input frame", args.image)
    if args.output and (not args.image or args.output.exists()):
        parser.error("--output requires --image and must not already exist")

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from modules.ctrl_world.model_loading import (
        prepare_local_hf_model_dir,
        resolve_ctrl_world_ckpt_path,
        summarize_weight_layout,
        torch_supports_safe_pickle_load,
    )

    cfg_path = _require_file("world-model config", args.config)
    wm_args = _load_wm_args(cfg_path)
    if args.checkpoint:
        wm_args.ckpt_path = _require_file("Ctrl-World checkpoint", args.checkpoint)
        wm_args.load_from_ckpt = True

    svd_resolved_path, svd_kwargs, svd_layout = prepare_local_hf_model_dir(
        getattr(wm_args, "svd_model_path", ""),
        component_name="Ctrl-World SVD backbone",
    )
    clip_resolved_path, clip_kwargs, clip_layout = prepare_local_hf_model_dir(
        getattr(wm_args, "clip_model_path", ""),
        component_name="Ctrl-World CLIP backbone",
    )

    ckpt_path = resolve_ctrl_world_ckpt_path(wm_args)
    warm_start = ckpt_path or "disabled"
    if args.load_model or args.image:
        if not ckpt_path:
            raise SystemExit("[preflight] --load-model requires a trained checkpoint")
        from modules.ctrl_world.model_loading import load_trusted_state_dict
        from modules.ctrl_world.models.ctrl_world_new import CtrlWorld

        model = CtrlWorld(wm_args)
        state = load_trusted_state_dict(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"[preflight] strict simulator load OK: {len(state)} entries", flush=True)
        del state
        if args.image:
            result = forward_smoke(model, wm_args, args.image, args.instruction)
            result["checkpoint"] = ckpt_path
            print(json.dumps(result), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "[preflight] world-model backbone compatibility OK "
        f"(torch={torch.__version__}, "
        f"pickle_supported={torch_supports_safe_pickle_load()}, "
        f"svd={summarize_weight_layout(svd_layout)}, svd_kwargs={svd_kwargs}, svd_runtime_path={svd_resolved_path}, "
        f"clip={summarize_weight_layout(clip_layout)}, clip_kwargs={clip_kwargs}, clip_runtime_path={clip_resolved_path}, "
        f"warm_start={warm_start})"
    )


if __name__ == "__main__":
    main()
