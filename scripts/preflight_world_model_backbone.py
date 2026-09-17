#!/usr/bin/env python

import argparse
import importlib.util
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/wm_online_config.py")
    parser.add_argument("--checkpoint", help="Override and require the trained simulator checkpoint")
    parser.add_argument("--load-model", action="store_true", help="Strictly load all simulator parameters on CPU")
    args = parser.parse_args()

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
    if args.load_model:
        if not ckpt_path:
            raise SystemExit("[preflight] --load-model requires a trained checkpoint")
        from modules.ctrl_world.model_loading import load_trusted_state_dict
        from modules.ctrl_world.models.ctrl_world_new import CtrlWorld

        model = CtrlWorld(wm_args)
        state = load_trusted_state_dict(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"[preflight] strict simulator load OK: {len(state)} entries", flush=True)
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
