from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_world.world_model.io import load_actions, read_image_sequence, read_video_frames
from real_world.world_model.schemas import RealWorldWMConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Preflight checks for real-world WM inference.")
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input-video", default=None)
    group.add_argument("--input-dir", default=None)
    parser.add_argument("--actions", default=None)
    parser.add_argument("--load-model", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with open(Path(args.config).expanduser(), "r", encoding="utf-8") as file_obj:
        config = RealWorldWMConfig.from_dict(yaml.safe_load(file_obj) or {})

    print(f"[preflight] config={args.config}")
    print(f"[preflight] wm_config_path={config.wm_config_path}")
    print(f"[preflight] ckpt_path={config.ckpt_path}")
    print(f"[preflight] device={config.device}, dtype={config.dtype}")

    if args.input_video:
        frames = read_video_frames(args.input_video, max_frames=config.num_history + 1)
        print(f"[preflight] decoded_video_frames={len(frames)}")
    if args.input_dir:
        frames = read_image_sequence(args.input_dir, max_frames=config.num_history + 1)
        print(f"[preflight] decoded_image_frames={len(frames)}")
    if args.actions:
        actions = load_actions(args.actions)
        print(f"[preflight] actions_shape={actions.shape}")

    if args.load_model:
        from real_world.world_model import RealWorldWMPredictor

        RealWorldWMPredictor.from_config(config)
        print("[preflight] model_load_ok")

    print("[preflight] real_world_wm_preflight_ok")


if __name__ == "__main__":
    main()
