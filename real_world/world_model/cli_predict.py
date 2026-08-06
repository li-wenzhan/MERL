from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from .io import load_actions, read_image_sequence, read_video_frames
from .predictor import RealWorldWMPredictor
from .schemas import RealWorldWMConfig, RealWorldWMRequest


def _load_config(path: str) -> RealWorldWMConfig:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as file_obj:
        payload: Dict[str, Any] = yaml.safe_load(file_obj) or {}
    return RealWorldWMConfig.from_dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Offline real-world Ctrl-World predictor")
    parser.add_argument("--config", required=True, help="YAML config for real-world WM inference.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-video", default=None, help="Input RGB camera video.")
    group.add_argument("--input-dir", default=None, help="Input image sequence directory.")
    parser.add_argument("--actions", required=True, help=".npy/.npz/.json/.jsonl action sequence.")
    parser.add_argument("--task", required=True, help="Task instruction text.")
    parser.add_argument("--output-dir", required=True, help="Directory for predicted video and rewards.")
    parser.add_argument("--max-input-frames", type=int, default=None)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--num-future-frames", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--camera-name", default="main")
    parser.add_argument("--input-fps", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load_config(args.config)
    if args.num_future_frames is not None:
        config.num_future_frames = int(args.num_future_frames)
    if args.num_inference_steps is not None:
        config.num_inference_steps = int(args.num_inference_steps)
    if args.device is not None:
        config.device = args.device

    if args.input_video:
        frames = read_video_frames(
            args.input_video,
            max_frames=args.max_input_frames,
            stride=args.video_stride,
        )
        input_source = args.input_video
    else:
        frames = read_image_sequence(args.input_dir, max_frames=args.max_input_frames)
        input_source = args.input_dir

    actions = load_actions(args.actions)
    request = RealWorldWMRequest(
        frames=frames,
        future_actions=actions,
        task_instruction=args.task,
        camera_name=args.camera_name,
        fps=args.input_fps or config.output_fps,
        metadata={
            "input_source": input_source,
            "actions_path": args.actions,
            "config_path": args.config,
        },
    )

    predictor = RealWorldWMPredictor.from_config(config)
    result = predictor.predict(request, output_dir=args.output_dir)
    print(f"[real-world-wm] pred_video={result.pred_video_path}")
    print(f"[real-world-wm] rewards={result.rewards_path}")
    print(f"[real-world-wm] result={result.result_path}")
    print(f"[real-world-wm] success_any={result.success_any}")


if __name__ == "__main__":
    main()
