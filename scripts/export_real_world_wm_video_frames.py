from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_world.world_model.io import read_video_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Export a real-world WM mp4 to RGB PNG frames.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frames = read_video_frames(args.video, max_frames=args.max_frames, stride=args.stride)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        Image.fromarray(frame).save(output_dir / f"frame_{idx:06d}.png")
    print(f"[wm-video-frames] video={args.video}")
    print(f"[wm-video-frames] output_dir={output_dir}")
    print(f"[wm-video-frames] frames={len(frames)}")


if __name__ == "__main__":
    main()
