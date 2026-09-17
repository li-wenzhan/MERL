"""Persist complete, unselected real-environment evaluation episodes."""

import json
from pathlib import Path
import uuid

import numpy as np


def save_episode(root, frames, metadata):
    import imageio.v2 as imageio
    from PIL import Image

    folder = Path(root) / f"task_{metadata['task_id']:02d}_trial_{metadata['trial_id']:02d}_{uuid.uuid4().hex[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(metadata, frame_count=len(frames), video=None, keyframes=[])
    if frames:
        images = [np.asarray(frame) for frame in frames]
        if any(frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3
               or frame.shape != images[0].shape for frame in images):
            raise ValueError("Evaluation frames must be consistent uint8 RGB arrays")
        video = folder / "episode.mp4"
        with imageio.get_writer(str(video), fps=30) as writer:
            for frame in images:
                writer.append_data(frame)
        record.update(video="episode.mp4", playback_fps=30)
        for index in sorted(set(int(x) for x in np.linspace(0, len(images) - 1, 5))):
            name = f"frame_{index:05d}.png"
            Image.fromarray(images[index]).save(folder / name)
            record["keyframes"].append({"frame_index": index, "path": name})
    (folder / "episode.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return folder
