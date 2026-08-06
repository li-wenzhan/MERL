# Offline Real-World World Model Interface

Goal: run Ctrl-World on real camera inputs without Ray, LIBERO, or DataProto.

Input contract:

- RGB camera video or an image directory.
- Robot action sequence with shape `[T, 7]`.
- Task instruction text.

Action contract:

- `action_input_range: model` means actions are already in Ctrl-World model space `[-1, 1]`.
- `action_input_range: zero_one` maps actions from `[0, 1]` to `[-1, 1]`.
- If only future actions are provided, history actions are zero-filled and recorded in `result.json`.

Run:

```bash
python -m real_world.world_model.cli_predict \
  --config real_world/configs/real_wm_infer.yaml \
  --input-video /path/to/real_trial.mp4 \
  --actions /path/to/real_trial_actions.npy \
  --task "Put the red block into the box." \
  --output-dir ./outputs/real_world_wm/trial_001
```

HDF5 input:

```bash
HDF5=tmp_files/hdf5/episode_25.hdf5 TASK="Put the red block into the box." bash examples/real_world_wm_predict_hdf5.sh
```

The HDF5 wrapper exports a prediction window from the episode before calling the same
`cli_predict` entrypoint. By default it reads only the top-view camera
`observations/images/cam_high`, selects `action[:, 7:14]` as the right-arm action,
downsamples 30 Hz robot logs with `FRAME_STRIDE=8`, and writes the extracted mp4/npy
under the output directory. It also saves `*_current.png`, `*_gt_future.mp4`, and
`pred_future_frames/*.png` so the input view, ground-truth future top view, and generated
future top view can be inspected directly. Set `CURRENT_FRAME=<frame_id>` to choose the
current observation explicitly. `PREDICT_FROM_FRAME=<frame_id>` is the more explicit alias:
the wrapper uses that HDF5 frame as the current top-view observation and predicts the
future from there. Set `CURRENT_FRAME_RATIO=0.0..1.0` to choose a relative position inside
the valid prediction range. Set `EXTRACT_ONLY=1` to only export the selected current frame,
history window, and ground-truth future before running the expensive world model. Set
`NUM_INFERENCE_STEPS=<steps>` to override the denoising steps from
`real_world/configs/real_wm_infer.yaml` for one run.

Sliding HDF5 prediction:

```bash
HDF5=tmp_files/hdf5/episode_25.hdf5 TASK="Put the red block into the box." SLIDE_START_FRAME=250 SLIDE_END_FRAME=350 FRAME_STRIDE=1 ACTION_STRIDE=1 FPS=30 SLIDE_CHUNK_STEP=8 bash examples/real_world_wm_predict_hdf5.sh

# For example:
# HDF5=/mnt/afs/L202500276/data/hdf5/episode_0.hdf5 TASK="Put the red block into the box." SLIDE_START_FRAME=300 SLIDE_END_FRAME=396 FRAME_STRIDE=1 ACTION_STRIDE=1 FPS=30 SLIDE_CHUNK_STEP=8 OUTPUT_DIR=/mnt/afs/L202500276/project/MeRL_new/outputs/real_world_wm/episode_0_hdf5 bash examples/real_world_wm_predict_hdf5.sh
```

When `SLIDE_START_FRAME` and `SLIDE_END_FRAME` are set, the wrapper loads the world model
once and repeatedly predicts fixed-size chunks until the half-open range
`[SLIDE_START_FRAME, SLIDE_END_FRAME)` is covered. With `NUM_FUTURE=8` and
`SLIDE_CHUNK_STEP=8`, the chunk starts are `250, 258, 266, ...`; the last chunk is trimmed
so the concatenated output stops at `SLIDE_END_FRAME`.

Sliding mode writes image folders by default, not videos. Each chunk under
`sliding_chunks/chunk_XXXX/` contains:

- `history_frames/`: the `NUM_HISTORY` frames before the current frame.
- `gt_future_frames/`: ground-truth future frames from the HDF5 file.
- `pred_future_frames/`: world-model predicted future frames.

The current frame is aligned with `gt_future_frames/frame_000000.png`, and the exact
indices are recorded in each `chunk_manifest.json`.

The full stitched image sequences are saved under `gt_future_sliding_frames/` and
`pred_future_sliding_frames/`, with metadata in `sliding_manifest.json`. Set
`SAVE_SLIDE_VIDEOS=1` only if mp4 files are also needed.

Outputs:

- `pred_future.mp4`
- `input_history.mp4`
- `rewards.json`
- `result.json`
