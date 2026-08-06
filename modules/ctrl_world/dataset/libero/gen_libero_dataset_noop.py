import argparse
import json
import os
import shutil
import time
from typing import Any, Dict

import h5py
import mediapy
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.wm_offline_config import wm_args as offline_wm_args
from verl.utils.libero_path import ensure_libero_root

ensure_libero_root(getattr(offline_wm_args, "libero_root", None))

from libero.libero import benchmark
from loguru import logger
from tqdm import tqdm


def create_directory(path: str, overwrite: bool = True):
    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def is_static_action(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a static action.

    A static action satisfies:
      (1) All action dimensions, except for the last one (gripper action), are near zero.
      (2) The gripper action equals the previous timestep's gripper action.
    """
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return (np.linalg.norm(action[:-1]) < threshold) and (
        gripper_action == prev_gripper_action
    )


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Keep original behavior: obs[t][::-1] (flip along axis 0).
    Ensure uint8 HWC 3-channel output for mediapy.write_video.
    """
    # flip as original code does
    frame = frame[::-1]

    # handle grayscale
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)

    # drop alpha if present
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]

    # convert to uint8
    if frame.dtype != np.uint8:
        # if in [0,1], scale
        if np.max(frame) <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def save_traj_aggregated(
    traj_save_dir: str,
    traj_name: str,
    demo_actions: np.ndarray,
    demo_ee_states: np.ndarray,
    demo_gripper_states: np.ndarray,
    demo_obs: np.ndarray,
    demo_rewards: np.ndarray,
    fps: int = 10,
    filter_static_actions: bool = False,
):
    """
    Save one trajectory as aggregated files:
      - states.npy : (T, ee_dim + gripper_dim)
      - actions.npy: (T, action_dim)
      - rewards.npy: (T,)
      - obs_video.mp4: video of frames
      - meta.json: per-traj metadata
    """
    create_directory(traj_save_dir)

    T = demo_obs.shape[0]
    assert demo_actions.shape[0] == T, "actions len != obs len"
    assert demo_ee_states.shape[0] == T, "ee_states len != obs len"
    assert demo_gripper_states.shape[0] == T, "gripper_states len != obs len"
    assert demo_rewards.shape[0] == T, "rewards len != obs len"

    states = []
    actions = []
    rewards = []
    frames = []

    num_static = 0

    for t in range(T):
        #! Skip transitions with static actions
        action = demo_actions[t]
        prev_action = demo_actions[-1] if len(demo_actions) > 0 else None
        if is_static_action(action, prev_action):
            num_static += 1
            if filter_static_actions:
                print(f"Skipping static action: {action}")
                continue

        state = np.concatenate([demo_ee_states[t], demo_gripper_states[t]], axis=-1)
        reward = demo_rewards[t]
        frame = _normalize_frame(demo_obs[t])

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        frames.append(frame)

    states_np = np.array(states)
    states_path = os.path.join(traj_save_dir, "states.npy")
    np.save(states_path, states_np)

    actions_np = np.array(actions)
    actions_path = os.path.join(traj_save_dir, "actions.npy")
    np.save(actions_path, actions_np)

    rewards_np = np.array(rewards)
    rewards_path = os.path.join(traj_save_dir, "rewards.npy")
    np.save(rewards_path, rewards_np)

    video_path = os.path.join(traj_save_dir, "obs_video.mp4")
    # mediapy.write_video(video_path, frames, fps=fps, codec='libx264')
    mediapy.write_video(video_path, frames, fps=fps)

    # per-traj meta
    meta = {
        "traj_name": traj_name,
        "num_all_steps": int(T),
        "num_static_actions_steps": num_static,
        "states_shape": list(states_np.shape),
        "states_dtype": str(states_np.dtype),
        "actions_shape": list(actions_np.shape),
        "actions_dtype": str(actions_np.dtype),
        "rewards_shape": list(rewards_np.shape),
        "rewards_dtype": str(rewards_np.dtype),
        "video_file": os.path.basename(video_path),
        "video_fps": int(fps),
    }
    meta_path = os.path.join(traj_save_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def gen_single_suite_dataset(suite_name, benchmark_dict, args):
    task_suite = benchmark_dict[suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    save_root = args.save_dir
    suite_path = suite_path = os.path.join(args.raw_data_dir, suite_name)
    filter_static_actions = args.filter_static_actions
    if args.filter_static_actions:
        save_root += "_no_noops"
        no_noops_suite_name = suite_name + "_no_noops"
        if os.path.exists(os.path.join(args.raw_data_dir, no_noops_suite_name)):
            filter_static_actions = False  # no need to filter, just use the no_noops suite
            suite_path = os.path.join(args.raw_data_dir, no_noops_suite_name)

    # breakpoint()

    save_dir = os.path.join(save_root, suite_name)
    # meta_info_save_dir = os.path.join(args.meta_info_save_dir, suite_name)
    create_directory(save_dir)
    # create_directory(meta_info_save_dir)

    # suite-level meta (min/max across all demos)
    meta_info = dict()
    meta_info["state_01"] = [float("inf")] * 8  # 6 + 2
    meta_info["state_99"] = [float("-inf")] * 8

    # travel tasks in this suite
    for task_id in tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        logger.info(f"Processing {suite_name}-{task.name}")

        task_data_path = os.path.join(
            suite_path, f"{task.name}_demo.hdf5"
        )
        if not os.path.exists(task_data_path):
            logger.warning(f"Task data file not found: {task_data_path}, skip")
            continue

        with h5py.File(task_data_path, "r") as task_data_file:
            task_data = task_data_file["data"]
            task_save_dir = os.path.join(save_dir, task.name)
            create_directory(task_save_dir)

            # travel demos in this task
            for i in range(100):
                demo_key = f"demo_{i}"
                if demo_key not in task_data:
                    continue

                demo_data = task_data[demo_key]
                demo_ee_states = demo_data["obs"]["ee_states"][()]  # (T, 6)
                demo_gripper_states = demo_data["obs"]["gripper_states"][()]  # (T, 2)
                demo_actions = demo_data["actions"][()]  # (T, 7)
                demo_obs = demo_data["obs"]["agentview_rgb"][()]  # (T, H, W, C)
                demo_rewards = demo_data["rewards"][()]  # (T,)

                # update suite-level meta_info
                ee_state_dim = demo_ee_states.shape[-1]
                gripper_state_dim = demo_gripper_states.shape[-1]
                assert ee_state_dim + gripper_state_dim == 8
                for d in range(ee_state_dim):
                    meta_info["state_01"][d] = min(
                        meta_info["state_01"][d], float(np.min(demo_ee_states[:, d]))
                    )
                    meta_info["state_99"][d] = max(
                        meta_info["state_99"][d], float(np.max(demo_ee_states[:, d]))
                    )
                for d in range(gripper_state_dim):
                    meta_info["state_01"][d + ee_state_dim] = min(
                        meta_info["state_01"][d + ee_state_dim],
                        float(np.min(demo_gripper_states[:, d])),
                    )
                    meta_info["state_99"][d + ee_state_dim] = max(
                        meta_info["state_99"][d + ee_state_dim],
                        float(np.max(demo_gripper_states[:, d])),
                    )

                # prepare per-traj save dir
                traj_save_dir = os.path.join(task_save_dir, f"traj_{i}")
                create_directory(traj_save_dir)

                # save aggregated files for this trajectory
                save_traj_aggregated(
                    traj_save_dir=traj_save_dir,
                    traj_name=f"{suite_name}/{task.name}/traj_{i}",
                    demo_actions=demo_actions,
                    demo_ee_states=demo_ee_states,
                    demo_gripper_states=demo_gripper_states,
                    demo_obs=demo_obs,
                    demo_rewards=demo_rewards,
                    fps=args.fps,
                    filter_static_actions=filter_static_actions,
                )

                logger.info(f"Saved aggregated traj: {traj_save_dir}")
                # breakpoint()

    # save suite-level meta info
    # meta_info_filepath = os.path.join(meta_info_save_dir, "stat.json")
    # with open(meta_info_filepath, "w") as f:
    #     json.dump(meta_info, f, indent=2)
    # logger.info(f"Saved suite meta info to: {meta_info_filepath}")


def main(args):
    print(f"Regenerating {args.libero_task_suites} dataset!")
    benchmark_dict = benchmark.get_benchmark_dict()
    suite_names = args.libero_task_suites
    for suite_name in suite_names:
        logger.info(f"Processing suite: {suite_name}")
        gen_single_suite_dataset(suite_name, benchmark_dict, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_dir",
        "-r",
        type=str,
        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets",
        default="/path/to/LIBERO",
    )
    parser.add_argument(
        "--libero_task_suites",
        "-t",
        type=str,
        nargs="+",
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
        help="LIBERO task suites. Example: -t libero_spatial libero_object",
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        "-s",
        type=str,
        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_static_action",
        default="/path/to/libero_regen/dataset",
    )
    parser.add_argument(
        "--meta_info_save_dir",
        "-ms",
        type=str,
        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_static_action",
        default="/path/to/libero_regen/dataset_meta_info",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for output video (default 10).",
    )
    parser.add_argument(
        "--filter_static_actions",
        "-f",
        action="store_true",
        help="Whether to filter static actions. Default: False",
    )
    args = parser.parse_args()

    main(args)
