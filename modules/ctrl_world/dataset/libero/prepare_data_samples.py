import argparse
import json
import os
import random

import mediapy
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from torch.utils.data import Dataset
from tqdm import tqdm


def prepare_dataset_sample(
    dataset_root_path,
    dataset_names,
    save_path,
    num_history=6,
    num_frames=5,
    window_sample=1,
    down_sample=2,
):
    sample_infos = []
    required_ts_len = num_history + num_frames
    reward_1_count = 0

    for dataset_name in dataset_names:
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        if not os.path.isdir(dataset_path):
            print(f"Warning: dataset does not exists, skip -> {dataset_path}")
            continue

        for task_name in os.listdir(dataset_path):
            task_path = os.path.join(dataset_path, task_name)
            if not os.path.isdir(task_path):
                print(f"Warning: 非任务目录，跳过 -> {task_path}")
                continue

            for traj_name in os.listdir(task_path):
                traj_path = os.path.join(task_path, traj_name)
                if not os.path.isdir(traj_path):
                    print(f"Warning: 非轨迹目录，跳过 -> {traj_path}")
                    continue

                action_path = os.path.join(traj_path, "actions.npy")
                reward_path = os.path.join(traj_path, "rewards.npy")
                try:
                    action_npy = np.load(action_path)
                    reward_npy = np.load(reward_path)
                except Exception as e:
                    print(
                        f"Warning: 读取文件失败，跳过轨迹 -> {traj_path}，原因：{str(e)}"
                    )
                    continue

                traj_length = action_npy.shape[0]
                # 跳过长度不足的轨迹
                if traj_length < required_ts_len:
                    print(
                        f"Warning: 轨迹长度不足{required_ts_len}，跳过 -> {traj_path}"
                    )
                    continue

                # -------------------------- 1. 间隔window_sample采样 --------------------------
                for t in range(0, traj_length, window_sample):
                    timestamps = []
                    # 从t开始，按down_sample间隔收集时间戳，直到满足长度要求
                    for i in range(t, traj_length, down_sample):
                        timestamps.append(i)
                        if len(timestamps) >= required_ts_len:
                            # 截断到正好required_ts_len个（避免超出）
                            timestamps = timestamps[:required_ts_len]
                            sample_info = {
                                "traj_path": traj_path,
                                "timestamps": timestamps,
                            }
                            sample_infos.append(sample_info)
                            break

                # -------------------------- 2. 强制添加reward=1的最后一帧采样 --------------------------
                if int(reward_npy[-1]) == 1:
                    reward_1_count += 1
                    timestamps = []
                    # 从最后一帧（traj_length-1）倒序收集时间戳
                    for i in range(traj_length - 1, -1, -down_sample):
                        timestamps.append(i)
                        if len(timestamps) >= required_ts_len:
                            # 截断+倒序，恢复时间正序，确保长度正确
                            timestamps = timestamps[:required_ts_len][::-1]
                            # 检查是否与已添加的采样重复，避免冗余
                            is_duplicate = any(
                                info["traj_path"] == traj_path
                                and info["timestamps"] == timestamps
                                for info in sample_infos
                            )
                            if not is_duplicate:
                                sample_info = {
                                    "traj_path": traj_path,
                                    "timestamps": timestamps,
                                }
                                sample_infos.append(sample_info)
                            break

    random.seed(42)
    random.shuffle(sample_infos)
    split_idx = int(len(sample_infos) * 0.8)
    train_sample_info = sample_infos[:split_idx]
    val_sample_info = sample_infos[split_idx:]

    print(
        f"Length of all samples: {len(sample_infos)}, reward=1 count: {reward_1_count}"
    )

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "train_sample_info.json"), "w") as f:
        json.dump(train_sample_info, f)
    with open(os.path.join(save_path, "val_sample_info.json"), "w") as f:
        json.dump(val_sample_info, f)

    print(f"Dataset sample info generation finished! Saved to -> {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset samples with specified parameters"
    )
    parser.add_argument(
        "--dataset-root-path",
        type=str,
        default="/path/to/libero_regen/dataset",
        help="Root path of the original dataset (default: /path/to/libero_regen/dataset)",
    )
    parser.add_argument(
        "--dataset-names",
        type=str,
        nargs="+",
        default=["libero_goal", "libero_10", "libero_spatial", "libero_object"],
        help="List of dataset names to process (default: ['libero_goal'])",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/path/to/Ctrl-World/dataset/libero",
        help="Path to save the processed dataset (default: /path/to/Ctrl-World/dataset/libero)",
    )
    parser.add_argument(
        "--num-history",
        type=int,
        default=6,
        help="Number of history frames (default: 6)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Total number of frames to sample (default: 5)",
    )
    parser.add_argument(
        "--window-sample",
        type=int,
        default=1,
        help="Window sample rate for frames (default: 1)",
    )
    parser.add_argument(
        "--down-sample",
        type=int,
        default=2,
        help="Down-sample rate for frames (default: 2); make fps 10 -> 5",
    )
    args = parser.parse_args()

    prepare_dataset_sample(
        dataset_root_path=args.dataset_root_path,
        dataset_names=args.dataset_names,
        save_path=args.save_path,
        num_history=args.num_history,
        num_frames=args.num_frames,
        window_sample=args.window_sample,
        down_sample=args.down_sample,
    )


if __name__ == "__main__":
    main()
