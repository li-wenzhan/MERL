import argparse
import json
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm


def prepare_dataset_sample(
    dataset_root_path: str,
    dataset_names: List[str],
    save_path: str,
    num_history: int = 6,
    num_frames: int = 5,
    window_sample: int = 1,
    down_sample: int = 2,
    positive_ratio: float = 0.4,
    upsample_positive: bool = False,
    val_split: float = 0.2,
    seed: int = 42,
    min_success_gap: int = 1,
    max_pos_per_traj: Optional[int] = None,
    max_total_samples: Optional[int] = None,
):
    """
    生成 sample_info 并按 positive_ratio 调整 reward=1 的占比。

    改进点:
      - 使用 set 加速去重；
      - 更稳健地识别 reward==1（先 cast int）；
      - 增加 min_success_gap / max_pos_per_traj 控制正样本增长；
      - 当可用样本少于目标时，会自动缩减目标总量并打印警告；
      - 可选 max_total_samples 用于固定输出样本总数（将对正/负样本进行采样/上采样）。
    """
    assert 0.0 <= positive_ratio <= 1.0, "positive_ratio must be in [0,1]"
    assert 0.0 <= val_split < 1.0, "val_split must be in [0,1)"

    random.seed(seed)
    np.random.seed(seed)

    sample_infos: List[Dict] = []
    seen_keys: Set[Tuple[str, Tuple[int, ...]]] = set()  # 加速去重
    required_ts_len = num_history + num_frames

    for dataset_name in dataset_names:
        print(f"Processing: {dataset_name}")
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        if not os.path.isdir(dataset_path):
            print(f"Warning: dataset does not exists, skip -> {dataset_path}")
            continue

        for task_name in tqdm(os.listdir(dataset_path)):
            task_path = os.path.join(dataset_path, task_name)
            if not os.path.isdir(task_path):
                continue

            for traj_name in os.listdir(task_path):
                traj_path = os.path.join(task_path, traj_name)
                if not os.path.isdir(traj_path):
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

                traj_length = int(action_npy.shape[0])
                if traj_length < required_ts_len:
                    # 长度不足
                    continue

                # ---------- 1) window_sample 常规采样 ----------
                for t in range(0, traj_length, window_sample):
                    timestamps = []
                    for i in range(t, traj_length, down_sample):
                        timestamps.append(int(i))
                        if len(timestamps) >= required_ts_len:
                            timestamps = timestamps[:required_ts_len]
                            key = (traj_path, tuple(timestamps))
                            if key not in seen_keys:
                                # label 用最后一帧的 reward 判定（先cast为int以避免 numpy types）
                                label = int(int(reward_npy[timestamps[-1]]) == 1)
                                sample_infos.append(
                                    {
                                        "traj_path": traj_path,
                                        "timestamps": [int(x) for x in timestamps],
                                        "label": int(label),
                                    }
                                )
                                seen_keys.add(key)
                            break

                # ---------- 2) 以 reward=1 结尾的片段作为正样本 ----------
                # 将 reward 转为整型 boolean 序列（稳健）
                reward_int = np.asarray(reward_npy).astype(int)
                success_indices = np.flatnonzero(reward_int == 1)  # 升序

                # 选取间隔足够的 success indices（避免几帧连续的 success 导致过多重复样本）
                chosen_success_ts = []
                last_chosen = -999999
                for s_idx in success_indices:
                    if (s_idx - last_chosen) >= min_success_gap:
                        chosen_success_ts.append(int(s_idx))
                        last_chosen = int(s_idx)
                    else:
                        # 跳过这个 success，太接近上一个被选中的 success
                        continue

                # 如果限制了每轨迹正样本数，裁剪（保留最晚的那些 success）
                if (
                    max_pos_per_traj is not None
                    and len(chosen_success_ts) > max_pos_per_traj
                ):
                    # 保留靠后的 success（通常靠后成功更有代表性）
                    chosen_success_ts = chosen_success_ts[-max_pos_per_traj:]

                for success_t in chosen_success_ts:
                    timestamps = []
                    for i in range(success_t, -1, -down_sample):
                        timestamps.append(int(i))
                        if len(timestamps) >= required_ts_len:
                            timestamps = timestamps[:required_ts_len][::-1]
                            key = (traj_path, tuple(timestamps))
                            if key not in seen_keys:
                                sample_infos.append(
                                    {
                                        "traj_path": traj_path,
                                        "timestamps": [int(x) for x in timestamps],
                                        "label": 1,
                                    }
                                )
                                seen_keys.add(key)
                            break

    # 按 label 分割
    pos_samples = [s for s in sample_infos if int(s["label"]) == 1]
    neg_samples = [s for s in sample_infos if int(s["label"]) == 0]

    available_pos = len(pos_samples)
    available_neg = len(neg_samples)
    total_candidates = len(sample_infos)
    if total_candidates == 0:
        raise RuntimeError("No samples found. Check dataset paths and parameters.")

    # 如果用户需要固定总样本数，使用 max_total_samples；否则默认用候选总数
    if max_total_samples is not None:
        if max_total_samples <= 0:
            raise ValueError("max_total_samples must be > 0")
        target_total = int(max_total_samples)
    else:
        target_total = total_candidates

    # 期望正样本数
    desired_pos = int(round(target_total * positive_ratio))

    # 若 available 不足以满足目标，则按可用量调整 target_total
    if desired_pos > available_pos:
        if upsample_positive and available_pos > 0:
            # 允许上采样 positive：保持 target_total 不变，后面会用 random.choices 补齐
            pass
        else:
            # 不能上采样或没有正样本 -> 缩小 target_total
            print(
                f"Warning: desired_pos ({desired_pos}) > available_pos ({available_pos}) and upsample_positive is False."
                " Reducing total target to available_pos + available_neg."
            )
            target_total = available_pos + available_neg
            desired_pos = min(available_pos, int(round(target_total * positive_ratio)))

    # 重新计算 target_neg
    desired_neg = target_total - desired_pos
    # 若负样本不够，进一步缩减 target_total（不对负样本上采样）
    if desired_neg > available_neg:
        print(
            f"Warning: desired_neg ({desired_neg}) > available_neg ({available_neg})."
            " Reducing total target to available_pos + available_neg."
        )
        target_total = available_pos + available_neg
        desired_pos = min(desired_pos, available_pos)
        desired_neg = target_total - desired_pos

    # 选择正样本
    if desired_pos == 0:
        chosen_pos: List[Dict] = []
    else:
        if desired_pos <= available_pos:
            chosen_pos = random.sample(pos_samples, desired_pos)
        else:
            # desired_pos > available_pos 且允许上采样
            chosen_pos = pos_samples.copy()
            needed = desired_pos - available_pos
            chosen_pos += random.choices(pos_samples, k=needed)

    # 选择负样本（不做上采样）
    if desired_neg == 0:
        chosen_neg: List[Dict] = []
    else:
        chosen_neg = random.sample(neg_samples, desired_neg)

    balanced_samples = chosen_pos + chosen_neg
    random.shuffle(balanced_samples)

    # 如果 max_total_samples 被指定但我们最终样本少于它，且允许上采样负样本（目前不允许），可以在这里做扩充
    # Split train / val
    split_idx = int(len(balanced_samples) * (1.0 - val_split))
    train_sample_info = balanced_samples[:split_idx]
    val_sample_info = balanced_samples[split_idx:]

    # 统计信息
    final_pos = sum(1 for s in balanced_samples if int(s["label"]) == 1)
    final_neg = len(balanced_samples) - final_pos
    actual_ratio = final_pos / max(len(balanced_samples), 1)
    print(
        f"Candidates total: {total_candidates}, pos_available: {available_pos}, neg_available: {available_neg}"
    )
    print(
        f"Target total: {target_total}, desired_pos: {desired_pos}, desired_neg: {desired_neg}"
    )
    print(
        f"Selected total: {len(balanced_samples)}, pos_selected: {final_pos}, neg_selected: {final_neg}, actual_pos_ratio: {actual_ratio:.4f}"
    )
    print(f"Train/Val sizes: {len(train_sample_info)}/{len(val_sample_info)}")

    # 保存
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "train_sample_info.json"), "w") as f:
        json.dump(train_sample_info, f)
    with open(os.path.join(save_path, "val_sample_info.json"), "w") as f:
        json.dump(val_sample_info, f)

    print(f"Saved train/val sample info to -> {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset samples with adjustable positive (reward=1) ratio."
    )
    parser.add_argument(
        "--dataset-root-path",
        type=str,
        required=True,
        help="Root path of the original dataset.",
    )
    parser.add_argument(
        "--dataset-names",
        type=str,
        nargs="+",
        default=["libero_goal"],
        help="List of dataset directories under root.",
    )
    parser.add_argument(
        "--save-path", type=str, required=True, help="Where to save train/val jsons."
    )
    parser.add_argument(
        "--num-history", type=int, default=6, help="Number of history frames."
    )
    parser.add_argument(
        "--num-frames", type=int, default=5, help="Number of future frames to predict."
    )
    parser.add_argument(
        "--window-sample", type=int, default=1, help="Window sampling stride."
    )
    parser.add_argument(
        "--down-sample", type=int, default=2, help="Down-sample stride for timestamps."
    )
    parser.add_argument(
        "--positive-ratio",
        type=float,
        default=0.4,
        help="Desired proportion of reward=1 samples in final dataset (0..1).",
    )
    parser.add_argument(
        "--upsample-positive",
        action="store_true",
        help="If set, allow upsampling positives (with replacement) when positives < target.",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split ratio (0..1)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--min-success-gap",
        type=int,
        default=1,
        help="Minimum frame gap between two chosen success timestamps within the same trajectory.",
    )
    parser.add_argument(
        "--max-pos-per-traj",
        type=int,
        default=None,
        help="Maximum number of positive samples allowed to come from the same trajectory (None=unlimited).",
    )
    parser.add_argument(
        "--max-total-samples",
        type=int,
        default=None,
        help="If set, aim to output exactly this many total samples (will sample/upsample to reach it if possible).",
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
        positive_ratio=args.positive_ratio,
        upsample_positive=args.upsample_positive,
        val_split=args.val_split,
        seed=args.seed,
        min_success_gap=args.min_success_gap,
        max_pos_per_traj=args.max_pos_per_traj,
        max_total_samples=args.max_total_samples,
    )


if __name__ == "__main__":
    main()
