import glob
import io
import json
import os
import random
from typing import Any, Dict, Iterable, List, Tuple

import mediapy
import numpy as np

# from torchvision import transforms as T
import torch
import torch.distributed as dist
import webdataset as wds
from diffusers.models import AutoencoderKLTemporalDecoder
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm


class DatasetLiberoOnline(Dataset):
    def __init__(
        self,
        args,
        device: str = "cuda",
        mode: str = "val",
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.device = device

        if args.is_img_pregenerated:
            print("Pregenerate image latents!")
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
                args.svd_model_path, subfolder="vae"
            ).to(device)

        if mode == "train":
            sample_json_path = os.path.join(
                args.dataset_sample_json_dir, "train_sample_info.json"
            )
        elif mode == "val":
            sample_json_path = os.path.join(
                args.dataset_sample_json_dir, "val_sample_info.json"
            )
        else:
            raise ValueError("mode must be train or val.")

        with open(sample_json_path, "r") as f:
            self.sample_info: List = json.load(f)

    def __len__(self):
        return len(self.sample_info)

    def preprocess_img(self, img: np.ndarray):
        img: torch.Tensor = (
            torch.tensor(img).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
        )  # [T, 3, H, W]
        img = torch.nn.functional.interpolate(
            img,
            size=self.args.img_resizes,
            mode="bilinear",
            align_corners=False,
        )
        return img

    def encode_img_to_latent(self, img: torch.Tensor):
        """
        img: (T, 3, h, w)
        return: (T, 4, 32, 32)
        """
        img = img.to(self.device)  # [T, 3, H, W]
        with torch.no_grad():
            latent: torch.Tensor = (
                self.vae.encode(img)
                .latent_dist.sample()
                .mul_(self.vae.config.scaling_factor)
                .cpu()
            )  # [T, 4, 24, 40]
        return latent

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_task_instruction(self, traj_path: str) -> str:
        # e.g. xxx/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/traj_0
        suite_name: str = traj_path.split("/")[-3]
        task_name: str = traj_path.split("/")[-2]
        if suite_name in ["libero_10", "libero_90"]:
            task_instruction = " ".join(task_name.split("_")[2:]).capitalize()
        else:
            task_instruction = task_name.replace("_", " ").capitalize()
        return task_instruction

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_info = self.sample_info[index]
        traj_path = sample_info["traj_path"]
        batch_item = dict()

        timestamps = sample_info["timestamps"]
        ts = np.array(timestamps)

        text: str = self.get_task_instruction(traj_path)
        batch_item["text"] = text  # string

        action_np = np.load(os.path.join(traj_path, "actions.npy"))[ts]
        # normalize action
        action_np = self.normalize_bound(action_np, 0, 1)
        action = torch.tensor(action_np)  # [T, C_a = 6 + 1]
        batch_item["action"] = action

        state_np = np.load(os.path.join(traj_path, "states.npy"))[ts]
        # todo: normalize state with state_01 and state_99
        state = torch.tensor(state_np)  # [T, C_s = 6 + 2]
        batch_item["state"] = state

        reward_np = np.load(os.path.join(traj_path, "rewards.npy"))[ts]
        reward = torch.tensor(reward_np)  # [T]
        batch_item["reward"] = reward

        obs_np = mediapy.read_video(os.path.join(traj_path, "obs_video.mp4"))[
            ts
        ]  # [T, H, W, 3]
        obs_img = self.preprocess_img(obs_np)  # [T, 3, H = 192, W = 320]
        batch_item["img"] = obs_img

        if self.args.is_img_pregenerated:
            obs_latent = self.encode_img_to_latent(obs_img)  # [T, 4, 24, 40]
            batch_item["latent"] = obs_latent  # [T, 4, 24, 40]

        return batch_item


class DatasetLiberoOnlineV2(IterableDataset):
    def __init__(
        self,
        shards_pattern: str,
        # stats_path: str,
        *,
        Ta: int = 8,  # action length = T_futr, 8
        To: int = 8,  # observation length = T_hist, 4
        stride: int = 1,
        action_dim: int = 7,
        image_size: Tuple[int, int] = (224, 224),
        not_repeat=True,
        episode_buf_size: int = 30,
        sample_buf_size: int = 10000,
        max_windows_per_episode: int = None,
        window_selection: str = "sliding",
    ):
        super().__init__()
        self.Ta, self.To, self.stride = Ta, To, stride
        self.context_length = self.Ta + self.To
        self.image_size = list(image_size)
        self.max_windows_per_episode = max_windows_per_episode
        self.window_selection = str(window_selection or "sliding").lower()

        try:
            shards: List[str] = sorted(glob.glob(shards_pattern, recursive=True))
        except:
            shards = []
            for pattern in shards_pattern:
                shards += sorted(glob.glob(pattern, recursive=True))
            shards_pattern = " ".join(shards_pattern)
        print(
            f"[WM Dataset] Found {len(shards)} shards from the pattern '{shards_pattern}'."
        )
        if not shards:
            raise FileNotFoundError(
                f"[WM Dataset] The pattern '{shards_pattern}' did not match any .tar files."
            )

        # ---------- 2. 读取归一化常量 ----------
        # stats = json.load(open(stats_path, "r"))
        # if "aloha" in shards_pattern:
        #     try:
        #         self.q01 = np.asarray(stats["action"]["min"], np.float32)
        #         self.q99 = np.asarray(stats["action"]["max"], np.float32)
        #     except:
        #         key = list(stats.keys())[0]
        #         self.q01 = np.asarray(stats[key]["action"]["min"], np.float32)
        #         self.q99 = np.asarray(stats[key]["action"]["max"], np.float32)
        #     self.finish_step_shift = -1
        # else:
        #     self.finish_step_shift = 0
        #     try:
        #         self.q01 = np.asarray(stats["action"]["q01"], np.float32)
        #         self.q99 = np.asarray(stats["action"]["q99"], np.float32)
        #     except:
        #         key = list(stats.keys())[0]
        #         self.q01 = np.asarray(stats[key]["action"]["q01"], np.float32)
        #         self.q99 = np.asarray(stats[key]["action"]["q99"], np.float32)
        self.finish_step_shift = 0

        # # ---------- 3. 分布式 rank ----------
        # world_size = dist.get_world_size() if dist.is_initialized() else 1

        # # ---------- 4. 估算 epoch size ----------
        # estimated_windows_per_shard = episode_per_shard * 350
        # self.epoch_size = estimated_windows_per_shard * len(shards) // world_size

        # ---------- 5. 构建 WebDataset Pipeline ----------
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        use_resample = ((world_size * 4) >= len(shards)) and (
            not not_repeat
        )  # * 4 is number of workers
        print(
            f"World size: {world_size}, Use resample: {use_resample}, Shard num: {len(shards)}"
        )
        seed = random.randint(0, 10000)
        self.ds = wds.DataPipeline(
            (
                wds.ResampledShards(shards, seed=seed)
                if use_resample
                else wds.SimpleShardList(shards, seed=seed)
            ),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(
                episode_buf_size, initial=episode_buf_size
            ),  # 保留 episode 级 shuffle（小 buffer）
            wds.to_tuple(
                "video.npy", "env_video.npy", "action.npy", "env_dones.npy", "meta.json"
            ),
            self._split_to_windows,
            # wds.shuffle(sample_buf_size, initial=sample_buf_size),  #! 去掉 sample 级 shuffle
        )

    def _preprocess_img(self, img: np.ndarray):
        img: torch.Tensor = (
            torch.tensor(img).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
        )  # [T, 3, H, W]
        img = torch.nn.functional.interpolate(
            img,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return img

    def _select_window_starts(self, finish_step: int) -> List[int]:
        finish_step = max(0, int(finish_step))
        max_start_exclusive = max(finish_step - self.Ta + 1, 1)
        starts = list(range(0, max_start_exclusive, self.stride))
        if not starts and finish_step > 0:
            starts = [0]
        if self.max_windows_per_episode is None or len(starts) <= self.max_windows_per_episode:
            return starts

        if self.window_selection == "head":
            return starts[: self.max_windows_per_episode]

        if self.window_selection == "uniform":
            indices = np.linspace(
                0,
                len(starts) - 1,
                num=self.max_windows_per_episode,
                dtype=np.int64,
            )
            selected = []
            seen = set()
            for idx in indices.tolist():
                idx = int(idx)
                if idx in seen:
                    continue
                seen.add(idx)
                selected.append(starts[idx])
            if len(selected) < self.max_windows_per_episode:
                for start in starts:
                    if start in selected:
                        continue
                    selected.append(start)
                    if len(selected) >= self.max_windows_per_episode:
                        break
            return selected

        return starts[: self.max_windows_per_episode]

    def _slice_with_edge_padding(
        self, array: np.ndarray, start: int, end: int
    ) -> np.ndarray:
        target_len = int(end - start)
        if target_len <= 0:
            return array[:0]
        if array.shape[0] <= 0:
            raise ValueError("Cannot build a window from an empty episode array.")

        left_pad = max(0, -int(start))
        right_pad = max(0, int(end) - int(array.shape[0]))
        src_start = max(0, int(start))
        src_end = min(int(array.shape[0]), int(end))

        window = array[src_start:src_end]
        if left_pad > 0:
            window = np.concatenate(
                [np.repeat(array[0:1], left_pad, axis=0), window],
                axis=0,
            )
        if right_pad > 0:
            window = np.concatenate(
                [window, np.repeat(array[-1:], right_pad, axis=0)],
                axis=0,
            )

        if window.shape[0] != target_len:
            raise ValueError(
                f"Window length mismatch: expected {target_len}, got {window.shape[0]}."
            )
        return window

    #! key imple
    def _split_to_windows(
        self, src: Iterable[Tuple[bytes, bytes, bytes, bytes]]
    ) -> Iterable[Dict]:
        for v_bytes, ev_bytes, a_bytes, ed_bytes, m_bytes in src:
            video_np = np.load(
                io.BytesIO(v_bytes), allow_pickle=False
            )  # (all_frames, H, W, c)
            env_video_np = np.load(
                io.BytesIO(ev_bytes), allow_pickle=False
            )  # (all_frames, H, W, c)
            action_np = np.load(
                io.BytesIO(a_bytes), allow_pickle=False
            )  # (all_frames, action_dim)
            env_dones_np = np.load(io.BytesIO(ed_bytes), allow_pickle=False)
            meta = json.loads(m_bytes.decode())

            available_steps = min(
                int(video_np.shape[0]),
                int(env_video_np.shape[0]),
                int(action_np.shape[0]),
                int(env_dones_np.shape[0]),
            )
            if available_steps <= 0:
                continue

            video_np = video_np[:available_steps]
            env_video_np = env_video_np[:available_steps]
            action_np = action_np[:available_steps]
            env_dones_np = env_dones_np[:available_steps]

            finish_step = int(meta.get("finish_step", available_steps)) + self.finish_step_shift
            finish_step = max(0, min(finish_step, available_steps))
            if finish_step <= 0:
                continue

            task_description = meta.get("task_description")
            for start in self._select_window_starts(finish_step):
                vs, ve = start - self.To + 1, start + self.Ta + 1  # T = T_a + T_o
                vid_window = self._slice_with_edge_padding(video_np, vs, ve)
                env_vid_window = self._slice_with_edge_padding(env_video_np, vs, ve)
                act_window = self._slice_with_edge_padding(action_np, vs, ve)
                env_done_window = self._slice_with_edge_padding(env_dones_np, vs, ve)

                # img = torch.from_numpy(vid_window).float() / 127.5 - 1  # [T, H, W, c]
                # img = img.permute(0, 3, 1, 2)  # (T, c, H, W)

                # env_img = torch.from_numpy(env_vid_window).float() / 127.5 - 1
                # env_img = env_img.permute(0, 3, 1, 2)  # (T, c, H, W)

                img = self._preprocess_img(vid_window)  # [T, c, H, W]
                env_img = self._preprocess_img(env_vid_window)  # [T, c, H, W]

                # act_np = action_np[start + 1 : start + self.Ta + 1]  # [T_a, action_dim]
                # # action = 2 * ((act_np - self.q01) / (self.q99 - self.q01)) - 1
                # action = torch.from_numpy(act_np).float()
                action = torch.from_numpy(act_window).float()  # [T, action_dim]

                reward = torch.from_numpy(env_done_window).float()  # [T]

                debug_meta = {
                    "fpath": meta.get("fpath", ""),
                    "unique_id": meta.get("unique_id", ""),
                    "episode_name": meta.get("episode_name", ""),
                    "window_start": vs,
                    "window_end": ve - 1,
                }

                self.image_size[0] = img.shape[2]  # H
                self.image_size[1] = img.shape[3]  # W
                yield {
                    "img": env_img,  # gt: env img video
                    "action": action,
                    "reward": reward,  # env done
                    "text": task_description,
                    "wm_img": img,  # wm img video
                    # "fps": 30,
                    # "num_frames": img.shape[0],
                    # "height": self.image_size[0],
                    # "width": self.image_size[1],
                    # "meta": debug_meta,
                }

    def __iter__(self):
        return iter(self.ds)
