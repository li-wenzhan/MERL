import argparse
import json
import os
from typing import Any, Dict, List

import mediapy

# import imageio
import numpy as np

# from torchvision import transforms as T
import torch
from diffusers.models import AutoencoderKLTemporalDecoder
from torch.utils.data import DataLoader, Dataset

# from omegaconf import OmegaConf
from tqdm import tqdm


class DatasetLibero(Dataset):
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
        batch_item["text"] = text

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
