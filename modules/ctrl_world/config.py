import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch


_CTRL_WORLD_ROOT = Path(__file__).resolve().parent
_CTRL_WORLD_SAMPLE_JSON_DIR = str(_CTRL_WORLD_ROOT / "dataset" / "libero")
_CTRL_WORLD_META_INFO_DIR = str(_CTRL_WORLD_ROOT / "dataset_meta_info")
_ACTION_ADAPTER_PATH = str(
    _CTRL_WORLD_ROOT / "models" / "action_adapter" / "model2_15_9.pth"
)

# ! Reader checklist:
# ! 1) Set svd_model_path / clip_model_path / ckpt_path / pi_ckpt to your local model directories.
# ! 2) Set dataset_sample_json_dir and dataset_meta_info_path to the dataset you prepared.
# ! 3) This file is mainly for Ctrl-World local rollout / inference utilities, not the online MERL Hydra config.


@dataclass
class wm_args:
    ########################### training args ##############################
    # model paths
    svd_model_path = "/path/to/models/stable-video-diffusion-img2vid"  # ! todo: set to your local SVD backbone directory
    clip_model_path = "/path/to/models/clip-vit-base-patch32"  # ! todo: set to your local CLIP backbone directory
    # Train a new model
    # ckpt_path = None
    # Load checkpoint
    ckpt_path = "/path/to/Ctrl_World/checkpoint.pt"  # ! todo: set to your local Ctrl-World checkpoint for inference / warm-start
    pi_ckpt = "/path/to/openpi/checkpoints/pi05_droid"  # ! todo: set only if you use OpenPI interaction rollout
    load_from_ckpt = False

    # dataset parameters
    # raw data
    num_views = 1  # for libero: 1, for droid: 3
    is_img_pregenerated = True  #! pregenerate images in dataset processes
    dataset_root_path = "dataset_example"
    dataset_names = "droid_subset"
    # meta info
    dataset_meta_info_path = _CTRL_WORLD_META_INFO_DIR  # ! repo-local default; override only if your dataset meta-info lives elsewhere
    #! for libero dataset sample
    dataset_sample_json_dir = _CTRL_WORLD_SAMPLE_JSON_DIR  # ! repo-local default; override only if your LIBERO sample-json lives elsewhere
    dataset_cfgs = dataset_names
    prob = [1.0]
    annotation_name = "annotation"  #'annotation_all_skip1'
    num_workers = 4
    down_sample = 2  # downsample 15hz to 5hz (3), downsample 10hz to 5hz
    skip_step = 1
    img_resizes = (192, 320)  #! preprocess image size

    # logs parameters
    debug = False
    tag = "libero_all_w_rm_1110"  #! "doird_subset"
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = tag
    wandb_project_name = "libero"
    videos_col = 3  #! each vid contains 3 slices

    # training parameters
    learning_rate = 5e-6  # 5e-6, 1e-5
    gradient_accumulation_steps = 2  # 1
    mixed_precision = "fp16"
    train_batch_size = 2  # 4
    shuffle = True
    num_train_epochs = 10  #! 100
    max_train_steps = 50000  #! 500000
    checkpointing_steps = 5000  #! 20000
    validation_steps = 2500  #! 1000
    max_grad_norm = 1.0
    # for val
    video_num = 3  #! 10

    ############################ model args ##############################

    # model parameters
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 2  # 7.5 #7.5 #7.5 #3.0
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 192
    # num history and num future predictions
    num_frames = 5
    num_history = 6
    action_dim = 7
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
    dtype = (
        torch.bfloat16
    )  # [torch.float32, torch.bfloat16] # during inference, we can use bfloat16 to accelerate the inference speed and save memory

    ########################### rollout args ############################
    # policy
    task_type: str = (
        "pickplace"  # choose from ['pickplace', 'towel_fold', 'wipe_table', 'tissue', 'close_laptop','tissue','drawer','stack']
    )
    gripper_max_dict = {
        "replay": 1.0,
        "pickplace": 0.75,
        "towel_fold": 0.95,
        "wipe_table": 0.95,
        "tissue": 0.97,
        "close_laptop": 0.95,
        "drawer": 0.75,
        "stack": 0.75,
    }
    ##############################################################################
    policy_type = "pi05"  # choose from ['pi05', 'pi0', 'pi0fast']
    action_adapter = _ACTION_ADAPTER_PATH  # ! todo: set if you use pi policy interaction rollout
    pred_step = 5  # predict 5 steps (1s) action each time
    policy_skip_step = 2  # horizon = (pred_step-1) * policy_skip_step
    interact_num = (
        12  # number of interactions (each interaction contains pred_step steps)
    )

    # wm
    data_stat_path = str(Path(_CTRL_WORLD_META_INFO_DIR) / "droid" / "stat.json")
    val_model_path = ckpt_path
    history_idx = [0, 0, -12, -9, -6, -3]

    # save
    save_dir = "synthetic_traj"

    # select different traj for different tasks
    def __post_init__(self):
        # Per-task gripper max
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.75)
        # Default task_name
        self.task_name = f"Rollouts_interact_pi"
        if self.task_type == "replay":
            self.task_name = "Rollouts_replay"

        # Configure per-task eval sets
        if self.task_type == "replay":
            self.val_dataset_dir = "dataset_example/droid_subset"
            self.val_id = [
                "899",
                "18599",
                "199",
            ]
            self.start_idx = [8, 14, 8] * len(self.val_id)
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_replay"

        elif self.task_type == "keyboard":
            self.val_dataset_dir = "dataset_example/droid_subset"
            self.val_id = ["1799"]
            self.start_idx = [23] * len(self.val_id)
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_keyboard"

        # elif self.task_type == "keyboard2":
        #     self.val_dataset_dir = "/path/to/droid_svd_v2"
        #     self.val_id = ["1499"]*100
        #     self.start_idx = [8] * len(self.val_id) # 2599 8 #9499 10
        #     self.instruction = [""] * len(self.val_id)
        #     self.task_name = "Rollouts_keyboard_1499"
        #     self.ineraction_num = 7

        elif self.task_type == "pickplace":
            self.interact_num = 15
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ["0001", "0002", "0003"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = [
                "pick up the green block and place in plate",
                "pick up the green block and place in plate",
                "pick up the blue block and place in plate",
            ]

        elif self.task_type == "towel_fold":
            self.interact_num = 15
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ["0004", "0005"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["fold the towel"] * len(self.val_id)

        elif self.task_type == "wipe_table":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ["0006", "0007"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = [
                "move the towel from left to right",
                "move the towel from left to right",
            ]

        elif self.task_type == "tissue":
            self.interact_num = 10
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ["0008", "0009"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["pull one tissue out of the box"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "close_laptop":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ["0010", "0011"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "stack":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ["0012", "0013"]
            self.start_idx = [5] * len(self.val_id)
            self.instruction = ["stack the blue block on the red block"] * len(
                self.val_id
            )

        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
