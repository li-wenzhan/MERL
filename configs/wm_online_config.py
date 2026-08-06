from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CTRL_WORLD_SAMPLE_JSON_DIR = str(
    _REPO_ROOT / "modules" / "ctrl_world" / "dataset" / "libero"
)

# MERL memory patch:
# 1. Default full WM eval sample count is bounded for 100-step debug runs.
# 2. Scripts can still override this value through Hydra.
# ! Reader checklist:
# ! 1) Set svd_model_path / clip_model_path to your local backbone directories.
# ! 2) Set ckpt_path only if you want to warm-start WM from an existing checkpoint.
# ! 3) Set dataset_sample_json_dir to the LIBERO sample-json directory used by Ctrl-World.


@dataclass
class wm_args:
    ########################### training args ##############################
    # model paths
    svd_model_path: str = (
        "/mnt/afs/task3_2/L202500276_lwz/models/stable-video-diffusion-img2vid"  # ! todo: set to your local SVD backbone directory
    )
    clip_model_path: str = (
        "/mnt/afs/task3_2/L202500276_lwz/models/clip-vit-base-patch32"  # ! todo: set to your local CLIP backbone directory
    )
    ckpt_path = "/mnt/afs/task3_2/L202500276_lwz/models/ctrl_world_ckpts/checkpoint-10000.pt"  # ! todo: optional warm-start WM checkpoint; ignored when load_from_ckpt=False
    # ckpt_path: Optional[str] = (
    #     "/path/to/MeRL/world_model/global_step_100/world_model.pth"
    # )
    load_from_ckpt: bool = False

    # dataset parameters
    num_views: int = 1
    is_img_pregenerated: bool = False
    dataset_root_path: str = "dataset_example"
    dataset_names: str = "droid_subset"
    dataset_meta_info_path: str = "dataset_meta_info"
    dataset_sample_json_dir: str = (
        _CTRL_WORLD_SAMPLE_JSON_DIR  # ! repo-local default; override only if you store Ctrl-World sample-json elsewhere
    )
    dataset_cfgs: str = field(default_factory=lambda: "droid_subset")
    prob: List[float] = field(default_factory=lambda: [1.0])
    annotation_name: str = "annotation"
    num_workers: int = 4
    down_sample: int = 2
    skip_step: int = 1
    img_resizes: Tuple[int, int] = field(default_factory=lambda: (192, 320))

    # logs parameters
    debug: bool = False
    tag: str = (
        "libero_all_with_rm_online"  # ! todo: rename this run tag so checkpoints/logs are written to your own experiment directory
    )
    output_dir: str = field(init=False)
    wandb_run_name: str = field(init=False)
    wandb_project_name: str = "libero"
    videos_col: int = 3

    # training parameters
    learning_rate: float = 1e-5  # 5e-6
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    train_batch_size: int = 4  # 2
    train_real_batch_size: int = 4
    imag_train_batch_size: int = 4
    eval_real_batch_size: int = 4
    shuffle: bool = True
    num_train_epochs: int = 2
    max_train_steps: int = 50000
    checkpointing_steps: int = 5000
    validation_steps: int = 2500
    max_grad_norm: float = 1.0
    video_num: int = 3
    seed: int = 1024
    self_forcing_weight: float = 1.0

    ############################ model args ##############################
    motion_bucket_id: int = 127
    fps: int = 4
    guidance_scale: float = 2
    num_inference_steps: int = 30  # 缩短一点？加速推理，原 50, trade-off项
    decode_chunk_size: int = 8
    width: int = 320
    height: int = 192
    num_frames: int = 8
    num_history: int = 8
    action_dim: int = 7
    text_cond: bool = True
    frame_level_cond: bool = True
    his_cond_zero: bool = False
    reward_threshold: float = 0.5
    fixed_eval_enabled: bool = False
    fixed_eval_root: str = (
        ""  # ! todo: set by the launcher to ./tmp_files/wm_eval_shared/<dataset_name>
    )
    fixed_eval_global_steps: int = 0
    fixed_eval_mini_split: str = "wm_eval_fixed_mini"
    fixed_eval_full_split: str = "wm_eval_fixed_full"
    fixed_eval_mini_batch_size: int = 4
    fixed_eval_full_batch_size: int = 4
    fixed_eval_mini_samples: int = 40
    fixed_eval_full_samples: int = 100
    fixed_eval_mini_windows_per_episode: int = 2
    fixed_eval_full_windows_per_episode: int = 4
    fixed_eval_window_selection: str = "uniform"
    fixed_eval_full_interval: int = 10
    eval_num_inference_steps: int = 30
    max_saved_outer_checkpoints: int = 1
    dtype: str = "torch.bfloat16"

    def __post_init__(self):
        self.output_dir = f"model_ckpt/{self.tag}"
        self.wandb_run_name = self.tag

        if self.dtype == "torch.bfloat16":
            self._dtype_obj = torch.bfloat16
        elif self.dtype == "torch.float32":
            self._dtype_obj = torch.float32
        else:
            self._dtype_obj = torch.bfloat16

    @property
    def dtype_obj(self):
        return self._dtype_obj

    def to_dict(self):
        result = asdict(self)
        if "_dtype_obj" in result:
            del result["_dtype_obj"]
        return result

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key == "tag":
                    self.__post_init__()
            else:
                raise AttributeError(f"Attribute {key} does not exist.")

    def get(self, key: str, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default

    def __repr__(self):
        return f"wm_args(tag='{self.tag}', learning_rate={self.learning_rate}, batch_size={self.train_batch_size})"


# test
# args = wm_args()
# print("dtype string:", args.dtype)
# print("actual dtype object:", args.dtype_obj)
# print("output_dir:", args.output_dir)
# print("to_dict result:", args.to_dict())
# print("Number of config items:", len(args.to_dict()))
