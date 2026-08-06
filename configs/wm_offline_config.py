from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CTRL_WORLD_SAMPLE_JSON_DIR = str(
    _REPO_ROOT / "modules" / "ctrl_world" / "dataset" / "libero"
)

# ! Reader checklist:
# ! 1) Set svd_model_path / clip_model_path to your local backbone directories.
# ! 2) Set ckpt_path only if you want to warm-start offline WM training.
# ! 3) Set libero_root to your local LIBERO codebase root for offline dataset tools.
# ! 4) Set dataset_sample_json_dir to the LIBERO sample-json directory used by Ctrl-World.


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
    ckpt_path = "/mnt/afs/task3_2/L202500276_lwz/models/ctrl_world_ckpts/checkpoint-1.pt"  # ! todo: optional warm-start checkpoint for offline WM training
    # ckpt_path: Optional[str] = None
    load_from_ckpt: bool = False

    # dataset parameters
    num_views: int = 1
    is_img_pregenerated: bool = True
    dataset_root_path: str = "dataset_example"
    dataset_names: str = "droid_subset"
    dataset_meta_info_path: str = "dataset_meta_info"
    libero_root: Optional[str] = (
        "/mnt/afs/L202500276/benchmark/LIBERO"  # ! todo: set to your local LIBERO repository root for offline dataset regeneration scripts
    )
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
        "libero_all_with_rm"  # ! todo: rename this offline experiment tag before launching training
    )
    output_dir: str = field(init=False)
    wandb_run_name: str = field(init=False)
    wandb_project_name: str = "libero"
    videos_col: int = 3

    # training parameters
    learning_rate: float = 5e-6
    gradient_accumulation_steps: int = 2
    mixed_precision: str = "fp16"
    train_batch_size: int = 4
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
    fps: int = 7
    guidance_scale: float = 2
    num_inference_steps: int = 50
    decode_chunk_size: int = 7
    width: int = 320
    height: int = 192
    num_frames: int = 8
    num_history: int = 8
    action_dim: int = 7
    text_cond: bool = True
    frame_level_cond: bool = True
    his_cond_zero: bool = False
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

    def __repr__(self):
        return f"wm_args(tag='{self.tag}', learning_rate={self.learning_rate}, batch_size={self.train_batch_size})"


# test
# args = wm_args()
# print("dtype string:", args.dtype)
# print("actual dtype object:", args.dtype_obj)
# print("output_dir:", args.output_dir)
# print("to_dict result:", args.to_dict())
# print("Number of config items:", len(args.to_dict()))
