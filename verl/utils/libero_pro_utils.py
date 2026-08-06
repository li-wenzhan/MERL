"""Utils for evaluating policies in LIBERO-PRO simulation environments."""

import json
import logging
import math
import os
import re
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import imageio
import numpy as np
import tensorflow as tf

import random

# from experiments.robot.robot_utils import (
#     DATE,
#     DATE_TIME,
# )


_GET_LIBERO_PATH = None
_OFFSCREEN_RENDER_ENV = None


def _require_libero_runtime():
    global _GET_LIBERO_PATH, _OFFSCREEN_RENDER_ENV

    if _GET_LIBERO_PATH is None or _OFFSCREEN_RENDER_ENV is None:
        try:
            from libero.libero import get_libero_path as imported_get_libero_path
            from libero.libero.envs import (
                OffScreenRenderEnv as imported_offscreen_render_env,
            )
        except ImportError as error:
            raise ImportError(
                "LIBERO_PRO is not importable. Set configs/evaluation_config.yaml:libero_pro_root "
                "or LIBERO_PRO_ROOT before using online LIBERO_PRO helpers."
            ) from error

        _GET_LIBERO_PATH = imported_get_libero_path
        _OFFSCREEN_RENDER_ENV = imported_offscreen_render_env

    return _GET_LIBERO_PATH, _OFFSCREEN_RENDER_ENV


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    get_libero_path, offscreen_render_env = _require_libero_runtime()
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = offscreen_render_env(**env_args)
    env.seed(
        0
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """

    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(
        img, expand_animations=False, dtype=tf.uint8
    )  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def get_libero_wrist_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


# def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
#     """Saves an MP4 replay of an episode."""
#     rollout_dir = f"./rollouts/{DATE}"
#     os.makedirs(rollout_dir, exist_ok=True)
#     processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
#     mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
#     video_writer = imageio.get_writer(mp4_path, fps=30)
#     for img in rollout_images:
#         video_writer.append_data(img)
#     video_writer.close()
#     print(f"Saved rollout MP4 at path {mp4_path}")
#     if log_file is not None:
#         log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
#     return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


# def normalize_gripper_action(action, binarize=True):
#     """
#     Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
#     Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
#     Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
#     the dataset wrapper.

#     Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
#     """
#     # Just normalize the last action to [-1,+1].
#     orig_low, orig_high = 0.0, 1.0
#     action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

#     if binarize:
#         # Binarize to -1 or +1.
#         action[..., -1] = np.sign(action[..., -1])

#     return action


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = (
        2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    )

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


# def invert_gripper_action(action):
#     """
#     Flips the sign of the gripper action (last dimension of action vector).
#     This is necessary for some environments where -1 = open, +1 = close, since
#     the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
#     """
#     action[..., -1] = action[..., -1] * -1.0
#     return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] = inverted_action[..., -1] * -1.0

    return inverted_action


def save_rollout_video(rollout_images, exp_name, task_name, step_idx, success):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./tmp_files/rollout/{exp_name}"
    os.makedirs(rollout_dir, exist_ok=True)
    # ran_id = random.randint(1, 10000)
    # processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    # mp4_path = f"{rollout_dir}/step={step_idx}--task={task_name}--success={success}--ran={ran_id}.mp4"
    mp4_path = f"{rollout_dir}/step={step_idx}--task={task_name}--success={success}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


# ------------------- Added from LIBERO PRO ----------------------
# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"
    LIBERO_OBJECT_OBJECT_OOD = "libero_object_object_ood"
    LIBERO_GOAL_OBJECT_OOD = "libero_goal_object_ood"
    LIBERO_10_OBJECT_OOD = "libero_10_object_ood"
    LIBERO_GOAL_RELATION_OOD = "libero_goal_relation_ood"
    LIBERO_SPATIAL_RELATION_OOD = "libero_spatial_relation_ood"
    LIBERO_10_RELATION_OOD = "libero_10_relation_ood"
    LIBERO_OBJECT_RELATION_OOD = "libero_object_relation_ood"
    LIBERO_10_SEMANTIC_OOD = "libero_10_semantic_ood"
    LIBERO_SPATIAL_SEMANTIC_OOD = "libero_spatial_semantic_ood"
    LIBERO_GOAL_SEMANTIC_OOD = "libero_goal_semantic_ood"
    LIBERO_OBJECT_SEMANTIC_OOD = "libero_object_semantic_ood"
    LIBERO_SPATIAL_OBJECT_OOD = "libero_spatial_object_ood"
    LIBERO_GOAL_TEMP = "libero_goal_temp"
    LIBERO_SPATIAL_TEMP = "libero_spatial_temp"
    LIBERO_10_TEMP = "libero_10_temp"
    LIBERO_OBJECT_TEMP = "libero_object_temp"
    LIBERO_ROBOCEREBRA = "libero_robocerebra"
    LIBERO_GOAL_LAN = "libero_goal_lan"
    LIBERO_SPATIAL_LAN = "libero_spatial_lan"
    LIBERO_10_LAN = "libero_10_lan"
    LIBERO_OBJECT_LAN = "libero_object_lan"
    LIBERO_GOAL_OBJECT = "libero_goal_object"
    LIBERO_SPATIAL_OBJECT = "libero_spatial_object"
    LIBERO_10_OBJECT = "libero_10_object"
    LIBERO_OBJECT_OBJECT = "libero_object_object"
    LIBERO_GOAL_SWAP = "libero_goal_swap"
    LIBERO_SPATIAL_SWAP = "libero_spatial_swap"
    LIBERO_10_SWAP = "libero_10_swap"
    LIBERO_OBJECT_SWAP = "libero_object_swap"
    LIBERO_GOAL_TASK = "libero_goal_task"
    LIBERO_SPATIAL_TASK = "libero_spatial_task"
    LIBERO_10_TASK = "libero_10_task"
    LIBERO_OBJECT_TASK = "libero_object_task"
    LIBERO_GOAL_ENV = "libero_goal_env"
    LIBERO_SPATIAL_ENV = "libero_spatial_env"
    LIBERO_10_ENV = "libero_10_env"
    LIBERO_OBJECT_ENV = "libero_object_env"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
    TaskSuite.LIBERO_OBJECT_OBJECT_OOD: 280,
    TaskSuite.LIBERO_GOAL_OBJECT_OOD: 300,
    TaskSuite.LIBERO_10_OBJECT_OOD: 520,
    TaskSuite.LIBERO_GOAL_RELATION_OOD: 300,
    TaskSuite.LIBERO_SPATIAL_RELATION_OOD: 220,
    TaskSuite.LIBERO_10_RELATION_OOD: 520,
    TaskSuite.LIBERO_OBJECT_RELATION_OOD: 280,
    TaskSuite.LIBERO_10_SEMANTIC_OOD: 520,
    TaskSuite.LIBERO_SPATIAL_SEMANTIC_OOD: 220,
    TaskSuite.LIBERO_GOAL_SEMANTIC_OOD: 300,
    TaskSuite.LIBERO_OBJECT_SEMANTIC_OOD: 280,
    TaskSuite.LIBERO_SPATIAL_OBJECT_OOD: 280,
    TaskSuite.LIBERO_GOAL_TEMP: 300,
    TaskSuite.LIBERO_SPATIAL_TEMP: 220,
    TaskSuite.LIBERO_10_TEMP: 520,
    TaskSuite.LIBERO_OBJECT_TEMP: 280,
    TaskSuite.LIBERO_ROBOCEREBRA: 100,
    TaskSuite.LIBERO_GOAL_LAN: 300,
    TaskSuite.LIBERO_SPATIAL_LAN: 220,
    TaskSuite.LIBERO_10_LAN: 520,
    TaskSuite.LIBERO_OBJECT_LAN: 280,
    TaskSuite.LIBERO_GOAL_OBJECT: 300,
    TaskSuite.LIBERO_SPATIAL_OBJECT: 220,
    TaskSuite.LIBERO_10_OBJECT: 520,
    TaskSuite.LIBERO_OBJECT_OBJECT: 280,
    TaskSuite.LIBERO_GOAL_SWAP: 300,
    TaskSuite.LIBERO_SPATIAL_SWAP: 220,
    TaskSuite.LIBERO_10_SWAP: 520,
    TaskSuite.LIBERO_OBJECT_SWAP: 280,
    TaskSuite.LIBERO_GOAL_TASK: 300,
    TaskSuite.LIBERO_SPATIAL_TASK: 220,
    TaskSuite.LIBERO_10_TASK: 520,
    TaskSuite.LIBERO_OBJECT_TASK: 280,
    TaskSuite.LIBERO_GOAL_ENV: 300,
    TaskSuite.LIBERO_SPATIAL_ENV: 220,
    TaskSuite.LIBERO_10_ENV: 520,
    TaskSuite.LIBERO_OBJECT_ENV: 280,
}


def load_initial_states(cfg, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def load_initial_states_by_path(initial_states_path: str, task_suite, task_id: int):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if initial_states_path != "DEFAULT":
        with open(initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        print(f"Using initial states from {initial_states_path}")
        return initial_states, all_initial_states
    else:
        print("Using default initial states")
        return initial_states, None


def extract_task_from_bddl(bddl_file_path):
    # 用正则匹配 (:language ... ) 中间的文本
    # 例如：(:language Pick the alphabet soup and place it in the basket)
    language_pattern = re.compile(r"\(:language\s*(.*?)\)", re.IGNORECASE | re.DOTALL)

    tasks = []  # 用来保存 (文件名, language_text)

    bddl_dir = Path(bddl_file_path)
    for bddl_file in bddl_dir.glob("*.bddl"):
        with bddl_file.open("r", encoding="utf-8") as f:
            content = f.read()

        # 查找所有 (:language ...) 段
        matches = language_pattern.findall(content)
        if matches:
            # 如果一个文件有多个 (:language ...)，这里全部提取
            for lang_text in matches:
                # 去掉首尾空白和换行
                lang_text_clean = lang_text.strip()
                tasks.append(lang_text_clean)
        else:
            task = content.split(":language")[1].split(")")[0].strip()
            tasks.append(task)

    return tasks
