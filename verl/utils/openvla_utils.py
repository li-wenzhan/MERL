"""Utils for evaluating OpenVLA or fine-tuned OpenVLA policies."""

import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

# Apply JSON numpy patch for serialization
json_numpy.patch()

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


_VLA_RUNTIME_FILES = {
    "openvla": (
        "configuration_prismatic.py",
        "modeling_prismatic.py",
        "processing_prismatic.py",
    ),
    "openvla-oft": (
        "configuration_prismatic.py",
        "constants.py",
        "modeling_prismatic.py",
        "processing_prismatic.py",
        "train_utils.py",
    ),
}


def _get_vla_runtime_source_dir(vla_name: str) -> Path:
    normalized_name = str(vla_name or "").strip()
    if normalized_name not in _VLA_RUNTIME_FILES:
        raise ValueError(f"Unsupported VLA runtime variant: {normalized_name}")

    runtime_dir_name = "openvla_oft" if normalized_name == "openvla-oft" else "openvla"
    return Path(__file__).resolve().parent / "vla_utils" / runtime_dir_name


def _get_transformers_dynamic_module_root() -> Path:
    hf_home = Path(
        os.path.expanduser(
            os.environ.get("HF_HOME", os.path.join("~", ".cache", "huggingface"))
        )
    )
    return hf_home / "modules" / "transformers_modules"


def _invalidate_transformers_dynamic_module_cache(pretrained_checkpoint: str) -> None:
    cache_root = _get_transformers_dynamic_module_root()
    if not cache_root.is_dir():
        return

    checkpoint_name = Path(pretrained_checkpoint).resolve().name
    normalized_checkpoint_name = checkpoint_name.replace("-", "_hyphen_").replace(
        ".", "_dot_"
    )
    candidate_names = {
        checkpoint_name,
        normalized_checkpoint_name,
        checkpoint_name.replace("-", "_").replace(".", "_"),
    }
    candidate_names_lower = {name.lower() for name in candidate_names}

    for candidate_name in candidate_names:
        candidate_path = cache_root / candidate_name
        if candidate_path.is_dir():
            shutil.rmtree(candidate_path)
            print(
                "Removed stale transformers dynamic-module cache at: "
                f"{os.path.abspath(candidate_path)}"
            )

    for candidate_path in cache_root.iterdir():
        if not candidate_path.is_dir():
            continue
        candidate_key = candidate_path.name.lower()
        if (
            candidate_key in candidate_names_lower
            or normalized_checkpoint_name.lower() in candidate_key
        ):
            shutil.rmtree(candidate_path)
            print(
                "Removed stale transformers dynamic-module cache at: "
                f"{os.path.abspath(candidate_path)}"
            )


def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    target_auto_map = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
        "AutoProcessor": "processing_prismatic.PrismaticProcessor",
    }

    lock_path = os.path.join(pretrained_checkpoint, ".config_json.lock")
    lock_file = open(lock_path, "w")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except Exception:
            pass

        def _load_json_file(path: str):
            with open(path, "r", encoding="utf-8") as file_obj:
                return json.load(file_obj)

        try:
            config = _load_json_file(config_path)
        except Exception as exc:
            backups = sorted(
                Path(pretrained_checkpoint).glob("config.json.back.*"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            recovered = None
            for backup_path in backups:
                try:
                    recovered = _load_json_file(str(backup_path))
                    shutil.copy2(str(backup_path), config_path)
                    print(
                        "Recovered invalid config.json from backup: "
                        f"{os.path.abspath(str(backup_path))}"
                    )
                    break
                except Exception:
                    continue
            if recovered is None:
                raise RuntimeError(
                    "config.json is missing, empty, or invalid and no valid "
                    f"backup was found under {pretrained_checkpoint}. Restore the "
                    "original model config.json before launching training."
                ) from exc
            config = recovered

        if config.get("auto_map") == target_auto_map:
            print(
                f"config.json auto_map already up to date at: {os.path.abspath(config_path)}"
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            pretrained_checkpoint, f"config.json.back.{timestamp}.{os.getpid()}"
        )
        shutil.copy2(config_path, backup_path)
        print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

        config["auto_map"] = target_auto_map

        tmp_path = os.path.join(
            pretrained_checkpoint, f".config.json.tmp.{os.getpid()}.{time.time_ns()}"
        )
        with open(tmp_path, "w", encoding="utf-8") as file_obj:
            json.dump(config, file_obj, indent=2)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(tmp_path, config_path)
    finally:
        try:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_UN)
        except Exception:
            pass
        lock_file.close()

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print(
        '  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"'
    )
    print('  - Set AutoProcessor to "processing_prismatic.PrismaticProcessor"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(
    curr_filepath: str, checkpoint_filepath: str, file_type: str
) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(
                f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}"
            )

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(
                f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}"
            )
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str, vla_name: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint runtime files for the
    requested VLA variant. If checkpoint files exist but differ, this function
    creates backups and overwrites them with the current repository versions.
    It also clears the corresponding transformers dynamic-module cache so the
    next trust_remote_code load cannot silently reuse stale Python files.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        vla_name: Runtime variant, e.g. "openvla" or "openvla-oft"
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    source_dir = _get_vla_runtime_source_dir(vla_name)
    for filename in _VLA_RUNTIME_FILES[str(vla_name).strip()]:
        curr_filepath = source_dir / filename
        if not curr_filepath.is_file():
            print(
                f"WARNING: `{curr_filepath}` does not exist in the current repository."
            )
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(str(curr_filepath), checkpoint_filepath, filename)

    _invalidate_transformers_dynamic_module_cache(pretrained_checkpoint)
