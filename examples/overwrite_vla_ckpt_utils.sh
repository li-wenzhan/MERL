#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <source_ckpt_path> [target_actor_asset_path]" >&2
    exit 1
fi

SOURCE_CKPT_PATH="$1"
TARGET_CKPT_PATH="${2:-$1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python - "$SOURCE_CKPT_PATH" "$TARGET_CKPT_PATH" "$SCRIPT_DIR" <<'PY'
import json
import os
import shutil
import sys
import time
from pathlib import Path


source = Path(sys.argv[1]).expanduser().resolve()
target = Path(sys.argv[2]).expanduser()
repo_root = Path(sys.argv[3]).expanduser().resolve()
if not target.is_absolute():
    target = (Path.cwd() / target).resolve()
else:
    target = target.resolve()

if not source.is_dir():
    raise SystemExit(f"Error: source checkpoint path does not exist: {source}")

runtime_files = [
    repo_root / "verl/utils/vla_utils/openvla_oft/configuration_prismatic.py",
    repo_root / "verl/utils/vla_utils/openvla_oft/constants.py",
    repo_root / "verl/utils/vla_utils/openvla_oft/modeling_prismatic.py",
    repo_root / "verl/utils/vla_utils/openvla_oft/processing_prismatic.py",
    repo_root / "verl/utils/vla_utils/openvla_oft/train_utils.py",
]
runtime_file_names = {path.name for path in runtime_files}
missing_runtime = [str(path) for path in runtime_files if not path.is_file()]
if missing_runtime:
    raise SystemExit("Missing OpenVLA runtime files:\n" + "\n".join(missing_runtime))

target.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


source_config_path = source / "config.json"
if not source_config_path.is_file():
    raise SystemExit(f"Error: source config.json is missing: {source_config_path}")

try:
    config = load_json(source_config_path)
    config_source = source_config_path
except Exception:
    config = None
    config_source = None
    backups = sorted(
        source.glob("config.json.back.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for backup_path in backups:
        try:
            config = load_json(backup_path)
            config_source = backup_path
            break
        except Exception:
            continue
    if config is None:
        raise SystemExit(
            "Error: source config.json is empty or invalid and no valid "
            f"config.json.back.* backup was found under {source}."
        )

same_dir = source == target
if not same_dir:
    for item in source.iterdir():
        if (
            item.name == "config.json"
            or item.name.startswith("config.json.back.")
            or item.name in runtime_file_names
        ):
            continue
        dst = target / item.name
        if dst.is_symlink():
            try:
                if Path(os.readlink(dst)) == item:
                    continue
            except OSError:
                pass
            dst.unlink()
        if dst.exists():
            continue
        try:
            os.symlink(str(item), str(dst), target_is_directory=item.is_dir())
        except FileExistsError:
            pass

target_auto_map = {
    "AutoConfig": "configuration_prismatic.OpenVLAConfig",
    "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    "AutoProcessor": "processing_prismatic.PrismaticProcessor",
}
config["auto_map"] = target_auto_map
tmp_config = target / f".config.json.tmp.{os.getpid()}.{time.time_ns()}"
target_config = target / "config.json"
with tmp_config.open("w", encoding="utf-8") as file_obj:
    json.dump(config, file_obj, indent=2)
    file_obj.write("\n")
    file_obj.flush()
    os.fsync(file_obj.fileno())
os.replace(tmp_config, target_config)

for runtime_file in runtime_files:
    runtime_target = target / runtime_file.name
    if runtime_target.is_symlink():
        runtime_target.unlink()
    shutil.copy2(runtime_file, runtime_target)

with (target / "SOURCE_MODEL_PATH.txt").open("w", encoding="utf-8") as file_obj:
    file_obj.write(str(source) + "\n")
    if config_source is not None:
        file_obj.write(f"config_source={config_source}\n")

print(f"[actor-assets] source: {source}")
print(f"[actor-assets] target: {target}")
print(f"[actor-assets] config source: {config_source}")
print("[actor-assets] OpenVLA runtime files are ready.")
PY
