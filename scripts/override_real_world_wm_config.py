from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Write a real-world WM YAML config with runtime overrides.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file_obj:
        payload: dict[str, Any] = yaml.safe_load(file_obj) or {}

    if args.num_inference_steps is not None:
        payload["num_inference_steps"] = int(args.num_inference_steps)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False, allow_unicode=True)

    print(f"[wm-config] wrote {output_path}")


if __name__ == "__main__":
    main()
