"""Read-only checkpoint validation and optional single-image CUDA inference.

Uses MERL's local OFT model classes without modifying the downloaded checkpoint.
This is an inference compatibility check, not a policy success evaluation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--unnorm-key", default="libero_10_no_noops")
    parser.add_argument("--image", help="Enable CUDA inference on this real RGB observation")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--output", help="Write a new JSON validation record")
    args = parser.parse_args()
    if args.image and not args.instruction.strip():
        parser.error("--instruction is required with --image")
    if args.output and Path(args.output).exists():
        parser.error("--output already exists")
    root = Path(args.checkpoint).expanduser().resolve()
    for name in ("config.json", "dataset_statistics.json", "tokenizer_config.json",
                 "tokenizer.json", "preprocessor_config.json", "model.safetensors.index.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    stats = json.loads((root / "dataset_statistics.json").read_text())
    if args.unnorm_key not in stats:
        raise ValueError(f"Unknown action normalization key; available: {list(stats)}")
    action_stats = stats[args.unnorm_key]["action"]
    if len(action_stats["q01"]) != 7 or len(action_stats["q99"]) != 7:
        raise ValueError("LIBERO requires seven action dimensions")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    from safetensors import safe_open

    count = 0
    for shard in sorted(set(index["weight_map"].values())):
        path = (root / shard).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Checkpoint shard escapes checkpoint directory")
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            expected = {key for key, value in index["weight_map"].items() if value == shard}
            if set(tensors.keys()) != expected:
                raise ValueError(f"Checkpoint index does not match shard: {shard}")
            count += len(expected)
    result = {"checkpoint": str(root), "tensor_count": count,
              "shard_count": len(set(index["weight_map"].values())),
              "unnorm_key": args.unnorm_key, "inference_checked": False}
    print("[preflight] checkpoint structure OK", result, flush=True)
    if args.image:
        os.environ["ROBOT_PLATFORM"] = "LIBERO"
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoTokenizer
        from verl.utils.vla_utils.openvla_oft.configuration_prismatic import OpenVLAConfig
        from verl.utils.vla_utils.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
        from verl.utils.vla_utils.openvla_oft.processing_prismatic import (
            PrismaticImageProcessor, PrismaticProcessor,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the inference preflight")
        torch.manual_seed(0)
        started = time.monotonic()
        config = OpenVLAConfig.from_pretrained(root, local_files_only=True)
        config.use_proprio = False
        model, loading = OpenVLAForActionPrediction.from_pretrained(
            root, config=config, torch_dtype=torch.bfloat16,
            attn_implementation="eager", low_cpu_mem_usage=True,
            local_files_only=True, output_loading_info=True,
        )
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if loading.get(field):
                raise RuntimeError(f"Checkpoint/model mismatch: {field}={loading[field]}")
        model.vision_backbone.set_num_images_in_input(1)
        model.norm_stats = stats
        model = model.eval().to("cuda")
        processor = PrismaticProcessor(
            image_processor=PrismaticImageProcessor.from_pretrained(root, local_files_only=True),
            tokenizer=AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False),
        )
        image = Image.open(args.image).convert("RGB")
        prompt = f"In: What action should the robot take to {args.instruction.strip().lower()}?\nOut:"
        inputs = processor(prompt, image, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            actions, _ = model.predict_action(**inputs, unnorm_key=args.unnorm_key)
        actions = np.asarray(actions)
        if actions.shape != (8, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid action output: shape={actions.shape}")
        torch.cuda.synchronize()
        result.update(inference_checked=True, actions=actions.tolist(), image=str(Path(args.image).resolve()),
                      instruction=args.instruction, elapsed_seconds=time.monotonic() - started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(), torch_version=torch.__version__,
                      purpose="single_image_compatibility_not_policy_evaluation")
        print("[preflight] OpenVLA-OFT CUDA inference OK", actions.shape, flush=True)
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
            file.write("\n")


if __name__ == "__main__":
    main()
