"""Fit a frozen residual snapshot from exact stored calibration windows.

Run: python -m merl.calibrate_trust --input windows.pt --output residual.pt --stage 1
Input is a weights-only torch dictionary; see docs/no_oracle_trust.md.
"""

import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch

from .trust import CalibrationBatch, ChunkFeatures, ResidualPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()
    data = torch.load(args.input, map_location="cpu", weights_only=True)
    features = ChunkFeatures(**data.pop("features"))
    allowed = {field.name for field in fields(CalibrationBatch)} - {"features"}
    if set(data) - allowed:
        raise ValueError(f"unknown calibration fields: {sorted(set(data) - allowed)}")
    for key in ("trajectory_ids", "anchor_ids", "predicted_anchor_ids"):
        data[key] = tuple(data[key])
    batch = CalibrationBatch(features=features, **data)
    predictor = ResidualPredictor(hidden_dim=args.hidden_dim, seed=args.seed)
    stats = predictor.fit(batch, stage=args.stage, steps=args.steps)
    predictor.save(args.output)
    stats.update(seed=args.seed, steps=args.steps, input=str(args.input.resolve()),
                 output=str(args.output.resolve()), metrics_split="calibration_train_only")
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
