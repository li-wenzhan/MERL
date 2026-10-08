#!/usr/bin/env python
"""Validate and fingerprint a frozen LIBERO BDDL/initial-state panel."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merl.libero_states import load_init_states


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl-dir", type=Path, required=True)
    parser.add_argument("--init-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="LIBERO-PRO checkout or asset directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_tasks < 1 or args.min_trials < 1:
        parser.error("expected-tasks and min-trials must be positive")
    if args.output.exists():
        parser.error("output already exists; retain the earlier manifest")
    bddls = {p.stem: p for p in args.bddl_dir.glob("*.bddl")}
    states = {p.stem: p for p in args.init_dir.glob("*.pruned_init")}
    if len(bddls) != args.expected_tasks or set(bddls) != set(states):
        raise ValueError(f"Incomplete panel: BDDL={len(bddls)}, init={len(states)}, "
                         f"missing={sorted(set(bddls) - set(states))}, extra={sorted(set(states) - set(bddls))}")
    records = []
    for task in sorted(bddls):
        values = np.asarray(load_init_states(states[task]))
        if len(values) < args.min_trials:
            raise ValueError(f"Too few initial states for {task}: {len(values)}")
        records.append({"task": task, "bddl_sha256": digest(bddls[task]),
                        "init_sha256": digest(states[task]), "shape": list(values.shape),
                        "unique_states": int(len(np.unique(values, axis=0)))})
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "bddl_dir": str(args.bddl_dir.resolve()), "init_dir": str(args.init_dir.resolve()),
                "config": str(args.config.resolve()), "config_sha256": digest(args.config),
                "source_root": str(args.source_root.resolve()),
                "source_revision_label": os.environ.get("LIBERO_PRO_REVISION"),
                "generation_seed": "not_recorded_by_external_generator",
                "task_count": len(records), "tasks": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Validated {len(records)} tasks; frozen panel manifest: {args.output}")


if __name__ == "__main__":
    main()
