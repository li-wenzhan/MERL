"""Create PPT-ready assets from every saved trial, without outcome selection."""

import argparse
import csv
import json
import math
from pathlib import Path


def load_panel(folder, protocol):
    records = []
    for path in sorted((folder / "episodes").glob("step_*/*/episode.json")):
        item = json.loads(path.read_text())
        if item["protocol_id"] != protocol["id"] or item["observation_source"] != "real_environment":
            raise ValueError(f"Incompatible evaluation artifact: {path}")
        records.append(dict(item, directory=str(path.parent)))
    if not records:
        return {}
    latest = max(r["global_step"] for r in records)
    panel = {}
    expected = {(task, trial) for task in protocol["task_ids"] for trial in protocol["trial_ids"]}
    for record in records:
        if record["global_step"] != latest:
            continue
        key = (record["task_id"], record["trial_id"])
        if key not in expected or key in panel:
            raise ValueError(f"Unexpected or duplicate trial {key} under {folder}")
        panel[key] = record
    return panel


def training_evidence(info):
    records = {}
    for path in Path(info["experiment_dir"]).glob("run_*.log"):
        for line in path.read_text(errors="replace").splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            try:
                row = json.loads(parts[2])
                if "train/global_step" in row:
                    records[int(row["train/global_step"])] = row
            except (ValueError, TypeError):
                continue
    def finite_values(key):
        return [float(row[key]) for row in records.values()
                if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
    return dict(completed_outer_steps=len(records),
                optimizer_step_metric_max=max(finite_values("actor/optimizer_step_count"), default=0),
                gradient_norm_max=max(finite_values("actor/grad_norm"), default=0),
                imagined_actor_tokens_logged=sum(finite_values("wm/actor_input_imag_token_count")),
                imagined_actor_weight_max=max(finite_values("wm/actor_input_imag_weight_mean"), default=0),
                world_model_update_steps=sum(finite_values("wm/update/steps_done")),
                checkpoints=[str(p) for p in sorted((Path(info["experiment_dir"]) / "actor").glob("global_step_*"))
                             if (p / "config.json").is_file()])


def font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_panel_frame(records, images, index):
    from PIL import Image, ImageDraw
    width, height = 320, 400
    canvas = Image.new("RGB", (width * len(records), height), "#101827")
    draw = ImageDraw.Draw(canvas)
    for column, (record, frame) in enumerate(zip(records, images)):
        x = column * width
        canvas.paste(Image.fromarray(frame).resize((width, width)), (x, 80))
        draw.text((x + 10, 6), record["label"], font=font(23), fill="white")
        result = "SUCCESS" if record["success"] else "NOT SUCCESSFUL"
        draw.text((x + 10, 34), f"{result} | steps {record['environment_steps']}", font=font(15), fill="#bdd8ef")
        frame_text = f"frame {min(index, record['frame_count'] - 1)}"
        if index >= record["frame_count"]:
            frame_text += " | episode ended (held)"
        draw.text((x + 10, 57), frame_text, font=font(13), fill="white")
    return canvas


def render_trial(records, output):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    readers = [imageio.get_reader(str(Path(r["directory"]) / r["video"])) for r in records]
    last = [None] * len(readers)
    length = max(r["frame_count"] for r in records)
    indices = set(int(x) for x in np.linspace(0, length - 1, 5))
    snapshots = []
    try:
        with imageio.get_writer(str(output.with_suffix(".mp4")), fps=30) as writer:
            for index in range(length):
                for i, (reader, record) in enumerate(zip(readers, records)):
                    if index < record["frame_count"]:
                        last[i] = reader.get_next_data()
                frame = render_panel_frame(records, last, index)
                writer.append_data(np.asarray(frame))
                if index in indices:
                    snapshots.append(frame.copy())
    finally:
        for reader in readers:
            reader.close()
    sheet = Image.new("RGB", (snapshots[0].width, 400 * len(snapshots)), "white")
    for row, frame in enumerate(snapshots):
        sheet.paste(frame, (0, row * 400))
    sheet.save(output.with_suffix(".png"))


def build_report(root):
    from PIL import Image, ImageDraw
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    expected = {(task, trial) for task in protocol["task_ids"] for trial in protocol["trial_ids"]}
    output = root / "report"
    output.mkdir(exist_ok=True)
    panels, summaries, rows = [], [], []
    for mode in ("MFRL", "MBRL", "MERL"):
        folder = root / mode
        if not (folder / "run_info.json").exists():
            continue
        info = json.loads((folder / "run_info.json").read_text())
        panel = load_panel(folder, protocol)
        valid = sum(r["valid"] for r in panel.values())
        complete = (set(panel) == expected and valid == len(expected)
                    and info["status"] == "completed" and info.get("assets_unchanged", False))
        successes = sum(r["valid"] and r["success"] for r in panel.values())
        evidence = training_evidence(info)
        warnings = []
        if not complete:
            warnings.append("Incomplete/invalid evaluation: no comparable success rate")
        if info["job"] == "train" and evidence["gradient_norm_max"] == 0:
            warnings.append("No nonzero finite actor gradient was logged; do not claim learned improvement")
        if mode == "MERL" and info["job"] == "train" and evidence["imagined_actor_tokens_logged"] == 0:
            warnings.append("No imagined actor tokens were logged; full MERL mechanism is not demonstrated")
        if mode in ("MERL", "MBRL") and info["job"] == "train" and evidence["imagined_actor_weight_max"] == 0:
            warnings.append("No positive imagined actor weight was logged; WM contribution is not demonstrated")
        if mode == "MERL" and info["job"] == "train" and evidence["world_model_update_steps"] == 0:
            warnings.append("No completed world-model updates were logged; simulator evolution is not demonstrated")
        if mode == "MBRL" and evidence["world_model_update_steps"] > 0:
            warnings.append("World-model updates were logged in MBRL; inspect the frozen-simulator contract")
        if any(not r["video"] or r["frame_count"] <= 0 for r in panel.values()):
            warnings.append("Some evaluation episodes have no saved video")
        summary = dict(mode=info["label"], status=info["status"], evaluation_complete=complete,
                       requested_trials=len(expected), observed_trials=len(panel), valid_trials=valid,
                       successes=successes, success_rate=successes / len(expected) if complete else None,
                       mean_environment_steps=(sum(r["environment_steps"] for r in panel.values()) / len(expected)) if complete else None,
                       elapsed_seconds=info.get("elapsed_seconds"), warnings=warnings, **evidence)
        summaries.append(summary)
        for key, record in sorted(panel.items()):
            rows.append({key: record[key] for key in ("label", "task_id", "trial_id", "success", "valid",
                          "environment_steps", "global_step", "failure_reason", "directory")})
        if complete and all(r["video"] for r in panel.values()):
            panels.append(panel)
    (output / "summary.json").write_text(json.dumps(dict(protocol=protocol, results=summaries), indent=2) + "\n")
    if rows:
        with (output / "trials.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    canvas = Image.new("RGB", (1200, 140 + 105 * max(1, len(summaries))), "#101827")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), "Short-budget pilot | real-environment evaluation", font=font(30), fill="white")
    draw.text((30, 66), f"Tasks {protocol['task_ids']} | {protocol['trials']} initial states/task | horizon 512", font=font(20), fill="#bdd8ef")
    for index, row in enumerate(summaries):
        y = 125 + index * 105
        score = f"{row['successes']}/{row['requested_trials']} ({row['success_rate']:.0%})" if row["success_rate"] is not None else "INCOMPLETE"
        draw.text((30, y), f"{row['mode']}: {score} | outer updates: {row['completed_outer_steps']}", font=font(27), fill="white")
        draw.text((30, y + 40), f"Status: {row['status']} | See summary.json for update evidence and limitations", font=font(17), fill="#bdd8ef")
    canvas.save(output / "scores.png")
    for task, trial in sorted(expected):
        if panels:
            render_trial([panel[(task, trial)] for panel in panels], output / f"task_{task:02d}_trial_{trial:02d}")
    notes = ["# Presentation assets", "", "Exploratory short-budget pilot, not paper reproduction or a proven method ranking.",
             "All requested trials are retained. Rates are withheld for incomplete or invalid panels.",
             "Videos align by stored frame index at 30 playback FPS; ended episodes hold their final frame and are labeled.",
             "The small fixed task panel uses evaluation states disjoint from online training states. SFT data overlap is unknown.",
             "This is descriptive evidence, not a general benchmark or significance test.",
             "Training budgets may differ in completed updates and interactions under the same wall-time cap.", ""]
    for row in summaries:
        notes.append(f"- {row['mode']}: status={row['status']}; observed={row['observed_trials']}/{row['requested_trials']}; success_rate={row['success_rate']}")
        notes.extend(f"  - {warning}" for warning in row["warnings"])
    (output / "README.md").write_text("\n".join(notes) + "\n")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build_report(parser.parse_args().root)
