from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MetricSpec:
    slug: str
    label: str
    group: str
    aliases: Tuple[str, ...] = ()
    derived: Optional[str] = None
    higher_is_better: bool = True


METRIC_SPECS: Tuple[MetricSpec, ...] = (
    MetricSpec(
        slug="success",
        label="Success Rate",
        group="core",
        aliases=(
            "val/success_rate/all",
            "success_rate/all",
            "val/test_score/all",
            "test_score/all",
        ),
    ),
    MetricSpec(
        slug="train_reward",
        label="Train Reward",
        group="core",
        aliases=(
            "train_reward/reward_all",
            "train_reward/reward_model",
            "train_reward/verifier",
            "critic/rewards/mean",
        ),
    ),
    MetricSpec(
        slug="critic_reward_mean",
        label="Critic Reward Mean",
        group="core",
        aliases=(
            "critic/rewards/mean",
            "train_reward/reward_all",
            "train_reward/reward_model",
            "train_reward/verifier",
        ),
    ),
    MetricSpec(
        slug="cum_real_samples",
        label="Cumulative Real Samples",
        group="core",
        derived="cum_real_samples",
    ),
    MetricSpec(
        slug="cum_wm_samples",
        label="Cumulative WM Samples",
        group="core",
        derived="cum_wm_samples",
    ),
    MetricSpec(
        slug="wm_ratio_real",
        label="Real Ratio",
        group="wm",
        aliases=("wm/ratio_real",),
    ),
    MetricSpec(
        slug="wm_ratio_wm",
        label="WM Ratio",
        group="wm",
        aliases=("wm/ratio_wm",),
    ),
    MetricSpec(
        slug="wm_loss",
        label="WM Loss",
        group="wm",
        aliases=("wm/loss", "wm/loss_ema"),
        higher_is_better=False,
    ),
    MetricSpec(
        slug="wm_loss_ema",
        label="WM Loss EMA",
        group="wm",
        aliases=("wm/loss_ema", "wm/loss"),
        higher_is_better=False,
    ),
    MetricSpec(
        slug="wm_ratio_signal_ema",
        label="WM Ratio Signal EMA",
        group="wm",
        aliases=("wm/ratio_signal_ema", "wm/ratio_signal"),
        higher_is_better=False,
    ),
    MetricSpec(
        slug="wm_confidence_ema",
        label="WM Confidence EMA",
        group="wm",
        aliases=("wm/confidence_ema", "wm/chunk_confidence_mean"),
    ),
    MetricSpec(
        slug="wm_chunk_confidence",
        label="WM Chunk Confidence",
        group="wm",
        aliases=("wm/chunk_confidence_mean",),
    ),
    MetricSpec(
        slug="wm_obs_error",
        label="WM Obs Error",
        group="wm",
        aliases=("wm/chunk_obs_error_mean",),
        higher_is_better=False,
    ),
    MetricSpec(
        slug="wm_done_error",
        label="WM Done Error",
        group="wm",
        aliases=("wm/chunk_done_error_mean",),
        higher_is_better=False,
    ),
    MetricSpec(
        slug="wm_current_imag_horizon",
        label="Current Imagined Horizon",
        group="wm",
        aliases=("wm/current_imag_horizon", "wm/next_imag_horizon"),
    ),
    MetricSpec(
        slug="wm_next_imag_horizon",
        label="Next Imagined Horizon",
        group="wm",
        aliases=("wm/next_imag_horizon", "wm/current_imag_horizon"),
    ),
    MetricSpec(
        slug="wm_eval_psnr",
        label="WM Eval PSNR",
        group="wm",
        aliases=("wm/eval/psnr", "wm/eval/full/psnr"),
    ),
    MetricSpec(
        slug="wm_eval_ssim",
        label="WM Eval SSIM",
        group="wm",
        aliases=("wm/eval/ssim", "wm/eval/full/ssim"),
    ),
    MetricSpec(
        slug="wm_eval_lpips",
        label="WM Eval LPIPS",
        group="wm",
        aliases=("wm/eval/lpips", "wm/eval/full/lpips"),
        higher_is_better=False,
    ),
    MetricSpec(
        slug="wm_eval_clips",
        label="WM Eval CLIP-S",
        group="wm",
        aliases=("wm/eval/clips", "wm/eval/full/clips", "wm/eval/clip", "wm/eval/full/clip"),
    ),
    MetricSpec(
        slug="wm_eval_reward_mse",
        label="WM Eval Reward MSE",
        group="wm",
        aliases=("wm/eval/reward_MSE", "wm/eval/full/reward_MSE"),
        higher_is_better=False,
    ),
    MetricSpec(
        slug="wm_num_real_sample",
        label="Real Samples Per Step",
        group="wm",
        aliases=("wm/num_real_sample",),
    ),
    MetricSpec(
        slug="wm_num_wm_sample",
        label="WM Samples Per Step",
        group="wm",
        aliases=("wm/num_wm_sample",),
    ),
    MetricSpec(
        slug="actor_pg_loss_real",
        label="Actor PG Loss Real",
        group="actor",
        aliases=("actor/pg_loss_real", "actor/pg_loss"),
        higher_is_better=False,
    ),
    MetricSpec(
        slug="actor_pg_loss_imag",
        label="Actor PG Loss Imag",
        group="actor",
        aliases=("actor/pg_loss_imag",),
        higher_is_better=False,
    ),
    MetricSpec(
        slug="actor_imag_weight_mean",
        label="Imag Weight Mean",
        group="actor",
        aliases=("actor/imag_weight_mean",),
    ),
    MetricSpec(
        slug="actor_ppo_kl",
        label="Actor PPO KL",
        group="actor",
        aliases=("actor/ppo_kl",),
        higher_is_better=False,
    ),
    MetricSpec(
        slug="actor_pg_clipfrac",
        label="Actor ClipFrac",
        group="actor",
        aliases=("actor/pg_clipfrac",),
        higher_is_better=False,
    ),
    MetricSpec(
        slug="actor_entropy_eval",
        label="Actor Entropy Eval",
        group="actor",
        aliases=("actor_after/entropy_loss_eval",),
        higher_is_better=False,
    ),
)


GROUP_ORDER: Tuple[Tuple[str, str], ...] = (
    ("core", "Core Metrics"),
    ("wm", "World Model Metrics"),
    ("actor", "Actor Metrics"),
)

MODE_ORDER: Tuple[str, ...] = ("MFRL", "MBRL", "MERL")

MODE_COLORS: Dict[str, str] = {
    "MFRL": "#2F5597",
    "MBRL": "#B55D2C",
    "MERL": "#1F7A4D",
}

SUMMARY_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("mode", "Mode"),
    ("experiment_name", "Experiment"),
    ("last_step", "Last Step"),
    ("best_success", "Best Success"),
    ("best_success_step", "Best Success Step"),
    ("final_success", "Final Success"),
    ("best_train_reward", "Best Train Reward"),
    ("final_train_reward", "Final Train Reward"),
    ("final_wm_ratio_wm", "Final WM Ratio"),
    ("final_wm_loss_ema", "Final WM Loss EMA"),
    ("final_wm_confidence_ema", "Final WM Confidence EMA"),
    ("final_wm_eval_psnr", "Final WM Eval PSNR"),
    ("final_wm_eval_ssim", "Final WM Eval SSIM"),
    ("final_wm_eval_lpips", "Final WM Eval LPIPS"),
    ("final_wm_eval_clips", "Final WM Eval CLIP-S"),
    ("final_wm_eval_reward_mse", "Final WM Eval Reward MSE"),
    ("cum_real_samples", "Cum Real Samples"),
    ("cum_wm_samples", "Cum WM Samples"),
    ("success_auc", "Success AUC"),
)


@dataclass
class ExperimentResult:
    mode: str
    label: str
    experiment_dir: Path
    analysis_dir: Path
    log_path: Path
    records: List[Dict[str, Any]]
    raw_rows: List[Dict[str, Any]]
    metric_rows: List[Dict[str, Any]]
    series: Dict[str, Dict[str, Any]]
    source_keys: Dict[str, str]
    summary: Dict[str, Any]


def parse_constant(value: str) -> float:
    if value == "NaN":
        return float("nan")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    raise ValueError(f"Unsupported constant: {value}")


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "null":
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            return None
        return numeric if math.isfinite(numeric) else None
    return None


def format_scalar(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return "-"
        if abs(numeric) >= 1000 and abs(numeric - round(numeric)) < 1e-6:
            return str(int(round(numeric)))
        if abs(numeric - round(numeric)) < 1e-9:
            return str(int(round(numeric)))
        return f"{numeric:.4f}"
    return str(value)


def slugify(value: str) -> str:
    kept = []
    for char in value.lower():
        if char.isalnum():
            kept.append(char)
        elif char in ("-", "_", " "):
            kept.append("-")
    collapsed = "".join(kept).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed or "comparison"


def unique_list(items: Iterable[Any]) -> List[Any]:
    output: List[Any] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def area_under_curve(points: Sequence[Dict[str, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    area = 0.0
    for left, right in zip(points[:-1], points[1:]):
        dx = right["x"] - left["x"]
        if dx <= 0:
            continue
        area += 0.5 * dx * (left["y"] + right["y"])
    return area


def last_value(points: Sequence[Dict[str, float]]) -> Optional[float]:
    if not points:
        return None
    return points[-1]["y"]


def best_point(
    points: Sequence[Dict[str, float]], higher_is_better: bool
) -> Tuple[Optional[float], Optional[int]]:
    if not points:
        return None, None
    selector = max if higher_is_better else min
    picked = selector(points, key=lambda point: point["y"])
    return picked["y"], int(picked["x"])


def downsample_points(
    points: Sequence[Dict[str, float]], max_points: int
) -> List[Dict[str, float]]:
    if len(points) <= max_points:
        return list(points)
    stride = max(1, math.ceil(len(points) / max_points))
    sampled = list(points[::stride])
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def resolve_experiment_path(path_like: str) -> Tuple[Path, Path]:
    path = Path(path_like).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".log":
            raise FileNotFoundError(f"Expected a .log file, got: {path}")
        return path.parent, path
    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    candidates = unique_list(
        list(path.glob("run_*.log")) + list(path.glob("*.log"))
    )
    if not candidates:
        raise FileNotFoundError(f"No log file found under: {path}")

    candidates = sorted(
        candidates,
        key=lambda file_path: (file_path.stat().st_mtime, file_path.name),
    )
    return path, candidates[-1]


def load_and_merge_log(log_path: Path) -> List[Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue

            timestamp, step_str, json_str = parts
            try:
                step = int(step_str)
                payload = json.loads(json_str, parse_constant=parse_constant)
            except Exception:
                continue

            row = merged.setdefault(step, {"timestamp": timestamp, "step": step})
            if timestamp and not row.get("timestamp"):
                row["timestamp"] = timestamp
            for key, value in payload.items():
                if value == "null":
                    continue
                row[key] = value

    return [merged[step] for step in sorted(merged)]


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def choose_source_key(
    records: Sequence[Dict[str, Any]], aliases: Sequence[str]
) -> Optional[str]:
    for alias in aliases:
        for row in records:
            if safe_float(row.get(alias)) is not None:
                return alias
    return None


def extract_points(
    records: Sequence[Dict[str, Any]], source_key: str
) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []
    for row in records:
        numeric = safe_float(row.get(source_key))
        if numeric is None:
            continue
        points.append({"x": float(row["step"]), "y": numeric})
    return points


def build_derived_series(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, float]]]:
    derived: Dict[str, List[Dict[str, float]]] = {
        "cum_real_samples": [],
        "cum_wm_samples": [],
    }

    cumulative_real = 0.0
    cumulative_wm = 0.0
    for row in records:
        step = float(row["step"])
        real_sample = (
            safe_float(row.get("rollout/real_num_samples"))
            or safe_float(row.get("wm/num_real_sample"))
            or safe_float(row.get("env/real_sample_actual"))
            or safe_float(row.get("env/real_sample_target"))
            or 0.0
        )
        wm_sample = safe_float(row.get("wm/num_wm_sample")) or 0.0
        cumulative_real += real_sample
        cumulative_wm += wm_sample
        derived["cum_real_samples"].append({"x": step, "y": cumulative_real})
        derived["cum_wm_samples"].append({"x": step, "y": cumulative_wm})
    return derived


def build_metric_series(
    records: Sequence[Dict[str, Any]], spec: MetricSpec, derived_cache: Dict[str, List[Dict[str, float]]]
) -> Tuple[Optional[str], List[Dict[str, float]]]:
    if spec.derived is not None:
        return spec.derived, list(derived_cache.get(spec.derived, []))

    source_key = choose_source_key(records, spec.aliases)
    if source_key is None:
        return None, []
    return source_key, extract_points(records, source_key)


def build_metric_rows(
    records: Sequence[Dict[str, Any]], series: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    metric_by_step: Dict[int, Dict[str, Any]] = {
        int(row["step"]): {"step": int(row["step"]), "timestamp": row.get("timestamp", "")}
        for row in records
    }

    for slug, bundle in series.items():
        for point in bundle["points"]:
            metric_by_step[int(point["x"])][slug] = point["y"]

    return [metric_by_step[step] for step in sorted(metric_by_step)]


def summarize_experiment(
    mode: str,
    input_path: str,
) -> ExperimentResult:
    experiment_dir, log_path = resolve_experiment_path(input_path)
    analysis_dir = experiment_dir / "analysis"
    records = load_and_merge_log(log_path)
    if not records:
        raise ValueError(f"No valid metric row parsed from: {log_path}")

    derived_cache = build_derived_series(records)
    series: Dict[str, Dict[str, Any]] = {}
    source_keys: Dict[str, str] = {}
    for spec in METRIC_SPECS:
        source_key, points = build_metric_series(records, spec, derived_cache)
        if source_key is not None:
            source_keys[spec.slug] = source_key
        series[spec.slug] = {
            "label": spec.label,
            "group": spec.group,
            "source_key": source_key,
            "higher_is_better": spec.higher_is_better,
            "points": points,
        }

    metric_rows = build_metric_rows(records, series)

    success_points = series["success"]["points"]
    reward_points = series["train_reward"]["points"]

    best_success, best_success_step = best_point(success_points, higher_is_better=True)
    best_reward, best_reward_step = best_point(reward_points, higher_is_better=True)

    last_step = int(records[-1]["step"])
    summary: Dict[str, Any] = {
        "mode": mode,
        "experiment_name": experiment_dir.name,
        "experiment_dir": str(experiment_dir),
        "analysis_dir": str(analysis_dir),
        "log_path": str(log_path),
        "log_file": log_path.name,
        "last_step": last_step,
        "num_steps": len(records),
        "best_success": best_success,
        "best_success_step": best_success_step,
        "final_success": last_value(success_points),
        "best_train_reward": best_reward,
        "best_train_reward_step": best_reward_step,
        "final_train_reward": last_value(reward_points),
        "success_auc": area_under_curve(success_points),
        "cum_real_samples": last_value(series["cum_real_samples"]["points"]),
        "cum_wm_samples": last_value(series["cum_wm_samples"]["points"]),
        "final_wm_ratio_real": last_value(series["wm_ratio_real"]["points"]),
        "final_wm_ratio_wm": last_value(series["wm_ratio_wm"]["points"]),
        "final_wm_loss": last_value(series["wm_loss"]["points"]),
        "final_wm_loss_ema": last_value(series["wm_loss_ema"]["points"]),
        "final_wm_ratio_signal_ema": last_value(series["wm_ratio_signal_ema"]["points"]),
        "final_wm_confidence_ema": last_value(series["wm_confidence_ema"]["points"]),
        "final_wm_chunk_confidence": last_value(series["wm_chunk_confidence"]["points"]),
        "final_wm_obs_error": last_value(series["wm_obs_error"]["points"]),
        "final_wm_done_error": last_value(series["wm_done_error"]["points"]),
        "final_wm_current_imag_horizon": last_value(series["wm_current_imag_horizon"]["points"]),
        "final_wm_next_imag_horizon": last_value(series["wm_next_imag_horizon"]["points"]),
        "final_wm_eval_psnr": last_value(series["wm_eval_psnr"]["points"]),
        "final_wm_eval_ssim": last_value(series["wm_eval_ssim"]["points"]),
        "final_wm_eval_lpips": last_value(series["wm_eval_lpips"]["points"]),
        "final_wm_eval_clips": last_value(series["wm_eval_clips"]["points"]),
        "final_wm_eval_reward_mse": last_value(series["wm_eval_reward_mse"]["points"]),
        "final_actor_pg_loss_real": last_value(series["actor_pg_loss_real"]["points"]),
        "final_actor_pg_loss_imag": last_value(series["actor_pg_loss_imag"]["points"]),
        "final_actor_imag_weight_mean": last_value(series["actor_imag_weight_mean"]["points"]),
        "final_actor_ppo_kl": last_value(series["actor_ppo_kl"]["points"]),
    }

    return ExperimentResult(
        mode=mode,
        label=mode,
        experiment_dir=experiment_dir,
        analysis_dir=analysis_dir,
        log_path=log_path,
        records=records,
        raw_rows=records,
        metric_rows=metric_rows,
        series=series,
        source_keys=source_keys,
        summary=summary,
    )


def summary_rows(results: Sequence[ExperimentResult]) -> List[Dict[str, Any]]:
    return [result.summary for result in results]


def expected_order_check(
    results: Sequence[ExperimentResult], metric_key: str
) -> Dict[str, Any]:
    available: List[Tuple[str, float]] = []
    for result in results:
        numeric = safe_float(result.summary.get(metric_key))
        if numeric is None:
            continue
        available.append((result.mode, numeric))

    if len(available) < 3:
        return {
            "metric": metric_key,
            "expected": list(reversed(MODE_ORDER)),
            "actual": [mode for mode, _ in sorted(available, key=lambda item: item[1], reverse=True)],
            "match": None,
        }

    actual = [mode for mode, _ in sorted(available, key=lambda item: item[1], reverse=True)]
    expected = ["MERL", "MBRL", "MFRL"]
    return {
        "metric": metric_key,
        "expected": expected,
        "actual": actual,
        "match": actual == expected,
    }


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    headers = [title for _, title in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [format_scalar(row.get(key)) for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_experiment_markdown(result: ExperimentResult) -> str:
    summary_row = result.summary
    source_rows = []
    for spec in METRIC_SPECS:
        source_rows.append(
            {
                "metric": spec.label,
                "slug": spec.slug,
                "source_key": result.source_keys.get(spec.slug, "-"),
            }
        )

    summary_block = markdown_table([summary_row], SUMMARY_COLUMNS)
    source_block = markdown_table(
        source_rows,
        (("metric", "Metric"), ("slug", "Slug"), ("source_key", "Source Key")),
    )

    return "\n".join(
        [
            f"# {result.mode} Experiment Summary",
            "",
            f"- Experiment dir: {result.experiment_dir}",
            f"- Log file: {result.log_path}",
            f"- Generated at: {dt.datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Key Summary",
            "",
            summary_block,
            "",
            "## Metric Source Keys",
            "",
            source_block,
            "",
            "## Notes",
            "",
            "- Blank WM fields are expected for MFRL.",
            "- For MERL, wm/ratio_wm and wm/confidence_ema should move adaptively instead of staying fixed.",
            "- For MBRL, success should improve over MFRL, but WM-only updates often look less stable than MERL on the same horizon.",
        ]
    )


def render_comparison_markdown(results: Sequence[ExperimentResult]) -> str:
    checks = [
        expected_order_check(results, "best_success"),
        expected_order_check(results, "final_success"),
        expected_order_check(results, "final_train_reward"),
    ]

    summary_block = markdown_table(summary_rows(results), SUMMARY_COLUMNS)

    check_lines = []
    for item in checks:
        expected = " > ".join(item["expected"])
        actual = " > ".join(item["actual"]) if item["actual"] else "-"
        if item["match"] is None:
            status = "N/A"
        else:
            status = "PASS" if item["match"] else "FAIL"
        check_lines.append(
            f"- {item['metric']}: expected {expected}, actual {actual}, status {status}"
        )

    return "\n".join(
        [
            "# Three-Mode Comparison Summary",
            "",
            f"Generated at: {dt.datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Summary Table",
            "",
            summary_block,
            "",
            "## Ranking Checks",
            "",
            *check_lines,
            "",
            "## Reading Guide",
            "",
            "- First compare best_success and final_success.",
            "- Then compare train_reward and critic_reward_mean together.",
            "- In pure imagined MBRL steps, train_reward/verifier can stay at 0 by design; use train_reward/reward_all or train_reward/reward_model as the primary reward signal.",
            "- For MERL and MBRL, inspect wm/loss_ema, wm/confidence_ema, wm/ratio_wm, and wm/eval/* together.",
            "- If WM metrics look healthy but success drops, inspect actor/pg_loss_real, actor/pg_loss_imag, and actor/imag_weight_mean.",
        ]
    )


def chart_specs(
    results: Sequence[ExperimentResult],
    max_points: int,
) -> List[Dict[str, Any]]:
    charts: List[Dict[str, Any]] = []
    for group_slug, group_title in GROUP_ORDER:
        for spec in METRIC_SPECS:
            if spec.group != group_slug:
                continue
            per_mode_series = []
            for result in results:
                bundle = result.series.get(spec.slug, {})
                points = bundle.get("points", [])
                if not points:
                    continue
                per_mode_series.append(
                    {
                        "name": result.mode,
                        "mode": result.mode,
                        "color": MODE_COLORS.get(result.mode, "#444444"),
                        "source_key": bundle.get("source_key"),
                        "points": [
                            [point["x"], point["y"]]
                            for point in downsample_points(points, max_points)
                        ],
                    }
                )
            if not per_mode_series:
                continue
            charts.append(
                {
                    "id": f"chart-{spec.slug}",
                    "metric_slug": spec.slug,
                    "group": group_title,
                    "title": spec.label,
                    "higher_is_better": spec.higher_is_better,
                    "series": per_mode_series,
                }
            )
    return charts


def dashboard_summary_rows(results: Sequence[ExperimentResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "mode": result.mode,
                "experiment_name": result.summary["experiment_name"],
                "last_step": result.summary["last_step"],
                "best_success": result.summary["best_success"],
                "final_success": result.summary["final_success"],
                "best_train_reward": result.summary["best_train_reward"],
                "final_train_reward": result.summary["final_train_reward"],
                "final_wm_ratio_wm": result.summary["final_wm_ratio_wm"],
                "final_wm_loss_ema": result.summary["final_wm_loss_ema"],
                "final_wm_confidence_ema": result.summary["final_wm_confidence_ema"],
                "cum_real_samples": result.summary["cum_real_samples"],
                "cum_wm_samples": result.summary["cum_wm_samples"],
            }
        )
    return rows


def html_summary_table(results: Sequence[ExperimentResult]) -> str:
    columns = (
        ("mode", "Mode"),
        ("experiment_name", "Experiment"),
        ("last_step", "Last Step"),
        ("best_success", "Best Success"),
        ("final_success", "Final Success"),
        ("best_train_reward", "Best Train Reward"),
        ("final_train_reward", "Final Train Reward"),
        ("final_wm_ratio_wm", "Final WM Ratio"),
        ("final_wm_loss_ema", "Final WM Loss EMA"),
        ("final_wm_confidence_ema", "Final WM Confidence EMA"),
        ("cum_real_samples", "Cum Real Samples"),
        ("cum_wm_samples", "Cum WM Samples"),
    )
    header_html = "".join([f"<th>{title}</th>" for _, title in columns])
    row_html = []
    for row in dashboard_summary_rows(results):
        cells = "".join([f"<td>{format_scalar(row.get(key))}</td>" for key, _ in columns])
        row_html.append(f"<tr>{cells}</tr>")
    return (
        "<table class=\"summary-table\">"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table>"
    )


def render_dashboard_html(
    title: str,
    results: Sequence[ExperimentResult],
    max_points: int,
) -> str:
    charts = chart_specs(results, max_points=max_points)
    grouped_cards: List[str] = []

    for _, group_title in GROUP_ORDER:
        group_charts = [chart for chart in charts if chart["group"] == group_title]
        if not group_charts:
            continue
        cards = []
        for chart in group_charts:
            source_map = ", ".join(
                [
                    f"{series['name']}={series['source_key']}"
                    for series in chart["series"]
                    if series.get("source_key")
                ]
            )
            cards.append(
                "".join(
                    [
                        "<section class=\"card\">",
                        f"<h3>{chart['title']}</h3>",
                        f"<div class=\"chart\" id=\"{chart['id']}\"></div>",
                        f"<p class=\"source\">Source: {source_map or '-'}</p>",
                        "</section>",
                    ]
                )
            )
        grouped_cards.append(
            "".join(
                [
                    f"<section class=\"group\"><h2>{group_title}</h2>",
                    "<div class=\"card-grid\">",
                    "".join(cards),
                    "</div></section>",
                ]
            )
        )

    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "title": title,
        "charts": charts,
    }

    return "".join(
        [
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\">",
            f"<title>{title}</title>",
            "<style>",
            "body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f5f1e8;color:#1f2933;}",
            ".page{max-width:1440px;margin:0 auto;padding:24px 28px 60px 28px;}",
            "h1{margin:0 0 8px 0;font-size:32px;}",
            "h2{margin:32px 0 16px 0;font-size:22px;}",
            "h3{margin:0 0 12px 0;font-size:18px;}",
            ".meta{color:#5b6670;margin-bottom:20px;}",
            ".summary-table{width:100%;border-collapse:collapse;background:#ffffff;border:1px solid #d7d2c8;}",
            ".summary-table th,.summary-table td{padding:10px 12px;border:1px solid #e5dfd5;text-align:left;font-size:14px;}",
            ".summary-table th{background:#efe9de;}",
            ".group{margin-top:28px;}",
            ".card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px;}",
            ".card{background:#fff;border:1px solid #d7d2c8;border-radius:14px;padding:18px;box-shadow:0 8px 24px rgba(31,41,51,0.06);}",
            ".chart{width:100%;min-height:320px;}",
            ".source{margin-top:10px;color:#5b6670;font-size:12px;word-break:break-all;}",
            ".legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;font-size:12px;color:#334155;}",
            ".legend-item{display:flex;align-items:center;gap:6px;}",
            ".legend-dot{width:10px;height:10px;border-radius:50%;display:inline-block;}",
            ".empty{padding:24px;color:#64748b;font-size:14px;}",
            "svg text{font-family:Segoe UI,Arial,sans-serif;}",
            "</style></head><body><div class=\"page\">",
            f"<h1>{title}</h1>",
            f"<p class=\"meta\">Generated at {payload['generated_at']}</p>",
            html_summary_table(results),
            "".join(grouped_cards),
            "</div><script>",
            f"const dashboardData = {json.dumps(payload, ensure_ascii=False)};",
            "function niceTicks(minValue,maxValue,count){",
            "  if(!Number.isFinite(minValue)||!Number.isFinite(maxValue)){return [];}",
            "  if(minValue===maxValue){const pad=Math.abs(minValue||1)*0.1;minValue-=pad;maxValue+=pad;}",
            "  const span=maxValue-minValue;",
            "  const rough=span/Math.max(count,1);",
            "  const power=Math.pow(10,Math.floor(Math.log10(Math.max(rough,1e-9))));",
            "  const scaled=rough/power;",
            "  let step=power;",
            "  if(scaled>5){step=10*power;}else if(scaled>2){step=5*power;}else if(scaled>1){step=2*power;}",
            "  const start=Math.floor(minValue/step)*step;",
            "  const end=Math.ceil(maxValue/step)*step;",
            "  const ticks=[];",
            "  for(let value=start; value<=end+step*0.5; value+=step){ticks.push(Number(value.toFixed(10)));}",
            "  return ticks;",
            "}",
            "function formatTick(value){",
            "  if(Math.abs(value)>=1000&&Math.abs(value-Math.round(value))<1e-9){return String(Math.round(value));}",
            "  if(Math.abs(value)>=100){return value.toFixed(1);}",
            "  if(Math.abs(value)>=1){return value.toFixed(2);}",
            "  return value.toFixed(4);",
            "}",
            "function renderChart(chart){",
            "  const host=document.getElementById(chart.id);",
            "  if(!host){return;}",
            "  const allPoints=chart.series.flatMap(series=>series.points).filter(point=>Number.isFinite(point[0])&&Number.isFinite(point[1]));",
            "  if(!allPoints.length){host.innerHTML='<div class=\"empty\">No data</div>';return;}",
            "  const width=Math.max(host.clientWidth||420,420);",
            "  const height=320;",
            "  const margin={top:18,right:18,bottom:42,left:62};",
            "  const xMin=Math.min(...allPoints.map(point=>point[0]));",
            "  const xMax=Math.max(...allPoints.map(point=>point[0]));",
            "  let yMin=Math.min(...allPoints.map(point=>point[1]));",
            "  let yMax=Math.max(...allPoints.map(point=>point[1]));",
            "  if(yMin===yMax){const pad=Math.abs(yMin||1)*0.1;yMin-=pad;yMax+=pad;}",
            "  const plotWidth=width-margin.left-margin.right;",
            "  const plotHeight=height-margin.top-margin.bottom;",
            "  const xScale=value=>margin.left+((value-xMin)/Math.max(xMax-xMin,1e-9))*plotWidth;",
            "  const yScale=value=>margin.top+plotHeight-((value-yMin)/Math.max(yMax-yMin,1e-9))*plotHeight;",
            "  const yTicks=niceTicks(yMin,yMax,5);",
            "  const xTicks=niceTicks(xMin,xMax,5);",
            "  let svg='';",
            "  svg += `<svg width=\"${width}\" height=\"${height}\" viewBox=\"0 0 ${width} ${height}\">`;",
            "  svg += `<rect x=\"0\" y=\"0\" width=\"${width}\" height=\"${height}\" fill=\"#ffffff\" rx=\"12\" ry=\"12\"></rect>`;",
            "  yTicks.forEach(tick=>{const y=yScale(tick);svg += `<line x1=\"${margin.left}\" y1=\"${y}\" x2=\"${width-margin.right}\" y2=\"${y}\" stroke=\"#e5dfd5\" stroke-width=\"1\"></line>`;svg += `<text x=\"${margin.left-10}\" y=\"${y+4}\" text-anchor=\"end\" font-size=\"11\" fill=\"#64748b\">${formatTick(tick)}</text>`;});",
            "  xTicks.forEach(tick=>{const x=xScale(tick);svg += `<line x1=\"${x}\" y1=\"${margin.top}\" x2=\"${x}\" y2=\"${height-margin.bottom}\" stroke=\"#f0ece3\" stroke-width=\"1\"></line>`;svg += `<text x=\"${x}\" y=\"${height-margin.bottom+18}\" text-anchor=\"middle\" font-size=\"11\" fill=\"#64748b\">${formatTick(tick)}</text>`;});",
            "  svg += `<line x1=\"${margin.left}\" y1=\"${height-margin.bottom}\" x2=\"${width-margin.right}\" y2=\"${height-margin.bottom}\" stroke=\"#334155\" stroke-width=\"1.2\"></line>`;",
            "  svg += `<line x1=\"${margin.left}\" y1=\"${margin.top}\" x2=\"${margin.left}\" y2=\"${height-margin.bottom}\" stroke=\"#334155\" stroke-width=\"1.2\"></line>`;",
            "  chart.series.forEach(series=>{const valid=series.points.filter(point=>Number.isFinite(point[0])&&Number.isFinite(point[1]));if(!valid.length){return;}const pointString=valid.map(point=>`${xScale(point[0])},${yScale(point[1])}`).join(' ');svg += `<polyline fill=\"none\" stroke=\"${series.color}\" stroke-width=\"2.4\" points=\"${pointString}\"></polyline>`;if(valid.length<=80){valid.forEach(point=>{svg += `<circle cx=\"${xScale(point[0])}\" cy=\"${yScale(point[1])}\" r=\"2.5\" fill=\"${series.color}\"></circle>`;});}});",
            "  svg += `<text x=\"${width/2}\" y=\"${height-8}\" text-anchor=\"middle\" font-size=\"12\" fill=\"#475569\">Global Step</text>`;",
            "  svg += '</svg>';",
            "  const legend = '<div class=\"legend\">' + chart.series.map(series => `<span class=\"legend-item\"><span class=\"legend-dot\" style=\"background:${series.color}\"></span>${series.name}</span>`).join('') + '</div>';",
            "  host.innerHTML = svg + legend;",
            "}",
            "dashboardData.charts.forEach(renderChart);",
            "</script></body></html>",
        ]
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_experiment_outputs(result: ExperimentResult, max_points: int) -> None:
    result.analysis_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(result.analysis_dir / "merged_log.csv", result.raw_rows)
    write_rows_csv(result.analysis_dir / "metric_curves.csv", result.metric_rows)
    write_json(
        result.analysis_dir / "experiment_summary.json",
        {
            "summary": result.summary,
            "source_keys": result.source_keys,
            "analysis_dir": str(result.analysis_dir),
            "log_path": str(result.log_path),
        },
    )
    write_text(
        result.analysis_dir / "experiment_summary.md",
        render_experiment_markdown(result),
    )
    write_text(
        result.analysis_dir / "experiment_dashboard.html",
        render_dashboard_html(
            title=f"{result.mode} Dashboard: {result.summary['experiment_name']}",
            results=[result],
            max_points=max_points,
        ),
    )


def comparison_payload(results: Sequence[ExperimentResult]) -> Dict[str, Any]:
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "experiments": [result.summary for result in results],
        "ranking_checks": [
            expected_order_check(results, "best_success"),
            expected_order_check(results, "final_success"),
            expected_order_check(results, "final_train_reward"),
        ],
    }


def write_comparison_outputs(
    results: Sequence[ExperimentResult],
    output_dir: Path,
    max_points: int,
    mirror_to_each_experiment: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = summary_rows(results)
    comparison_md = render_comparison_markdown(results)
    comparison_json = comparison_payload(results)
    comparison_html = render_dashboard_html(
        title="Three-Mode Comparison Dashboard",
        results=results,
        max_points=max_points,
    )

    write_rows_csv(output_dir / "comparison_summary.csv", comparison_rows)
    write_text(output_dir / "comparison_summary.md", comparison_md)
    write_json(output_dir / "comparison_data.json", comparison_json)
    write_text(output_dir / "comparison_dashboard.html", comparison_html)

    if not mirror_to_each_experiment:
        return

    for result in results:
        write_rows_csv(result.analysis_dir / "comparison_summary.csv", comparison_rows)
        write_text(result.analysis_dir / "comparison_summary.md", comparison_md)
        write_json(result.analysis_dir / "comparison_data.json", comparison_json)
        write_text(result.analysis_dir / "comparison_dashboard.html", comparison_html)


def default_output_dir(results: Sequence[ExperimentResult]) -> Path:
    common_root = Path(os.path.commonpath([str(result.experiment_dir) for result in results]))
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "_".join([result.mode.lower() for result in results])
    return common_root / "comparisons" / f"compare_{slug}_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MFRL, MBRL, and MERL experiment logs, then write summaries and dashboards.",
    )
    parser.add_argument("--mfrl", type=str, default=None, help="Experiment dir or log file for MFRL.")
    parser.add_argument("--mbrl", type=str, default=None, help="Experiment dir or log file for MBRL.")
    parser.add_argument("--merl", type=str, default=None, help="Experiment dir or log file for MERL.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional cross-mode output directory. Default: checkpoints/comparisons/<timestamp>.",
    )
    parser.add_argument(
        "--max-html-points",
        type=int,
        default=400,
        help="Max points per rendered line in HTML dashboards.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not mirror cross-mode outputs into each experiment analysis directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    requested: List[Tuple[str, str]] = []
    if args.mfrl:
        requested.append(("MFRL", args.mfrl))
    if args.mbrl:
        requested.append(("MBRL", args.mbrl))
    if args.merl:
        requested.append(("MERL", args.merl))

    if not requested:
        raise SystemExit("At least one of --mfrl, --mbrl, or --merl must be provided.")

    results: List[ExperimentResult] = []
    for mode, input_path in requested:
        result = summarize_experiment(mode=mode, input_path=input_path)
        write_experiment_outputs(result, max_points=args.max_html_points)
        results.append(result)

    results.sort(key=lambda item: MODE_ORDER.index(item.mode) if item.mode in MODE_ORDER else 999)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(results)
    )
    write_comparison_outputs(
        results=results,
        output_dir=output_dir,
        max_points=args.max_html_points,
        mirror_to_each_experiment=not args.no_mirror,
    )

    print("Comparison completed.")
    print(f"Cross-mode outputs: {output_dir}")
    for result in results:
        print(f"Per-experiment outputs: {result.analysis_dir}")


if __name__ == "__main__":
    main()
