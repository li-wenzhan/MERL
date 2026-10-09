"""Legacy score-per-trajectory diagnostics, using score per trajectory."""

import json
import math
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_constant(x):
    # Handle NaN / Infinity in legacy logs.
    if x == "NaN":
        return float("nan")
    if x == "Infinity":
        return float("inf")
    if x == "-Infinity":
        return float("-inf")
    raise ValueError(f"Unknown constant: {x}")


def load_and_merge_log(log_path: str) -> pd.DataFrame:
    """
    Merge records sharing the same step, such as an evaluation-only record
    followed by a complete training record.
    """
    merged = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue

            timestamp, step_str, json_str = parts
            try:
                step = int(step_str)
                data = json.loads(json_str, parse_constant=parse_constant)
            except Exception:
                continue

            if step not in merged:
                merged[step] = {
                    "timestamp": timestamp,
                    "step": step,
                }

            merged[step].update(data)

    if not merged:
        raise ValueError("No valid log records were parsed.")

    df = pd.DataFrame([merged[k] for k in sorted(merged.keys())])
    df = df.sort_values("step").reset_index(drop=True)
    return df


def build_se_curve(
    df: pd.DataFrame,
    score_key: str = "val/success_rate/all",
    real_key: str = "rollout/real_num_samples",
    wm_key: str = "wm/num_wm_sample",
):
    """
    Compute legacy diagnostics:
    1) cumulative_real_samples
    2) cumulative_total_samples
    3) SE_real = score / cumulative_real_samples
    4) SE_total = score / cumulative_total_samples
    """

    # Treat missing legacy sample counts as zero.
    real_source = next(
        (
            key
            for key in (
                real_key,
                "rollout/real_num_samples",
                "wm/num_real_sample",
                "env/real_sample_actual",
                "env/real_sample_target",
            )
            if key in df.columns
        ),
        None,
    )
    real_samples = pd.to_numeric(
        df[real_source] if real_source is not None else pd.Series(0, index=df.index),
        errors="coerce",
    ).fillna(0)

    wm_samples = pd.to_numeric(
        df[wm_key] if wm_key in df.columns else pd.Series(0, index=df.index),
        errors="coerce",
    ).fillna(0)

    # With no WM counts, use the real branch only.
    total_samples = real_samples + wm_samples

    df = df.copy()
    df["real_samples"] = real_samples
    df["wm_samples"] = wm_samples
    df["total_samples"] = total_samples

    df["cum_real_samples"] = df["real_samples"].cumsum()
    df["cum_total_samples"] = df["total_samples"].cumsum()

    # The default score is the evaluation success rate.
    score_source = next(
        (
            key
            for key in (
                score_key,
                "val/success_rate/all",
                "success_rate/all",
                "val/test_score/all",
                "test_score/all",
            )
            if key in df.columns
        ),
        None,
    )
    df["score"] = pd.to_numeric(
        df[score_source] if score_source is not None else np.nan,
        errors="coerce",
    )

    # Compute the diagnostic only at steps with evaluation results.
    eval_df = df[df["score"].notna()].copy()

    eval_df["SE_real"] = eval_df["score"] / eval_df["cum_real_samples"].replace(0, np.nan)
    eval_df["SE_total"] = eval_df["score"] / eval_df["cum_total_samples"].replace(0, np.nan)

    return df, eval_df


def print_summary(eval_df: pd.DataFrame):
    if eval_df.empty:
        print("No evaluation scores found; sample efficiency is unavailable.")
        return

    print("\n=== Legacy sample-efficiency diagnostics ===")
    print(eval_df[
        ["step", "score", "cum_real_samples", "cum_total_samples", "SE_real", "SE_total"]
    ].to_string(index=False))

    best_idx = eval_df["SE_real"].idxmax()
    last_idx = eval_df.index[-1]

    print("\n=== Key results ===")
    print(
        f"Best SE_real: step={int(eval_df.loc[best_idx, 'step'])}, "
        f"score={eval_df.loc[best_idx, 'score']:.6f}, "
        f"cum_real={eval_df.loc[best_idx, 'cum_real_samples']:.0f}, "
        f"SE_real={eval_df.loc[best_idx, 'SE_real']:.6f}"
    )
    print(
        f"Final SE_real: step={int(eval_df.loc[last_idx, 'step'])}, "
        f"score={eval_df.loc[last_idx, 'score']:.6f}, "
        f"cum_real={eval_df.loc[last_idx, 'cum_real_samples']:.0f}, "
        f"SE_real={eval_df.loc[last_idx, 'SE_real']:.6f}"
    )


def plot_se_curve(eval_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostic versus global step.
    plt.figure(figsize=(8, 5))
    plt.plot(eval_df["step"], eval_df["SE_real"], marker="o", label="SE_real")
    plt.plot(eval_df["step"], eval_df["SE_total"], marker="s", label="SE_total")
    plt.xlabel("global_step")
    plt.ylabel("Sample Efficiency")
    plt.title("SE Curve vs Global Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "se_curve_vs_step.png", dpi=200)
    plt.close()

    # Performance versus real trajectory count.
    plt.figure(figsize=(8, 5))
    plt.plot(eval_df["cum_real_samples"], eval_df["score"], marker="o")
    plt.xlabel("Cumulative Real Samples")
    plt.ylabel("Score / SR")
    plt.title("Performance vs Real Samples")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "performance_vs_real_samples.png", dpi=200)
    plt.close()

    # Diagnostic versus real trajectory count.
    plt.figure(figsize=(8, 5))
    plt.plot(eval_df["cum_real_samples"], eval_df["SE_real"], marker="o")
    plt.xlabel("Cumulative Real Samples")
    plt.ylabel("SE_real")
    plt.title("SE Curve vs Real Samples")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "se_curve_vs_real_samples.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str, help="Training log path")
    parser.add_argument(
        "--score_key",
        type=str,
        default="val/success_rate/all",
        help="Score field; defaults to val/success_rate/all",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="se_outputs",
        help="Output directory",
    )
    args = parser.parse_args()

    df = load_and_merge_log(args.log_path)
    full_df, eval_df = build_se_curve(
        df,
        score_key=args.score_key,
        real_key="rollout/real_num_samples",
        wm_key="wm/num_wm_sample",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(out_dir / "merged_log.csv", index=False)
    eval_df.to_csv(out_dir / "se_curve.csv", index=False)

    print_summary(eval_df)
    plot_se_curve(eval_df, out_dir)

    print(f"\nOutputs saved to: {out_dir.resolve()}")
    print("  - merged_log.csv")
    print("  - se_curve.csv")
    print("  - se_curve_vs_step.png")
    print("  - performance_vs_real_samples.png")
    print("  - se_curve_vs_real_samples.png")


if __name__ == "__main__":
    main()
