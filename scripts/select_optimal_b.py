#!/usr/bin/env python3
"""Select optimal B value using elbow detection on variance and rank stability metrics.

Aggregates elbow points from:
- Null variance: diff_std, diff_var (decreasing metrics)
- Rank stability: mean_spearman_rho, mean_topk_jaccard_5 (increasing metrics)

Outputs chosen_b.json with selected B and rationale.

Usage:
    python scripts/select_optimal_b.py --analysis-type lv --output-dir output/lv_full_analysis
    python scripts/select_optimal_b.py --analysis-type year --output-dir output/year_full_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _compute_elbow(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_col: str,
    increasing: bool,
) -> pd.DataFrame:
    """Compute elbow point for a metric using perpendicular distance method.

    Args:
        df: DataFrame with B values and metric values
        group_cols: Columns to group by (e.g., ["control", "lv_id"])
        metric_col: Column containing metric values
        increasing: True if higher values are better (stability), False if lower (variance)

    Returns:
        DataFrame with elbow B value for each group
    """
    rows: list[dict] = []

    for group_key, group in df.groupby(group_cols, sort=True):
        curve = (
            group.groupby("b", as_index=False)[metric_col]
            .mean()
            .sort_values("b")
            .reset_index(drop=True)
        )
        if len(curve) < 3:
            continue

        x = curve["b"].astype(float).to_numpy()
        y = curve[metric_col].astype(float).to_numpy()

        # Normalize x to [0, 1]
        x_range = x.max() - x.min()
        x_norm = (x - x.min()) / x_range if x_range > 0 else np.zeros_like(x)

        # Normalize y to [0, 1]
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        y_range = y_max - y_min
        y_norm = (y - y_min) / y_range if y_range > 0 else np.zeros_like(y)

        # For decreasing metrics, flip so elbow detection works
        if not increasing:
            y_norm = 1.0 - y_norm

        # Perpendicular distance from curve to straight line
        line = np.linspace(y_norm[0], y_norm[-1], len(y_norm))
        distance = y_norm - line
        idx = int(np.argmax(distance))

        row = {
            "metric": str(metric_col),
            "elbow_b": int(curve["b"].iloc[idx]),
            "elbow_value": float(curve[metric_col].iloc[idx]),
            "elbow_distance": float(distance[idx]),
        }
        # Add group columns
        if isinstance(group_key, tuple):
            for col, val in zip(group_cols, group_key):
                row[col] = str(val)
        else:
            row[group_cols[0]] = str(group_key)

        rows.append(row)

    return pd.DataFrame(rows)


def load_variance_elbows(variance_dir: Path, analysis_type: str) -> pd.DataFrame:
    """Load and compute elbows from null variance experiment."""
    if analysis_type == "lv":
        summary_path = variance_dir / "feature_variance_summary.csv"
        group_cols = ["control", "lv_id"]
    else:
        summary_path = variance_dir / "feature_variance_summary.csv"
        group_cols = ["control", "go_id"]

    if not summary_path.exists():
        print(f"Warning: {summary_path} not found, skipping variance elbows")
        return pd.DataFrame()

    df = pd.read_csv(summary_path)

    elbows = []
    for metric in ["diff_std", "diff_var"]:
        if metric in df.columns:
            elbow_df = _compute_elbow(df, group_cols, metric, increasing=False)
            elbows.append(elbow_df)

    return pd.concat(elbows, ignore_index=True) if elbows else pd.DataFrame()


def load_rank_stability_elbows(rank_dir: Path, analysis_type: str) -> pd.DataFrame:
    """Load and compute elbows from rank stability experiment."""
    if analysis_type == "lv":
        summary_path = rank_dir / "lv_stability_summary.csv"
        group_cols = ["control", "lv_id"]
    else:
        summary_path = rank_dir / "go_stability_summary.csv"
        group_cols = ["control", "go_id"]

    if not summary_path.exists():
        # Try legacy path
        legacy_path = rank_dir / "metapath_pairwise_metrics.csv"
        if legacy_path.exists():
            df = pd.read_csv(legacy_path)
            if "control" not in df.columns:
                df["control"] = "combined"
        else:
            print(f"Warning: {summary_path} not found, skipping rank stability elbows")
            return pd.DataFrame()
    else:
        df = pd.read_csv(summary_path)

    elbows = []
    for metric in ["mean_spearman_rho", "mean_topk_jaccard_5", "mean_rbo"]:
        if metric in df.columns:
            elbow_df = _compute_elbow(df, group_cols, metric, increasing=True)
            elbows.append(elbow_df)

    return pd.concat(elbows, ignore_index=True) if elbows else pd.DataFrame()


def select_optimal_b(
    variance_dir: Path,
    rank_dir: Path,
    analysis_type: str,
    aggregation: str = "median",
) -> tuple[int, pd.DataFrame]:
    """Select optimal B by aggregating elbows from variance and rank stability.

    Args:
        variance_dir: Directory containing null variance experiment results
        rank_dir: Directory containing rank stability experiment results
        analysis_type: "lv" or "year"
        aggregation: "median" or "max"

    Returns:
        Tuple of (chosen_b, elbow_summary_df)
    """
    variance_elbows = load_variance_elbows(variance_dir, analysis_type)
    rank_elbows = load_rank_stability_elbows(rank_dir, analysis_type)

    all_elbows = pd.concat([variance_elbows, rank_elbows], ignore_index=True)

    if all_elbows.empty:
        raise ValueError("No elbow data found. Run variance and rank stability experiments first.")

    # Compute statistics
    elbow_b_values = all_elbows["elbow_b"].dropna()
    min_b = int(elbow_b_values.min())
    median_b = int(round(elbow_b_values.median()))
    max_b = int(elbow_b_values.max())

    # Select based on aggregation method
    if aggregation == "median":
        chosen_b = median_b
    else:  # max
        chosen_b = max_b

    print(f"Elbow B values: min={min_b}, median={median_b}, max={max_b}")
    print(f"Chosen B ({aggregation}): {chosen_b}")

    return chosen_b, all_elbows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-type",
        choices=["lv", "year"],
        required=True,
        help="Type of analysis (lv or year)",
    )
    parser.add_argument(
        "--variance-dir",
        help="Directory containing null variance experiment results",
    )
    parser.add_argument(
        "--rank-dir",
        help="Directory containing rank stability experiment results",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for chosen_b.json and elbow_summary.csv",
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "max"],
        default="median",
        help="How to aggregate elbow B values (default: median)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set default directories based on analysis type
    if args.analysis_type == "lv":
        variance_dir = Path(args.variance_dir or "output/lv_experiment/lv_null_variance_experiment")
        rank_dir = Path(args.rank_dir or "output/lv_experiment/lv_rank_stability_experiment")
    else:
        variance_dir = Path(args.variance_dir or "output/year_null_variance_exp/year_null_variance_experiment")
        rank_dir = Path(args.rank_dir or "output/year_rank_stability_exp/year_rank_stability_experiment")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select optimal B
    chosen_b, elbow_summary = select_optimal_b(
        variance_dir, rank_dir, args.analysis_type, args.aggregation
    )

    # Save elbow summary
    elbow_path = output_dir / "elbow_summary.csv"
    elbow_summary.to_csv(elbow_path, index=False)
    print(f"Saved elbow summary: {elbow_path}")

    # Save chosen B as JSON
    result = {
        "chosen_b": chosen_b,
        "aggregation": args.aggregation,
        "analysis_type": args.analysis_type,
        "min_elbow_b": int(elbow_summary["elbow_b"].min()),
        "median_elbow_b": int(round(elbow_summary["elbow_b"].median())),
        "max_elbow_b": int(elbow_summary["elbow_b"].max()),
        "n_elbows": len(elbow_summary),
        "metrics_used": sorted(elbow_summary["metric"].unique().tolist()),
    }

    chosen_path = output_dir / "chosen_b.json"
    with open(chosen_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved chosen B: {chosen_path}")
    print(f"\nChosen B = {chosen_b}")


if __name__ == "__main__":
    main()
