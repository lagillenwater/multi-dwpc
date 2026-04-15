#!/usr/bin/env python3
"""Select optimal B value by finding where variance/rank metrics stabilize.

For each per-entity-per-metric curve, finds the smallest sampled B at which the
metric has reached (and stays at) a configurable fraction of the way from its
first-sampled value to its asymptote (last-sampled value). Aggregates those
stabilization points across metrics and entities via median (default) or max.

Metrics used:
- Null variance (decreasing): diff_std, diff_var
- Rank stability (increasing): mean_spearman_rho, mean_topk_jaccard_5, mean_rbo

Outputs chosen_b.json with selected B and rationale.

Usage:
    python scripts/select_optimal_b.py --analysis-type lv --output-dir output/lv_full_analysis
    python scripts/select_optimal_b.py --analysis-type year --output-dir output/year_full_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save_figure(fig, out_path_stem: Path) -> None:
    """Save a figure as both .pdf and .png using the same stem."""
    out_path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out_path_stem.with_suffix(ext), bbox_inches="tight", dpi=200)


def _compute_stabilization_point(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_col: str,
    increasing: bool,
    threshold: float,
) -> pd.DataFrame:
    """Return the smallest B at which a metric has stabilized for each group.

    Stabilization = the metric is within (1 - threshold) of its asymptote, and
    stays there for every subsequent sampled B. We scan backwards from the
    final B, treating the last-sampled value as the asymptote, and look for
    the last point that violates the target so we can pick one beyond it.

    Args:
        df: DataFrame with a "b" column and the metric column.
        group_cols: Grouping columns (e.g., ["control", "lv_id"]).
        metric_col: Name of the metric column.
        increasing: True if the metric grows toward its asymptote (stability);
            False if it shrinks toward its asymptote (variance).
        threshold: Fraction of the baseline-to-asymptote distance that must be
            covered. 0.9 means "within 10% of asymptote".

    Returns:
        DataFrame with one row per group containing the stabilization point.
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

        b_vals = curve["b"].astype(float).to_numpy()
        y = curve[metric_col].astype(float).to_numpy()

        baseline = float(y[0])
        asymptote = float(y[-1])

        if asymptote == baseline:
            chosen_idx = 0
            target = baseline
        else:
            if increasing:
                target = baseline + threshold * (asymptote - baseline)
                passes = y >= target
            else:
                target = baseline - threshold * (baseline - asymptote)
                passes = y <= target

            if passes.all():
                chosen_idx = 0
            elif not passes[-1]:
                # Asymptote itself fails target -- curve hasn't stabilized;
                # fall back to the last sampled B.
                chosen_idx = len(y) - 1
            else:
                failures = np.flatnonzero(~passes)
                chosen_idx = int(failures[-1]) + 1

        row = {
            "metric": str(metric_col),
            "stabilization_b": int(b_vals[chosen_idx]),
            "stabilization_value": float(y[chosen_idx]),
            "baseline_value": baseline,
            "asymptote_value": asymptote,
            "target_value": float(target),
            "threshold": float(threshold),
        }
        if isinstance(group_key, tuple):
            for col, val in zip(group_cols, group_key):
                row[col] = str(val)
        else:
            row[group_cols[0]] = str(group_key)

        rows.append(row)

    return pd.DataFrame(rows)


def _plot_stabilization_curves(
    raw_df: pd.DataFrame,
    group_cols: list[str],
    metric_col: str,
    increasing: bool,
    threshold: float,
    title: str,
    out_stem: Path,
) -> None:
    """Grid of curves (one panel per entity x control), marking each stabilization point."""
    if raw_df.empty or metric_col not in raw_df.columns:
        return

    control_col, entity_col = group_cols
    controls = sorted(raw_df[control_col].astype(str).unique().tolist())
    entities = sorted(raw_df[entity_col].astype(str).unique().tolist())
    if not controls or not entities:
        return

    n_cols = min(len(entities), 4)
    n_rows = (len(entities) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        sharex=True, squeeze=False,
    )

    control_colors = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(controls)}

    for idx, entity in enumerate(entities):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        for control in controls:
            sub = raw_df[
                (raw_df[control_col].astype(str) == control)
                & (raw_df[entity_col].astype(str) == entity)
            ]
            if sub.empty:
                continue
            curve = (
                sub.groupby("b", as_index=False)[metric_col]
                .mean()
                .sort_values("b")
                .reset_index(drop=True)
            )
            if len(curve) < 2:
                continue
            b_vals = curve["b"].astype(float).to_numpy()
            y = curve[metric_col].astype(float).to_numpy()
            color = control_colors[control]
            ax.plot(b_vals, y, marker="o", color=color, linewidth=1.6, label=control)

            if len(curve) >= 3:
                baseline = float(y[0])
                asymptote = float(y[-1])
                if asymptote != baseline:
                    if increasing:
                        target = baseline + threshold * (asymptote - baseline)
                        passes = y >= target
                    else:
                        target = baseline - threshold * (baseline - asymptote)
                        passes = y <= target
                    if passes.all():
                        chosen_idx = 0
                    elif not passes[-1]:
                        chosen_idx = len(y) - 1
                    else:
                        chosen_idx = int(np.flatnonzero(~passes)[-1]) + 1
                    ax.axhline(target, color=color, linestyle=":", linewidth=0.9, alpha=0.6)
                    ax.scatter(
                        [b_vals[chosen_idx]], [y[chosen_idx]],
                        marker="*", s=180, color=color,
                        edgecolors="black", linewidths=0.8, zorder=5,
                    )
        ax.set_title(str(entity), fontsize=10)
        ax.grid(alpha=0.25)
        if r == n_rows - 1:
            ax.set_xlabel("B")
        if c == 0:
            ax.set_ylabel(metric_col)

    # Blank unused axes
    for k in range(len(entities), n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), title="Control")
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(fig, out_stem)
    plt.close(fig)


def _plot_stabilization_distribution(
    summary_df: pd.DataFrame,
    chosen_b: int,
    out_stem: Path,
) -> None:
    """Histogram of per-curve stabilization_b, colored by metric, with chosen B marked."""
    if summary_df.empty:
        return

    metrics = sorted(summary_df["metric"].unique().tolist())
    all_bs = sorted(summary_df["stabilization_b"].astype(int).unique().tolist())
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.get_cmap("tab10")
    for mi, metric in enumerate(metrics):
        vals = summary_df[summary_df["metric"] == metric]["stabilization_b"].astype(int)
        counts = vals.value_counts().reindex(all_bs, fill_value=0).sort_index()
        xs = np.arange(len(all_bs)) + mi * width - (len(metrics) - 1) * width / 2
        ax.bar(xs, counts.values, width=width, color=cmap(mi % 10), label=metric)

    ax.set_xticks(np.arange(len(all_bs)))
    ax.set_xticklabels([str(b) for b in all_bs])
    ax.set_xlabel("Stabilization B")
    ax.set_ylabel("Number of curves")
    ax.axvline(
        all_bs.index(int(chosen_b)) if int(chosen_b) in all_bs else -1,
        color="black", linestyle="--", linewidth=1.4, label=f"Chosen B = {chosen_b}",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Distribution of per-curve stabilization B values")
    fig.tight_layout()
    _save_figure(fig, out_stem)
    plt.close(fig)


def load_variance_stabilization(
    variance_dir: Path, analysis_type: str, threshold: float
) -> pd.DataFrame:
    """Compute stabilization points for null-variance metrics."""
    summary_path = variance_dir / "feature_variance_summary.csv"
    group_cols = ["control", "lv_id"] if analysis_type == "lv" else ["control", "go_id"]

    if not summary_path.exists():
        print(f"Warning: {summary_path} not found, skipping variance metrics")
        return pd.DataFrame()

    df = pd.read_csv(summary_path)

    frames = []
    for metric in ["diff_std", "diff_var"]:
        if metric in df.columns:
            frames.append(
                _compute_stabilization_point(
                    df, group_cols, metric, increasing=False, threshold=threshold
                )
            )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_rank_stability_stabilization(
    rank_dir: Path, analysis_type: str, threshold: float
) -> pd.DataFrame:
    """Compute stabilization points for rank-stability metrics."""
    if analysis_type == "lv":
        summary_path = rank_dir / "lv_stability_summary.csv"
        group_cols = ["control", "lv_id"]
    else:
        summary_path = rank_dir / "go_stability_summary.csv"
        group_cols = ["control", "go_id"]

    if summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        legacy_path = rank_dir / "metapath_pairwise_metrics.csv"
        if legacy_path.exists():
            df = pd.read_csv(legacy_path)
            if "control" not in df.columns:
                df["control"] = "combined"
        else:
            print(f"Warning: {summary_path} not found, skipping rank stability metrics")
            return pd.DataFrame()

    frames = []
    for metric in ["mean_spearman_rho", "mean_topk_jaccard_5", "mean_rbo"]:
        if metric in df.columns:
            frames.append(
                _compute_stabilization_point(
                    df, group_cols, metric, increasing=True, threshold=threshold
                )
            )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_optimal_b(
    variance_dir: Path,
    rank_dir: Path,
    analysis_type: str,
    aggregation: str,
    threshold: float,
) -> tuple[int, pd.DataFrame]:
    """Select optimal B by aggregating per-curve stabilization points.

    Returns:
        Tuple of (chosen_b, stabilization_summary_df).
    """
    variance_points = load_variance_stabilization(variance_dir, analysis_type, threshold)
    rank_points = load_rank_stability_stabilization(rank_dir, analysis_type, threshold)

    all_points = pd.concat([variance_points, rank_points], ignore_index=True)

    if all_points.empty:
        raise ValueError(
            "No stabilization data found. Run variance and rank stability experiments first."
        )

    b_values = all_points["stabilization_b"].dropna()
    min_b = int(b_values.min())
    # interpolation="higher" guarantees the reported median is a B value that
    # at least one curve actually stabilized at (i.e., a sampled B). With the
    # default linear interpolation, an even-sized sample with tied middle
    # groups can yield a B that was never tested (e.g. median of [...20s..., ...30s...]
    # becomes 25). The higher variant picks the upper-middle observation,
    # which is also the conservative choice for stability coverage.
    median_b = int(b_values.quantile(0.5, interpolation="higher"))
    max_b = int(b_values.max())

    chosen_b = median_b if aggregation == "median" else max_b

    print(
        f"Stabilization B (threshold={threshold}): "
        f"min={min_b}, median={median_b}, max={max_b}"
    )
    print(f"Chosen B ({aggregation}): {chosen_b}")

    return chosen_b, all_points


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
        help="Output directory for chosen_b.json and stabilization_summary.csv",
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "max"],
        default="median",
        help="How to aggregate per-curve stabilization B values (default: median)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help=(
            "Fraction of the baseline-to-asymptote distance required to declare "
            "stabilization (default: 0.9 = within 10%% of asymptote)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.analysis_type == "lv":
        variance_dir = Path(args.variance_dir or "output/lv_experiment/lv_null_variance_experiment")
        rank_dir = Path(args.rank_dir or "output/lv_experiment/lv_rank_stability_experiment")
    else:
        variance_dir = Path(
            args.variance_dir or "output/year_null_variance_exp/year_null_variance_experiment"
        )
        rank_dir = Path(
            args.rank_dir or "output/year_rank_stability_exp/year_rank_stability_experiment"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chosen_b, summary = select_optimal_b(
        variance_dir, rank_dir, args.analysis_type, args.aggregation, args.threshold
    )

    summary_path = output_dir / "stabilization_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved stabilization summary: {summary_path}")

    stab_b_counts = (
        summary["stabilization_b"].value_counts().sort_index().astype(int).to_dict()
    )
    stab_b_counts = {str(int(k)): int(v) for k, v in stab_b_counts.items()}

    n_curves = len(summary)
    curves_at_max_sampled_b = int(
        (summary["stabilization_b"] == summary["stabilization_b"].max()).sum()
    )

    result = {
        "chosen_b": chosen_b,
        "aggregation": args.aggregation,
        "threshold": args.threshold,
        "analysis_type": args.analysis_type,
        "min_stabilization_b": int(summary["stabilization_b"].min()),
        "median_stabilization_b": int(
            summary["stabilization_b"].quantile(0.5, interpolation="higher")
        ),
        "max_stabilization_b": int(summary["stabilization_b"].max()),
        "stabilization_b_counts": stab_b_counts,
        "n_curves": n_curves,
        "n_curves_at_max_sampled_b": curves_at_max_sampled_b,
        "metrics_used": sorted(summary["metric"].unique().tolist()),
    }

    chosen_path = output_dir / "chosen_b.json"
    with open(chosen_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved chosen B: {chosen_path}")
    print(f"\nChosen B = {chosen_b}")

    # ---- Plots ----
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    entity_col = "lv_id" if args.analysis_type == "lv" else "go_id"
    group_cols = ["control", entity_col]

    variance_summary_path = variance_dir / "feature_variance_summary.csv"
    if variance_summary_path.exists():
        variance_raw = pd.read_csv(variance_summary_path)
        for metric in ("diff_std", "diff_var"):
            if metric in variance_raw.columns:
                _plot_stabilization_curves(
                    variance_raw,
                    group_cols=group_cols,
                    metric_col=metric,
                    increasing=False,
                    threshold=args.threshold,
                    title=f"Null-variance curves: {metric} (stabilization @ {int(args.threshold * 100)}%)",
                    out_stem=plots_dir / f"stabilization_curves_{metric}",
                )

    rank_summary_path = (
        rank_dir / ("lv_stability_summary.csv" if args.analysis_type == "lv" else "go_stability_summary.csv")
    )
    if rank_summary_path.exists():
        rank_raw = pd.read_csv(rank_summary_path)
    elif (rank_dir / "metapath_pairwise_metrics.csv").exists():
        rank_raw = pd.read_csv(rank_dir / "metapath_pairwise_metrics.csv")
        if "control" not in rank_raw.columns:
            rank_raw["control"] = "combined"
    else:
        rank_raw = pd.DataFrame()

    if not rank_raw.empty:
        for metric in ("mean_spearman_rho", "mean_topk_jaccard_5", "mean_rbo"):
            if metric in rank_raw.columns:
                _plot_stabilization_curves(
                    rank_raw,
                    group_cols=group_cols,
                    metric_col=metric,
                    increasing=True,
                    threshold=args.threshold,
                    title=f"Rank-stability curves: {metric} (stabilization @ {int(args.threshold * 100)}%)",
                    out_stem=plots_dir / f"stabilization_curves_{metric}",
                )

    _plot_stabilization_distribution(
        summary_df=summary,
        chosen_b=chosen_b,
        out_stem=plots_dir / "stabilization_b_distribution",
    )

    print(f"Saved plots: {plots_dir}")


if __name__ == "__main__":
    main()
