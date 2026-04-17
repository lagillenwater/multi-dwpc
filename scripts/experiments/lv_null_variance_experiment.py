#!/usr/bin/env python3
"""LV null-variance analysis over explicit replicate summary artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.lv_replicate_analysis import FEATURE_KEYS, build_b_seed_runs, load_summary_bank  # noqa: E402
from src.replicate_analysis import summarize_feature_variance, summarize_overall_variance  # noqa: E402


def _parse_int_list(arg: str) -> list[int]:
    values = [int(tok.strip()) for tok in str(arg).split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _plot_distribution(feature_df: pd.DataFrame, metric_col: str, y_label: str, title: str, out_path: Path) -> None:
    work = feature_df[["control", "b", metric_col]].dropna().copy()
    if work.empty:
        return
    controls = sorted(work["control"].astype(str).unique().tolist())
    colors = {"permuted": "#1f77b4", "random": "#d62728"}

    fig, axes = plt.subplots(1, len(controls), figsize=(6.5 * len(controls), 5), sharey=True)
    if len(controls) == 1:
        axes = [axes]

    for ax, control in zip(axes, controls):
        subset = work[work["control"].astype(str) == control].copy()
        b_values = sorted(subset["b"].astype(int).unique().tolist())
        data = [subset[subset["b"].astype(int) == b][metric_col].to_numpy(dtype=float) for b in b_values]
        positions = np.arange(1, len(b_values) + 1)
        ax.boxplot(
            data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            boxprops={"facecolor": colors.get(control, "#cccccc"), "alpha": 0.35},
            medianprops={"color": "#111111", "linewidth": 1.4},
            whiskerprops={"color": "#444444"},
            capprops={"color": "#444444"},
        )
        rng = np.random.default_rng(42)
        for pos, vals in zip(positions, data):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.16, 0.16, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos, dtype=float) + jitter,
                vals,
                s=16,
                alpha=0.35,
                color=colors.get(control, "#333333"),
                edgecolors="none",
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([str(b) for b in b_values])
        ax.set_xlabel("B")
        ax.set_title(control)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(y_label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/lv_experiment")
    parser.add_argument("--analysis-output-dir", default=None)
    parser.add_argument("--b-values", default="1,2,5,10,20")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    exp_root = Path(args.analysis_output_dir) if args.analysis_output_dir else output_dir / "lv_null_variance_experiment"
    exp_root.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary_bank(output_dir)
    runs_df = build_b_seed_runs(summary_df, _parse_int_list(args.b_values), _parse_int_list(args.seeds))
    feature_df = summarize_feature_variance(
        runs_df,
        feature_keys=["control", "b", *FEATURE_KEYS],
    )
    overall_df = summarize_overall_variance(
        feature_df,
        overall_keys=["control", "b"],
        runs_df=runs_df,
        replicate_col="seed",
    )

    runs_df.to_csv(exp_root / "all_runs_wide.csv", index=False)
    feature_df.to_csv(exp_root / "feature_variance_summary.csv", index=False)
    overall_df.to_csv(exp_root / "overall_variance_summary.csv", index=False)

    _plot_distribution(
        feature_df,
        metric_col="diff_var",
        y_label="Variance(diff) across seeds",
        title="LV per-feature variance by B",
        out_path=exp_root / "variance_overall_by_group.png",
    )
    _plot_distribution(
        feature_df,
        metric_col="diff_std",
        y_label="SD(diff) across seeds",
        title="LV per-feature SD by B",
        out_path=exp_root / "sd_overall_by_group.png",
    )

    print(f"Saved runs: {exp_root / 'all_runs_wide.csv'}")
    print(f"Saved feature summary: {exp_root / 'feature_variance_summary.csv'}")
    print(f"Saved overall summary: {exp_root / 'overall_variance_summary.csv'}")


if __name__ == "__main__":
    main()
