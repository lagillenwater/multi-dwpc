#!/usr/bin/env python3
"""
Quantify year-based null variance across permutation/random replicates.

This script analyzes DWPC outputs produced by scripts/compute_dwpc_direct.py and
computes variance of (real_mean_dwpc - null_mean_dwpc) across null replicates.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTROL_MAP = {
    "perm": "permuted",
    "random": "random",
}


def _parse_file(path: Path, base_name: str) -> tuple[int, str, int] | None:
    pattern = re.compile(
        rf"^dwpc_{re.escape(base_name)}_(?P<year>\d{{4}})_(?P<kind>real|perm|random)(?:_(?P<rep>\d+))?\.csv$"
    )
    match = pattern.match(path.name)
    if not match:
        return None
    year = int(match.group("year"))
    kind = str(match.group("kind"))
    rep = int(match.group("rep")) if match.group("rep") else 0
    return year, kind, rep


def _group_dwpc_means(path: Path, chunksize: int) -> pd.DataFrame:
    usecols = ["go_id", "metapath", "dwpc"]
    chunk_aggs = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        grouped = (
            chunk.groupby(["go_id", "metapath"], as_index=False)["dwpc"]
            .agg(sum_dwpc="sum", n_pairs="count")
        )
        chunk_aggs.append(grouped)

    if not chunk_aggs:
        return pd.DataFrame(columns=["go_id", "metapath", "mean_dwpc"])

    agg = (
        pd.concat(chunk_aggs, ignore_index=True)
        .groupby(["go_id", "metapath"], as_index=False)
        .agg(sum_dwpc=("sum_dwpc", "sum"), n_pairs=("n_pairs", "sum"))
    )
    agg["mean_dwpc"] = agg["sum_dwpc"] / agg["n_pairs"]
    return agg[["go_id", "metapath", "mean_dwpc"]]


def _collect_diffs(args: argparse.Namespace) -> pd.DataFrame:
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    real_by_year: dict[int, pd.DataFrame] = {}
    control_runs: list[pd.DataFrame] = []

    files = sorted(results_dir.glob("dwpc_*.csv"))
    if not files:
        raise FileNotFoundError(f"No DWPC result files found under {results_dir}")

    for path in files:
        parsed = _parse_file(path, base_name=args.base_name)
        if parsed is None:
            continue

        year, kind, rep = parsed
        print(f"[load] {path.name}")
        grouped = _group_dwpc_means(path, chunksize=args.chunksize)
        grouped["year"] = int(year)

        if kind == "real":
            if year in real_by_year:
                print(f"[warn] Multiple real files for year={year}; overwriting with {path.name}")
            real_by_year[year] = grouped.rename(columns={"mean_dwpc": "real_mean_dwpc"})
            continue

        control = CONTROL_MAP[kind]
        grouped["control"] = control
        grouped["replicate"] = int(rep)
        grouped = grouped.rename(columns={"mean_dwpc": "null_mean_dwpc"})
        control_runs.append(
            grouped[
                ["year", "control", "replicate", "go_id", "metapath", "null_mean_dwpc"]
            ]
        )

    if not real_by_year:
        raise ValueError("No real-year DWPC files were found.")
    if not control_runs:
        raise ValueError("No null DWPC files (perm/random) were found.")

    runs_df = pd.concat(control_runs, ignore_index=True)
    out_frames = []
    for year, group in runs_df.groupby("year", sort=True):
        if year not in real_by_year:
            print(f"[warn] Missing real file for year={year}; skipping null runs for this year.")
            continue

        real_df = real_by_year[year][["year", "go_id", "metapath", "real_mean_dwpc"]]
        merged = group.merge(real_df, on=["year", "go_id", "metapath"], how="inner")
        merged["diff"] = merged["real_mean_dwpc"] - merged["null_mean_dwpc"]
        out_frames.append(merged)

    if not out_frames:
        raise ValueError("No matched (real, null) feature rows were produced.")
    return pd.concat(out_frames, ignore_index=True)


def _feature_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["year", "control", "go_id", "metapath"]
    for key, group in runs_df.groupby(keys, dropna=False):
        diff_vals = group["diff"].to_numpy(dtype=float)
        n = len(diff_vals)
        rows.append(
            {
                "year": int(key[0]),
                "control": str(key[1]),
                "go_id": key[2],
                "metapath": key[3],
                "n_replicates": int(n),
                "diff_mean": float(np.nanmean(diff_vals)),
                "diff_std": float(np.nanstd(diff_vals, ddof=1)) if n > 1 else np.nan,
                "diff_var": float(np.nanvar(diff_vals, ddof=1)) if n > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "control", "metapath", "go_id"])


def _overall_summary(feature_df: pd.DataFrame, runs_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        feature_df.groupby(["year", "control"], as_index=False)
        .agg(
            n_features=("metapath", "size"),
            mean_diff_std=("diff_std", "mean"),
            mean_diff_var=("diff_var", "mean"),
            median_diff_std=("diff_std", "median"),
            median_diff_var=("diff_var", "median"),
        )
        .sort_values(["year", "control"])
    )
    reps = (
        runs_df.groupby(["year", "control"], as_index=False)["replicate"]
        .nunique()
        .rename(columns={"replicate": "n_replicates"})
    )
    return out.merge(reps, on=["year", "control"], how="left")


def _plot_overall(
    overall_df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    years = sorted(overall_df["year"].dropna().astype(int).unique().tolist())
    controls = sorted(overall_df["control"].dropna().astype(str).unique().tolist())
    if not years or not controls:
        return

    width = 0.8 / max(len(controls), 1)
    x = np.arange(len(years))
    colors = {"permuted": "#1f77b4", "random": "#d62728"}

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, control in enumerate(controls):
        vals = []
        for year in years:
            sub = overall_df[
                (overall_df["year"].astype(int) == int(year))
                & (overall_df["control"].astype(str) == control)
            ]
            vals.append(float(sub[y_col].iloc[0]) if not sub.empty else np.nan)
        ax.bar(
            x + (i - (len(controls) - 1) / 2) * width,
            vals,
            width=width,
            label=control,
            color=colors.get(control),
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Control")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_sd(feature_df: pd.DataFrame, output_dir: Path, max_features: int) -> None:
    for (year, control), subset in feature_df.groupby(["year", "control"], dropna=False):
        subset = subset.dropna(subset=["diff_std", "diff_var"]).copy()
        if subset.empty:
            continue

        subset["feature_label"] = (
            subset["metapath"].astype(str) + " | " + subset["go_id"].astype(str)
        )
        top_features = (
            subset.sort_values("diff_var", ascending=False)
            .head(max_features)["feature_label"]
            .tolist()
        )
        top_df = subset[subset["feature_label"].isin(top_features)].copy()
        if top_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5.5))
        top_df = top_df.sort_values("diff_var", ascending=False)
        ax.bar(
            np.arange(len(top_df)),
            top_df["diff_std"].to_numpy(dtype=float),
            color="#1f77b4" if str(control) == "permuted" else "#d62728",
            alpha=0.9,
        )
        ax.set_xticks(np.arange(len(top_df)))
        ax.set_xticklabels(top_df["feature_label"].tolist(), rotation=75, ha="right", fontsize=7)
        ax.set_xlabel("Feature (metapath | GO)")
        ax.set_ylabel("SD(diff) across null replicates")
        ax.set_title(f"Top feature SD ({control}, {int(year)})")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"sd_by_feature_{control}_{int(year)}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    exp_dir = output_dir / "year_null_variance_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)

    runs_df = _collect_diffs(args)
    feature_df = _feature_summary(runs_df)
    overall_df = _overall_summary(feature_df, runs_df)

    min_rep = int(args.min_replicates)
    low_rep_groups = overall_df[overall_df["n_replicates"] < min_rep]
    if not low_rep_groups.empty:
        print(
            f"[warn] Some (year, control) groups have fewer than {min_rep} replicates. "
            "Variance estimates may be unstable."
        )
        print(low_rep_groups[["year", "control", "n_replicates"]].to_string(index=False))

    runs_df.to_csv(exp_dir / "all_runs_long.csv", index=False)
    feature_df.to_csv(exp_dir / "feature_variance_summary.csv", index=False)
    overall_df.to_csv(exp_dir / "overall_variance_summary.csv", index=False)

    _plot_overall(
        overall_df=overall_df,
        y_col="mean_diff_var",
        y_label="Mean variance of diff across features",
        title="Year Null Variance (real - null) by control",
        output_path=exp_dir / "variance_overall_by_group.png",
    )
    _plot_overall(
        overall_df=overall_df,
        y_col="mean_diff_std",
        y_label="Mean SD of diff across features",
        title="Year Null SD (real - null) by control",
        output_path=exp_dir / "sd_overall_by_group.png",
    )
    _plot_feature_sd(feature_df=feature_df, output_dir=exp_dir, max_features=args.max_feature_lines)

    print("\n[done] Year null variance experiment complete.")
    print(f"  Results dir: {Path(args.results_dir)}")
    print(f"  Output dir: {exp_dir}")
    print(
        "  Groups analyzed: "
        f"{overall_df[['year', 'control']].drop_duplicates().shape[0]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute variance of year-based real-vs-null DWPC contrasts across null replicates."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="output/dwpc_direct/all_GO_positive_growth/results",
        help="Directory with dwpc_*.csv result files from compute_dwpc_direct.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/year_null_variance_exp",
        help="Output workspace directory.",
    )
    parser.add_argument(
        "--base-name",
        default="all_GO_positive_growth",
        help="Dataset base name used in DWPC result filenames.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="CSV chunk size for grouped DWPC aggregation.",
    )
    parser.add_argument(
        "--min-replicates",
        type=int,
        default=2,
        help="Warn when a (year, control) group has fewer replicates than this.",
    )
    parser.add_argument(
        "--max-feature-lines",
        type=int,
        default=12,
        help="Maximum number of per-group features to show in SD bar plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args=args)


if __name__ == "__main__":
    main()
