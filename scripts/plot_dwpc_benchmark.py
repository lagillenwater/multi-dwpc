#!/usr/bin/env python3
"""
Plot runtime comparisons from benchmark_dwpc_methods.py output.
"""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "method",
    "metapath",
    "n_pairs",
    "time_seconds",
    "time_per_pair_ms",
    "errors",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method"] = out["method"].astype(str).str.strip()
    out["metapath"] = out["metapath"].astype(str).str.strip()
    out["n_pairs"] = pd.to_numeric(out["n_pairs"], errors="coerce")
    out["time_seconds"] = pd.to_numeric(out["time_seconds"], errors="coerce")
    out["time_per_pair_ms"] = pd.to_numeric(out["time_per_pair_ms"], errors="coerce")
    out["errors"] = pd.to_numeric(out["errors"], errors="coerce").fillna(0).astype(int)
    out = out.dropna(subset=["n_pairs", "time_per_pair_ms"]).copy()
    out["n_pairs"] = out["n_pairs"].astype(int)
    return out


def _plot_per_metapath(df: pd.DataFrame, output_dir: Path) -> None:
    metapaths = sorted(df["metapath"].unique())
    if not metapaths:
        return

    methods = ["Direct", "API"]
    method_colors = {"Direct": "#1f77b4", "API": "#d62728"}
    n_cols = min(3, len(metapaths))
    n_rows = int(ceil(len(metapaths) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for idx, metapath in enumerate(metapaths):
        ax = axes_flat[idx]
        subset = df[df["metapath"] == metapath].copy()
        for method in methods:
            method_df = subset[subset["method"] == method].sort_values("n_pairs")
            if method_df.empty:
                continue
            ax.plot(
                method_df["n_pairs"],
                method_df["time_per_pair_ms"],
                marker="o",
                linewidth=2.0,
                label=method,
                color=method_colors.get(method),
            )
        ax.set_yscale("log")
        ax.set_title(metapath)
        ax.set_xlabel("Pair count")
        ax.set_ylabel("Time per pair (ms, log scale)")
        ax.grid(alpha=0.3)
        ax.legend()

    for idx in range(len(metapaths), len(axes_flat)):
        axes_flat[idx].axis("off")

    _save(fig, output_dir / "time_per_pair_by_metapath.png")


def _plot_overall(df: pd.DataFrame, output_dir: Path) -> None:
    overall = (
        df.groupby(["method", "n_pairs"], as_index=False)
        .agg(time_per_pair_ms=("time_per_pair_ms", "median"))
        .sort_values(["method", "n_pairs"])
    )
    if overall.empty:
        return

    method_colors = {"Direct": "#1f77b4", "API": "#d62728"}
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for method in ["Direct", "API"]:
        subset = overall[overall["method"] == method]
        if subset.empty:
            continue
        ax.plot(
            subset["n_pairs"],
            subset["time_per_pair_ms"],
            marker="o",
            linewidth=2.2,
            label=f"{method} (median across metapaths)",
            color=method_colors.get(method),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Pair count")
    ax.set_ylabel("Time per pair (ms, log scale)")
    ax.set_title("DWPC Runtime vs Pair Count")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, output_dir / "time_per_pair_overall.png")


def _plot_speedup(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    wide = df.pivot_table(
        index=["metapath", "n_pairs"],
        columns="method",
        values="time_per_pair_ms",
        aggfunc="median",
    )
    if "Direct" not in wide.columns or "API" not in wide.columns:
        return pd.DataFrame()

    wide = wide.dropna(subset=["Direct", "API"]).reset_index()
    if wide.empty:
        return wide
    wide["speedup_api_over_direct"] = wide["API"] / wide["Direct"]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for metapath, subset in wide.groupby("metapath", sort=True):
        subset = subset.sort_values("n_pairs")
        ax.plot(
            subset["n_pairs"],
            subset["speedup_api_over_direct"],
            marker="o",
            linewidth=1.6,
            alpha=0.9,
            label=metapath,
        )

    geom = (
        wide.groupby("n_pairs", as_index=False)
        .agg(
            speedup_geom_mean=(
                "speedup_api_over_direct",
                lambda x: float(np.exp(np.mean(np.log(np.clip(x, 1e-12, None))))),
            )
        )
        .sort_values("n_pairs")
    )
    ax.plot(
        geom["n_pairs"],
        geom["speedup_geom_mean"],
        color="black",
        linewidth=2.6,
        linestyle="--",
        marker="s",
        label="Geometric mean",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Pair count")
    ax.set_ylabel("API / Direct speedup (log scale)")
    ax.set_title("Direct DWPC Speedup Over API")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, output_dir / "speedup_api_over_direct.png")
    return wide


def _write_summaries(
    df: pd.DataFrame,
    speedup_df: pd.DataFrame,
    output_dir: Path,
    warm_min_pairs: int,
) -> None:
    warm_df = df[df["n_pairs"] >= warm_min_pairs].copy()

    method_summary = (
        warm_df.groupby("method", as_index=False)
        .agg(
            n_rows=("time_per_pair_ms", "size"),
            time_per_pair_min_ms=("time_per_pair_ms", "min"),
            time_per_pair_median_ms=("time_per_pair_ms", "median"),
            time_per_pair_mean_ms=("time_per_pair_ms", "mean"),
            time_per_pair_max_ms=("time_per_pair_ms", "max"),
        )
        .sort_values("method")
    )
    method_summary.to_csv(output_dir / "benchmark_summary_by_method.csv", index=False)

    metapath_summary = (
        warm_df.groupby(["metapath", "method"], as_index=False)
        .agg(
            n_rows=("time_per_pair_ms", "size"),
            time_per_pair_median_ms=("time_per_pair_ms", "median"),
            time_per_pair_mean_ms=("time_per_pair_ms", "mean"),
        )
        .sort_values(["metapath", "method"])
    )
    metapath_summary.to_csv(output_dir / "benchmark_summary_by_metapath.csv", index=False)

    if not speedup_df.empty:
        warm_speedup = speedup_df[speedup_df["n_pairs"] >= warm_min_pairs].copy()
        if not warm_speedup.empty:
            speedup_summary = (
                warm_speedup.groupby("metapath", as_index=False)
                .agg(
                    speedup_min=("speedup_api_over_direct", "min"),
                    speedup_median=("speedup_api_over_direct", "median"),
                    speedup_mean=("speedup_api_over_direct", "mean"),
                    speedup_max=("speedup_api_over_direct", "max"),
                )
                .sort_values("speedup_mean", ascending=False)
            )
            speedup_summary.to_csv(
                output_dir / "benchmark_speedup_summary.csv", index=False
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and summarize DWPC benchmark runtime comparison."
    )
    parser.add_argument(
        "--input",
        default="output/benchmark_results.csv",
        help="CSV from scripts/benchmark_dwpc_methods.py",
    )
    parser.add_argument(
        "--output-dir",
        default="output/benchmark_plots",
        help="Directory to write benchmark plots and summary tables.",
    )
    parser.add_argument(
        "--warm-min-pairs",
        type=int,
        default=50,
        help="Treat runs with n_pairs >= this as warmed-cache measurements.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark input not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    df = _coerce(df)
    if df.empty:
        raise ValueError(f"No valid rows in {input_path}")

    _plot_per_metapath(df=df, output_dir=output_dir)
    _plot_overall(df=df, output_dir=output_dir)
    speedup_df = _plot_speedup(df=df, output_dir=output_dir)
    _write_summaries(
        df=df,
        speedup_df=speedup_df,
        output_dir=output_dir,
        warm_min_pairs=max(1, int(args.warm_min_pairs)),
    )

    print("DWPC benchmark plotting complete.")
    print(f"  Input: {input_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Rows: {len(df):,}")
    print(f"  Metapaths: {df['metapath'].nunique()}")
    print(f"  Methods: {', '.join(sorted(df['method'].unique()))}")


if __name__ == "__main__":
    main()
