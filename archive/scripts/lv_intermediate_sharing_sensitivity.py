#!/usr/bin/env python3
"""Sensitivity analysis for path_top_k parameter in intermediate sharing.

Runs lv_intermediate_sharing.py with multiple path_top_k values and compares:
- % genes sharing intermediates
- % intermediates shared
- Top intermediate coverage
- Stability of top intermediates across k values

Usage:
    python scripts/lv_intermediate_sharing_sensitivity.py \
        --lv-output-dirs output/lv_single_target_refactor \
        --path-top-k-values 10,50,100,500,1000 \
        --output-dir output/lv_intermediate_sharing_sensitivity
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lv-output-dirs",
        nargs="+",
        default=["output/lv_single_target_refactor"],
        help="Paths to LV experiment output directories.",
    )
    parser.add_argument(
        "--path-top-k-values",
        type=str,
        default="10,50,100,500,1000",
        help="Comma-separated list of path_top_k values to test.",
    )
    parser.add_argument(
        "--b",
        type=int,
        default=10,
        help="B value for metapath selection.",
    )
    parser.add_argument(
        "--dwpc-threshold",
        type=str,
        default="p75",
        help="DWPC threshold.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/lv_intermediate_sharing_sensitivity",
        help="Output directory for sensitivity analysis.",
    )
    parser.add_argument(
        "--skip-compute",
        action="store_true",
        help="Skip computation, only aggregate existing results.",
    )
    return parser.parse_args()


def run_intermediate_sharing(
    lv_output_dirs: list[str],
    path_top_k: int,
    b: int,
    dwpc_threshold: str,
    output_dir: Path,
) -> None:
    """Run lv_intermediate_sharing.py for a specific path_top_k value."""
    k_output_dir = output_dir / f"k{path_top_k}"

    cmd = [
        sys.executable,
        "scripts/lv_intermediate_sharing.py",
        "--lv-output-dirs", *lv_output_dirs,
        "--path-top-k", str(path_top_k),
        "--b", str(b),
        "--dwpc-threshold", dwpc_threshold,
        "--output-dir", str(k_output_dir),
    ]

    print(f"\n{'='*60}")
    print(f"Running with path_top_k = {path_top_k}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Warning: Command failed with return code {result.returncode}")


def load_results(output_dir: Path, k_values: list[int]) -> pd.DataFrame:
    """Load results from all k values into a single DataFrame."""
    all_rows = []

    for k in k_values:
        k_dir = output_dir / f"k{k}"
        metapath_file = k_dir / "intermediate_sharing_by_metapath.csv"

        if not metapath_file.exists():
            print(f"Warning: No results for k={k} at {metapath_file}")
            continue

        df = pd.read_csv(metapath_file)
        df["path_top_k"] = k
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def load_top_intermediates(output_dir: Path, k_values: list[int]) -> pd.DataFrame:
    """Load top intermediates from all k values."""
    all_rows = []

    for k in k_values:
        k_dir = output_dir / f"k{k}"
        top_int_file = k_dir / "top_intermediates_by_metapath.csv"

        if not top_int_file.exists():
            continue

        df = pd.read_csv(top_int_file)
        df["path_top_k"] = k
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def compute_stability_metrics(
    combined_df: pd.DataFrame,
    k_values: list[int],
) -> pd.DataFrame:
    """Compute stability metrics across k values."""
    stability_rows = []

    # Group by LV and metapath
    for (lv_id, metapath), group in combined_df.groupby(["lv_id", "metapath"]):
        if len(group) < 2:
            continue

        row = {
            "lv_id": lv_id,
            "metapath": metapath,
        }

        # Check if metrics are stable across k values
        for metric in ["pct_genes_sharing", "pct_intermediates_shared",
                       "top1_intermediate_coverage", "top5_intermediate_coverage"]:
            if metric not in group.columns:
                continue
            values = group.set_index("path_top_k")[metric].reindex(k_values)
            valid_values = values.dropna()
            if len(valid_values) >= 2:
                row[f"{metric}_std"] = valid_values.std()
                row[f"{metric}_range"] = valid_values.max() - valid_values.min()
                row[f"{metric}_cv"] = valid_values.std() / valid_values.mean() if valid_values.mean() > 0 else None

        stability_rows.append(row)

    return pd.DataFrame(stability_rows)


def compute_top_intermediate_overlap(
    top_int_df: pd.DataFrame,
    k_values: list[int],
    top_n: int = 5,
) -> pd.DataFrame:
    """Compute Jaccard overlap of top intermediates across k values."""
    overlap_rows = []

    for (lv_id, metapath), group in top_int_df.groupby(["lv_id", "metapath"]):
        # Get top-n intermediates for each k
        top_sets = {}
        for k in k_values:
            k_data = group[group["path_top_k"] == k]
            top_ints = set(k_data.nsmallest(top_n, "intermediate_rank")["intermediate_id"])
            if top_ints:
                top_sets[k] = top_ints

        if len(top_sets) < 2:
            continue

        # Compute pairwise Jaccard
        k_list = sorted(top_sets.keys())
        for i, k1 in enumerate(k_list):
            for k2 in k_list[i+1:]:
                set1, set2 = top_sets[k1], top_sets[k2]
                jaccard = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0
                overlap_rows.append({
                    "lv_id": lv_id,
                    "metapath": metapath,
                    "k1": k1,
                    "k2": k2,
                    "jaccard_top_n": jaccard,
                    "n_shared": len(set1 & set2),
                    "n_union": len(set1 | set2),
                })

    return pd.DataFrame(overlap_rows)


def save_figure(fig: plt.Figure, fig_dir: Path, name: str, dpi: int = 150) -> None:
    """Save figure as both PNG and PDF."""
    fig.savefig(fig_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_curves(
    combined_df: pd.DataFrame,
    k_values: list[int],
    output_dir: Path,
) -> None:
    """Plot how metrics change with path_top_k."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    lv_ids = combined_df["lv_id"].unique()
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    # Metrics to plot
    metrics = [
        ("pct_genes_sharing", "% Genes Sharing Intermediates"),
        ("pct_intermediates_shared", "% Intermediates Shared"),
        ("top1_intermediate_coverage", "Top-1 Intermediate Coverage (%)"),
        ("n_unique_intermediates", "Number of Unique Intermediates"),
    ]

    for metric_col, metric_label in metrics:
        if metric_col not in combined_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, lv_id in enumerate(lv_ids):
            lv_data = combined_df[combined_df["lv_id"] == lv_id]

            # Aggregate across metapaths (median)
            agg = lv_data.groupby("path_top_k")[metric_col].median()

            ax.plot(
                agg.index,
                agg.values,
                marker="o",
                label=lv_id,
                color=colors[idx % len(colors)],
                linewidth=2,
                markersize=8,
            )

            # Add error bands (IQR)
            q25 = lv_data.groupby("path_top_k")[metric_col].quantile(0.25)
            q75 = lv_data.groupby("path_top_k")[metric_col].quantile(0.75)
            ax.fill_between(
                agg.index,
                q25.values,
                q75.values,
                alpha=0.2,
                color=colors[idx % len(colors)],
            )

        ax.set_xlabel("path_top_k")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Sensitivity of {metric_label} to path_top_k")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_figure(fig, fig_dir, f"sensitivity_{metric_col}")

    print(f"Figures saved to {fig_dir}")


def plot_top_intermediate_stability(
    overlap_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot Jaccard overlap of top intermediates across k values."""
    if overlap_df.empty:
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    lv_ids = overlap_df["lv_id"].unique()
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, lv_id in enumerate(lv_ids):
        lv_data = overlap_df[overlap_df["lv_id"] == lv_id]

        # Create x-axis label for k pairs
        lv_data = lv_data.copy()
        lv_data["k_pair"] = lv_data.apply(lambda r: f"{r['k1']}-{r['k2']}", axis=1)

        # Aggregate across metapaths
        agg = lv_data.groupby("k_pair")["jaccard_top_n"].mean()

        ax.bar(
            np.arange(len(agg)) + idx * 0.2,
            agg.values,
            width=0.2,
            label=lv_id,
            color=colors[idx % len(colors)],
            alpha=0.8,
        )

    ax.set_xlabel("k value pairs")
    ax.set_ylabel("Mean Jaccard Overlap of Top-5 Intermediates")
    ax.set_title("Stability of Top Intermediates Across path_top_k Values")
    ax.set_xticks(np.arange(len(agg)) + 0.2)
    ax.set_xticklabels(agg.index, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    save_figure(fig, fig_dir, "top_intermediate_stability")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = [int(k.strip()) for k in args.path_top_k_values.split(",")]
    print(f"Testing path_top_k values: {k_values}")

    # Run intermediate sharing for each k value
    if not args.skip_compute:
        for k in k_values:
            run_intermediate_sharing(
                args.lv_output_dirs,
                path_top_k=k,
                b=args.b,
                dwpc_threshold=args.dwpc_threshold,
                output_dir=output_dir,
            )

    # Load and combine results
    print("\n" + "="*60)
    print("Aggregating results...")
    print("="*60)

    combined_df = load_results(output_dir, k_values)
    if combined_df.empty:
        print("No results found to aggregate.")
        return

    combined_df.to_csv(output_dir / "combined_results.csv", index=False)
    print(f"Saved combined results: {output_dir / 'combined_results.csv'}")

    # Load top intermediates
    top_int_df = load_top_intermediates(output_dir, k_values)
    if not top_int_df.empty:
        top_int_df.to_csv(output_dir / "combined_top_intermediates.csv", index=False)

    # Compute stability metrics
    stability_df = compute_stability_metrics(combined_df, k_values)
    if not stability_df.empty:
        stability_df.to_csv(output_dir / "stability_metrics.csv", index=False)
        print(f"Saved stability metrics: {output_dir / 'stability_metrics.csv'}")

    # Compute top intermediate overlap
    overlap_df = compute_top_intermediate_overlap(top_int_df, k_values)
    if not overlap_df.empty:
        overlap_df.to_csv(output_dir / "top_intermediate_overlap.csv", index=False)
        print(f"Saved top intermediate overlap: {output_dir / 'top_intermediate_overlap.csv'}")

    # Generate plots
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_sensitivity_curves(combined_df, k_values, output_dir)
    plot_top_intermediate_stability(overlap_df, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("Sensitivity Analysis Summary")
    print("="*60)

    for lv_id in combined_df["lv_id"].unique():
        lv_data = combined_df[combined_df["lv_id"] == lv_id]
        target_name = lv_data["target_name"].iloc[0]
        print(f"\n{lv_id} / {target_name}:")

        for k in k_values:
            k_data = lv_data[lv_data["path_top_k"] == k]
            if k_data.empty:
                continue

            median_sharing = k_data["pct_genes_sharing"].median()
            median_int_shared = k_data.get("pct_intermediates_shared", pd.Series([None])).median()
            median_top1 = k_data.get("top1_intermediate_coverage", pd.Series([None])).median()

            print(f"  k={k:4d}: sharing={median_sharing:5.1f}%, "
                  f"int_shared={median_int_shared:5.1f}%, "
                  f"top1_cov={median_top1:5.1f}%")


if __name__ == "__main__":
    main()
