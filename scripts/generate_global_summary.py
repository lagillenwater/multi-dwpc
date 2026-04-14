#!/usr/bin/env python3
"""Generate global summary metrics for LV or Year intermediate sharing analysis.

Aggregates per-gene-set metrics from intermediate sharing results:
- Metapath counts and effect sizes
- Gene coverage statistics
- Intermediate sharing and convergence metrics

Usage:
    python scripts/generate_global_summary.py --analysis-type lv \
        --input-dir output/lv_intermediate_sharing --output-dir output/lv_global_summary

    python scripts/generate_global_summary.py --analysis-type year \
        --input-dir output/year_intermediate_sharing --output-dir output/year_global_summary

    # With chosen B from elbow selection
    python scripts/generate_global_summary.py --analysis-type lv \
        --input-dir output/lv_intermediate_sharing \
        --chosen-b-json output/lv_full_analysis/chosen_b.json \
        --output-dir output/lv_global_summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_intermediate_sharing_data(
    input_dir: Path,
    b_value: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load intermediate sharing results.

    Args:
        input_dir: Directory containing intermediate sharing results
        b_value: If specified, load from b{value}/ subdirectory

    Returns:
        Tuple of (by_metapath_df, top_intermediates_df)
    """
    if b_value is not None:
        data_dir = input_dir / f"b{b_value}"
    else:
        data_dir = input_dir

    by_metapath_path = data_dir / "intermediate_sharing_by_metapath.csv"
    top_int_path = data_dir / "top_intermediates_by_metapath.csv"

    if not by_metapath_path.exists():
        raise FileNotFoundError(f"Not found: {by_metapath_path}")

    by_metapath_df = pd.read_csv(by_metapath_path)

    if top_int_path.exists():
        top_int_df = pd.read_csv(top_int_path)
    else:
        top_int_df = pd.DataFrame()

    return by_metapath_df, top_int_df


def compute_global_summary(
    by_metapath_df: pd.DataFrame,
    top_int_df: pd.DataFrame,
    analysis_type: str,
    b_value: int | None = None,
) -> pd.DataFrame:
    """Compute global summary metrics per gene set.

    Args:
        by_metapath_df: Per-metapath sharing statistics
        top_int_df: Top intermediates per metapath
        analysis_type: "lv" or "year"
        b_value: B value used (for inclusion in output)

    Returns:
        DataFrame with one row per gene set
    """
    if analysis_type == "lv":
        group_cols = ["lv_id", "target_id", "target_name", "node_type"]
        id_col = "lv_id"
    else:
        group_cols = ["go_id", "target_id", "target_name", "node_type"]
        id_col = "go_id"

    # Filter to existing columns
    group_cols = [c for c in group_cols if c in by_metapath_df.columns]

    summary_rows = []

    for group_key, group in by_metapath_df.groupby(group_cols, sort=True):
        if isinstance(group_key, tuple):
            row = dict(zip(group_cols, group_key))
        else:
            row = {group_cols[0]: group_key}

        # Add B value if available
        if b_value is not None:
            row["b"] = b_value
        elif "b" in group.columns:
            row["b"] = group["b"].iloc[0]

        # Metapath statistics
        row["n_metapaths_selected"] = len(group)

        # Top metapath by effect size
        top_mp_row = group.loc[group["effect_size_d"].idxmax()]
        row["top_metapath"] = top_mp_row["metapath"]
        row["top_metapath_d"] = top_mp_row["effect_size_d"]

        # Effect size distribution
        row["median_effect_size_d"] = group["effect_size_d"].median()
        row["max_effect_size_d"] = group["effect_size_d"].max()
        row["min_effect_size_d"] = group["effect_size_d"].min()

        # Gene coverage
        row["n_genes_total"] = group["n_genes_total"].iloc[0]

        # Compute unique genes with paths across all metapaths
        # Use the max n_genes_with_paths as approximation (true value requires path data)
        row["max_genes_with_paths_per_mp"] = group["n_genes_with_paths"].max()
        row["median_genes_with_paths_per_mp"] = group["n_genes_with_paths"].median()

        # Approximate coverage: max genes with paths / total genes
        if row["n_genes_total"] > 0:
            row["approx_pct_genes_covered"] = (
                row["max_genes_with_paths_per_mp"] / row["n_genes_total"] * 100
            )
        else:
            row["approx_pct_genes_covered"] = 0.0

        # Intermediate sharing metrics (medians across metapaths)
        if "top1_intermediate_coverage" in group.columns:
            row["median_top1_coverage"] = group["top1_intermediate_coverage"].median()
            row["max_top1_coverage"] = group["top1_intermediate_coverage"].max()

        if "top5_intermediate_coverage" in group.columns:
            row["median_top5_coverage"] = group["top5_intermediate_coverage"].median()

        if "pct_intermediates_shared_quarter" in group.columns:
            row["median_pct_shared_quarter"] = group["pct_intermediates_shared_quarter"].median()

        if "pct_intermediates_shared_majority" in group.columns:
            row["median_pct_shared_majority"] = group["pct_intermediates_shared_majority"].median()

        if "pct_intermediates_shared_all" in group.columns:
            row["median_pct_shared_all"] = group["pct_intermediates_shared_all"].median()

        if "n_unique_intermediates" in group.columns:
            row["median_n_intermediates"] = group["n_unique_intermediates"].median()
            row["total_unique_intermediates"] = group["n_unique_intermediates"].sum()

        if "n_intermediates_cover_80pct" in group.columns:
            row["median_n_for_80pct_coverage"] = group["n_intermediates_cover_80pct"].median()

        # Jaccard similarity
        if "median_jaccard_to_group" in group.columns:
            row["median_jaccard"] = group["median_jaccard_to_group"].median()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Sort by number of metapaths selected (descending)
    if "n_metapaths_selected" in summary_df.columns:
        summary_df = summary_df.sort_values("n_metapaths_selected", ascending=False)

    return summary_df.reset_index(drop=True)


def compute_cross_b_summary(
    input_dir: Path,
    b_values: list[int],
    analysis_type: str,
) -> pd.DataFrame:
    """Compute summary across multiple B values.

    Args:
        input_dir: Directory containing b{value}/ subdirectories
        b_values: List of B values to aggregate
        analysis_type: "lv" or "year"

    Returns:
        DataFrame with metrics across B values
    """
    all_summaries = []

    for b in b_values:
        try:
            by_metapath_df, top_int_df = load_intermediate_sharing_data(input_dir, b)
            summary = compute_global_summary(by_metapath_df, top_int_df, analysis_type, b)
            all_summaries.append(summary)
        except FileNotFoundError:
            print(f"Warning: No data for B={b}, skipping")
            continue

    if not all_summaries:
        return pd.DataFrame()

    return pd.concat(all_summaries, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-type",
        choices=["lv", "year"],
        required=True,
        help="Type of analysis (lv or year)",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing intermediate sharing results",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for global summary",
    )
    parser.add_argument(
        "--chosen-b-json",
        help="Path to chosen_b.json from B selection (uses this B for summary)",
    )
    parser.add_argument(
        "--b-values",
        help="Comma-separated B values to summarize (e.g., '2,5,10,20,30')",
    )
    parser.add_argument(
        "--b",
        type=int,
        help="Single B value to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine B value(s) to use
    chosen_b = None
    if args.chosen_b_json:
        with open(args.chosen_b_json) as f:
            chosen_b_data = json.load(f)
        chosen_b = chosen_b_data["chosen_b"]
        print(f"Using chosen B = {chosen_b} from {args.chosen_b_json}")

    if args.b_values:
        b_values = [int(b.strip()) for b in args.b_values.split(",")]
        print(f"Summarizing across B values: {b_values}")

        # Generate cross-B summary
        cross_b_summary = compute_cross_b_summary(input_dir, b_values, args.analysis_type)
        if not cross_b_summary.empty:
            cross_b_path = output_dir / "global_summary_all_b.csv"
            cross_b_summary.to_csv(cross_b_path, index=False)
            print(f"Saved cross-B summary: {cross_b_path}")

        # If chosen_b specified, also generate focused summary at that B
        if chosen_b is not None and chosen_b in b_values:
            by_metapath_df, top_int_df = load_intermediate_sharing_data(input_dir, chosen_b)
            summary = compute_global_summary(
                by_metapath_df, top_int_df, args.analysis_type, chosen_b
            )
            summary_path = output_dir / "global_summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"Saved summary at chosen B={chosen_b}: {summary_path}")

    elif args.b is not None or chosen_b is not None:
        b_value = args.b if args.b is not None else chosen_b
        print(f"Loading data for B = {b_value}")

        by_metapath_df, top_int_df = load_intermediate_sharing_data(input_dir, b_value)
        print(f"Loaded {len(by_metapath_df)} metapath records")

        summary = compute_global_summary(
            by_metapath_df, top_int_df, args.analysis_type, b_value
        )
        summary_path = output_dir / "global_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")

    else:
        # No B specified, try to load from root input_dir
        print("Loading data from root input directory (no B subdirectory)")
        by_metapath_df, top_int_df = load_intermediate_sharing_data(input_dir, None)
        print(f"Loaded {len(by_metapath_df)} metapath records")

        summary = compute_global_summary(
            by_metapath_df, top_int_df, args.analysis_type, None
        )
        summary_path = output_dir / "global_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Global Summary Statistics")
    print("=" * 60)

    if "summary" in dir() and not summary.empty:
        print(f"Gene sets analyzed: {len(summary)}")
        print(f"Total metapaths selected: {summary['n_metapaths_selected'].sum()}")
        if "median_top1_coverage" in summary.columns:
            print(f"Median top-1 coverage: {summary['median_top1_coverage'].median():.1f}%")
        if "median_pct_shared_majority" in summary.columns:
            print(f"Median % shared by majority: {summary['median_pct_shared_majority'].median():.1f}%")


if __name__ == "__main__":
    main()
