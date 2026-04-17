#!/usr/bin/env python3
"""Compute intermediate sharing statistics for LV gene sets.

For each LV with supported metapaths:
1. Select metapaths by effect size (Cohen's d > threshold) at specified B value(s)
2. Filter genes by DWPC percentile (default: top 25% per metapath)
3. Enumerate path instances for filtered genes
4. Compute intermediate coverage and sharing metrics

Supports multi-B mode for running analysis across multiple B values.

Usage:
    # Single B value (default B=10)
    python scripts/lv_intermediate_sharing.py --b 10 --output-dir output/lv_int_share

    # Multiple B values
    python scripts/lv_intermediate_sharing.py --b-values 2,5,10,20,30 --output-dir output/lv_int_share

    # With custom DWPC percentile threshold
    python scripts/lv_intermediate_sharing.py --b 10 --dwpc-percentile 75 --output-dir output/lv_int_share

Outputs:
    - intermediate_sharing_by_metapath.csv: Per-metapath sharing statistics
    - intermediate_sharing_summary.csv: Aggregated summary per LV
    - top_intermediates_by_metapath.csv: Top 20 intermediates per metapath
    - selected_metapaths.csv: Metapaths selected by effect size threshold
    - dropped_lvs.csv: LVs dropped due to no metapaths meeting threshold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

import sys  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))

from src.path_enumeration import (  # noqa: E402
    EdgeLoader,
    load_node_maps,
    load_node_names,
    parse_metapath,
)
from src.intermediate_sharing import (  # noqa: E402
    compute_dwpc_thresholds,
    compute_intermediate_coverage,
    enumerate_gene_intermediates,
    load_runs_at_b,
)


def _select_metapaths_by_effect_size(
    results_df: pd.DataFrame,
    d_threshold: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select metapaths by effect size (Cohen's d) from permutation null.

    Per web_tool_discussion.md (section 1.1c, 1.2):
    - Use permutation null only (not random, not consensus)
    - Rank metapaths by effect size d
    - Select metapaths with d > 0.2 (small+ effect per Cohen's benchmarks)

    Args:
        results_df: DataFrame with columns including 'effect_size_d' and 'lv_id'
        d_threshold: Minimum effect size to include (default 0.2 = small effect)

    Returns:
        Tuple of:
        - DataFrame of selected metapaths with metapath_rank column
        - DataFrame of dropped LVs with reason
    """
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected_rows = []
    dropped_rows = []

    for lv_id, group in results_df.groupby("lv_id"):
        if group.empty:
            continue

        target_id = group["target_id"].iloc[0]
        target_name = group["target_name"].iloc[0]
        node_type = group["node_type"].iloc[0]
        n_metapaths_total = len(group)
        max_d = group["effect_size_d"].max()

        # Filter to metapaths with d > threshold
        above_threshold = group[group["effect_size_d"] > d_threshold].copy()

        if above_threshold.empty:
            dropped_rows.append({
                "lv_id": lv_id,
                "target_id": target_id,
                "target_name": target_name,
                "node_type": node_type,
                "n_metapaths_total": n_metapaths_total,
                "n_metapaths_selected": 0,
                "max_effect_size_d": max_d,
                "reason": f"No metapaths with effect size d > {d_threshold}",
            })
            continue

        # Sort by effect size descending and assign ranks
        above_threshold = above_threshold.sort_values(
            "effect_size_d", ascending=False
        ).reset_index(drop=True)
        above_threshold["metapath_rank"] = range(1, len(above_threshold) + 1)
        selected_rows.append(above_threshold)

    if not selected_rows:
        dropped_df = pd.DataFrame(dropped_rows) if dropped_rows else pd.DataFrame()
        return pd.DataFrame(), dropped_df

    return pd.concat(selected_rows, ignore_index=True), pd.DataFrame(dropped_rows)


def _compute_sharing_stats(gene_intermediates: dict[int, set[str]]) -> dict:
    """Compute sharing statistics for a set of genes."""
    n_genes = len(gene_intermediates)
    if n_genes == 0:
        # Return 0 (not None) so metapaths without paths are included in median calculations
        return {
            "n_genes_with_paths": 0,
            "n_genes_sharing": 0,
            "pct_genes_sharing": 0.0,
            "n_unique_intermediates": 0,
            "median_jaccard_to_group": 0.0,
            "mean_jaccard_to_group": 0.0,
        }

    # Count genes that share at least one intermediate with another gene
    # Also compute gene-to-group Jaccard similarity
    n_sharing = 0
    jaccard_scores = []
    for gene_id, ints in gene_intermediates.items():
        other_ints = set()
        for other_gene, other_gene_ints in gene_intermediates.items():
            if other_gene != gene_id:
                other_ints.update(other_gene_ints)
        if ints & other_ints:
            n_sharing += 1
        # Jaccard: |intersection| / |union|
        union = ints | other_ints
        if union:
            jaccard = len(ints & other_ints) / len(union)
            jaccard_scores.append(jaccard)

    all_intermediates = set()
    for ints in gene_intermediates.values():
        all_intermediates.update(ints)

    return {
        "n_genes_with_paths": n_genes,
        "n_genes_sharing": n_sharing,
        "pct_genes_sharing": n_sharing / n_genes * 100 if n_genes > 0 else None,
        "n_unique_intermediates": len(all_intermediates),
        "median_jaccard_to_group": float(np.median(jaccard_scores)) if jaccard_scores else None,
        "mean_jaccard_to_group": float(np.mean(jaccard_scores)) if jaccard_scores else None,
    }


def _load_dwpc_from_numpy(lv_output_dir: Path, lv_id: str) -> Dict[Tuple, float]:
    """Load per-gene DWPC values from numpy arrays.

    Returns dict mapping (lv_id, metapath, gene_id) -> dwpc
    """
    scores_path = lv_output_dir / "gene_feature_scores.npy"
    genes_path = lv_output_dir / "gene_ids.npy"
    manifest_path = lv_output_dir / "feature_manifest.csv"

    if not all(p.exists() for p in [scores_path, genes_path, manifest_path]):
        return {}

    scores = np.load(scores_path)  # (n_genes, n_features)
    gene_ids = np.load(genes_path)  # (n_genes,)
    manifest = pd.read_csv(manifest_path)  # feature_idx -> (lv_id, metapath)

    dwpc_lookup = {}
    for _, row in manifest.iterrows():
        feature_idx = row["feature_idx"]
        metapath = row["metapath"]

        # Get DWPC values for all genes for this feature
        feature_scores = scores[:, feature_idx]

        for gene_idx, gene_id in enumerate(gene_ids):
            dwpc = feature_scores[gene_idx]
            if dwpc > 0:  # Only store non-zero values
                key = (lv_id, metapath, int(gene_id))
                dwpc_lookup[key] = float(dwpc)

    return dwpc_lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lv-output-dirs",
        nargs="+",
        default=["output/lv_experiment"],
        help="Paths to LV experiment output directories (can specify multiple).",
    )
    parser.add_argument(
        "--b",
        type=int,
        default=None,
        help="Single B value for metapath selection. Use --b-values for multiple.",
    )
    parser.add_argument(
        "--b-values",
        default=None,
        help="Comma-separated B values to run (e.g., '2,5,10,20,30'). Creates subdirectories per B.",
    )
    parser.add_argument("--path-top-k", type=int, default=100)
    parser.add_argument("--degree-d", type=float, default=0.5)
    parser.add_argument(
        "--effect-size-threshold", type=float, default=0.5,
        help="Minimum effect size (Cohen's d) for metapath selection. Default 0.5 (medium effect).",
    )
    parser.add_argument(
        "--dwpc-percentile", type=float, default=75.0,
        help="DWPC percentile threshold within (gene_set, metapath). Default 75 (top 25%%).",
    )
    parser.add_argument("--output-dir", default="output/lv_intermediate_sharing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and concatenate data from all LV output directories
    # Note: results_df is loaded per-B value later, but we collect runs_paths here
    runs_paths = []
    top_genes_frames = []
    lv_targets_frames = []
    dwpc_lookup: dict[tuple, float] = {}

    for lv_output_dir_str in args.lv_output_dirs:
        lv_output_dir = Path(lv_output_dir_str)
        print(f"\nLoading from: {lv_output_dir}")

        runs_path = lv_output_dir / "lv_rank_stability_experiment" / "all_runs_long.csv"
        top_genes_path = lv_output_dir / "lv_top_genes.csv"
        lv_targets_path = lv_output_dir / "lv_targets.csv"

        if not runs_path.exists():
            print(f"  Warning: {runs_path} not found, skipping")
            continue

        runs_paths.append(runs_path)
        top_genes_df = pd.read_csv(top_genes_path)
        top_genes_frames.append(top_genes_df)
        lv_targets_frames.append(pd.read_csv(lv_targets_path))

        # Load per-gene DWPC from numpy arrays
        lv_ids = top_genes_df["lv_id"].unique()
        for lv_id in lv_ids:
            lv_dwpc = _load_dwpc_from_numpy(lv_output_dir, lv_id)
            dwpc_lookup.update(lv_dwpc)
            if lv_dwpc:
                print(f"  Loaded {len(lv_dwpc)} DWPC entries for {lv_id}")

    if not runs_paths:
        print("No valid LV output directories found.")
        return

    top_genes = pd.concat(top_genes_frames, ignore_index=True).drop_duplicates()
    lv_targets = pd.concat(lv_targets_frames, ignore_index=True).drop_duplicates()

    if dwpc_lookup:
        print(f"\nTotal DWPC lookup entries: {len(dwpc_lookup)}")
    else:
        print("\nWarning: No DWPC data found, DWPC filtering disabled")

    # Compute DWPC percentile thresholds for filtering
    # For each (gene_set, metapath), include only genes above the specified percentile
    dwpc_percentile = args.dwpc_percentile
    if dwpc_lookup and dwpc_percentile > 0:
        dwpc_thresholds = compute_dwpc_thresholds(dwpc_lookup, dwpc_percentile)
        print(f"DWPC percentile threshold: {dwpc_percentile} (top {100 - dwpc_percentile:.0f}% of genes per metapath)")
        print(f"Computed thresholds for {len(dwpc_thresholds)} (lv_id, metapath) pairs")
    else:
        dwpc_thresholds = {}
        print("DWPC filtering disabled (no DWPC data or percentile=0)")

    # Determine B values to run
    if args.b_values:
        b_values = [int(b.strip()) for b in args.b_values.split(",")]
        print(f"\nMulti-B mode: running for B = {b_values}")
    elif args.b is not None:
        b_values = [args.b]
    else:
        b_values = [10]  # Default
        print("\nUsing default B = 10")

    print(f"\nLoaded {len(top_genes)} top genes")
    print(f"Loaded {len(lv_targets)} LV target entries")

    # Load node name mappings for human-readable intermediate names
    node_name_maps = load_node_names(REPO_ROOT)
    print(f"Loaded node name maps for: {list(node_name_maps.keys())}")

    # Setup for path enumeration (shared across B values)
    edges_dir = REPO_ROOT / "data" / "edges"
    edge_loader = EdgeLoader(edges_dir)

    # Initialize node maps (will be populated on first B iteration)
    maps = None

    # Process each B value
    for b_value in b_values:
        print(f"\n{'='*60}")
        print(f"Processing B = {b_value}")
        print("=" * 60)

        # Always write to a b{value}/ subdirectory so downstream consumers
        # (generate_gene_table.py, plot_metapath_subgraphs.py, etc.) can locate
        # results uniformly regardless of single- vs multi-B runs.
        b_out_dir = out_dir / f"b{b_value}"
        b_out_dir.mkdir(parents=True, exist_ok=True)

        # Load results at this B value
        results_frames = []
        for runs_path in runs_paths:
            try:
                results_frames.append(load_runs_at_b(runs_path, b_value, ["lv_id", "target_id", "target_name", "node_type", "metapath"]))
            except ValueError as e:
                print(f"  Warning: {e}")
                continue

        if not results_frames:
            print(f"No results found for B = {b_value}, skipping")
            continue

        results_df = pd.concat(results_frames, ignore_index=True).drop_duplicates()
        print(f"Loaded {len(results_df)} metapath results at b={b_value}")

        # Select metapaths by effect size (per web_tool_discussion.md section 1.1c)
        # Uses permutation null only, ranks by Cohen's d
        # Default threshold 0.2 = small effect per Cohen's benchmarks
        selected_mp, dropped_lvs = _select_metapaths_by_effect_size(
            results_df, d_threshold=args.effect_size_threshold
        )

        # Report dropped LVs
        if not dropped_lvs.empty:
            print(f"\n=== Dropped LVs (no metapaths with d > {args.effect_size_threshold}) ===")
            for _, row in dropped_lvs.iterrows():
                print(f"  {row['lv_id']} / {row['target_name']}: "
                      f"{row['n_metapaths_total']} metapaths, max d = {row['max_effect_size_d']:.3f}")
            dropped_lvs.to_csv(b_out_dir / "dropped_lvs.csv", index=False)
            print(f"Saved: {b_out_dir / 'dropped_lvs.csv'}")

        if selected_mp.empty:
            print(f"No metapaths found with effect size d > {args.effect_size_threshold}")
            continue

        print(f"\nSelected {len(selected_mp)} metapaths across {selected_mp['lv_id'].nunique()} LVs")

        # Save selected metapaths
        selected_mp.to_csv(b_out_dir / "selected_metapaths.csv", index=False)
        print(f"Saved: {b_out_dir / 'selected_metapaths.csv'}")

        # Summarize selection
        selection_summary = selected_mp.groupby("lv_id").agg(
            n_metapaths_selected=("metapath", "count"),
            median_effect_size_d=("effect_size_d", "median"),
            max_effect_size_d=("effect_size_d", "max"),
            target_name=("target_name", "first"),
            node_type=("node_type", "first"),
        ).reset_index()
        print(f"\n=== Metapath selection summary (b={b_value}, d > {args.effect_size_threshold}) ===")
        print(selection_summary.to_string(index=False))

        # Load node maps on first iteration (shared across B values)
        if maps is None:
            all_node_types = set()
            for mp in selected_mp["metapath"].unique():
                nodes, _ = parse_metapath(mp)
                all_node_types.update(nodes)
            print(f"Loading node maps for: {sorted(all_node_types)}")
            maps = load_node_maps(REPO_ROOT, list(all_node_types))
            print(f"Gene map has {len(maps.id_to_pos.get('G', {}))} entries")

        # Process each LV
        sharing_rows = []
        top_intermediates_rows = []
        gene_path_records: list[dict] = []

        for lv_id, group in selected_mp.groupby("lv_id"):
            target_name = group["target_name"].iloc[0]
            node_type = group["node_type"].iloc[0]

            # Get genes for this LV (convert to int for node map lookup)
            lv_genes_raw = top_genes[top_genes["lv_id"] == lv_id]["gene_identifier"].tolist()
            lv_genes = [int(g) for g in lv_genes_raw]
            print(f"\n{lv_id} / {target_name}: {len(lv_genes)} genes, {len(group)} metapaths")

            # Get target position from lv_targets (single target per LV)
            lv_target_row = lv_targets[lv_targets["lv_id"] == lv_id]
            if lv_target_row.empty:
                print(f"  Warning: No target found for {lv_id}")
                continue
            target_position = lv_target_row["target_position"].iloc[0]
            print(f"  Target position: {target_position}")

            for _, mp_row in group.iterrows():
                metapath = mp_row["metapath"]
                mp_rank = mp_row["metapath_rank"]
                # Effect size from permutation null (already computed)
                effect_size = mp_row.get("effect_size_d", 0)

                is_first_mp = (mp_rank == 1)

                gene_ints = enumerate_gene_intermediates(
                    lv_genes, int(target_position), metapath, edge_loader, maps,
                    path_top_k=args.path_top_k, degree_d=args.degree_d,
                    debug=is_first_mp,
                    gene_set_id=lv_id,
                    dwpc_lookup=dwpc_lookup or None,
                    dwpc_thresholds=dwpc_thresholds or None,
                    record_paths=gene_path_records,
                    record_extra={"lv_id": lv_id},
                )
                if gene_ints:
                    print(f"    Found paths for {len(gene_ints)} genes")

                stats = _compute_sharing_stats(gene_ints)
                coverage_stats, intermediate_stats = compute_intermediate_coverage(
                    gene_ints, node_name_maps=node_name_maps
                )
                stats.update(coverage_stats)
                stats.update({
                    "lv_id": lv_id,
                    "target_id": lv_target_row["target_id"].iloc[0],
                    "target_name": target_name,
                    "node_type": node_type,
                    "metapath": metapath,
                    "metapath_rank": mp_rank,
                    "effect_size_d": effect_size,
                    "n_genes_total": len(lv_genes),
                })
                sharing_rows.append(stats)

                # Store top intermediates (top 20 per metapath)
                for rank, int_stat in enumerate(intermediate_stats[:20], 1):
                    top_intermediates_rows.append({
                        "lv_id": lv_id,
                        "target_id": lv_target_row["target_id"].iloc[0],
                        "target_name": target_name,
                        "metapath": metapath,
                        "metapath_rank": mp_rank,
                        "intermediate_rank": rank,
                        "intermediate_id": int_stat["intermediate_id"],
                        "intermediate_name": int_stat.get("intermediate_name"),
                        "n_genes_using": int_stat["n_genes_using"],
                        "pct_genes_using": int_stat["pct_genes_using"],
                    })

                # Classify effect size per Cohen's benchmarks
                d_category = "large" if effect_size >= 0.8 else "medium" if effect_size >= 0.5 else "small"

                if stats['n_genes_with_paths'] > 0:
                    pct_maj = stats.get('pct_intermediates_shared_majority', 0) or 0
                    top1 = stats.get('top1_intermediate_coverage', 0) or 0
                    print(f"  {metapath} (d={effect_size:.2f}, {d_category}): "
                          f"{stats['n_genes_with_paths']}/{len(lv_genes)} genes, "
                          f"top1={top1:.0f}%, maj_shared={pct_maj:.0f}%")
                else:
                    print(f"  {metapath} (d={effect_size:.2f}, {d_category}): no paths")

        # Save detailed results for this B value
        sharing_df = pd.DataFrame(sharing_rows)
        sharing_df["b"] = b_value  # Add B value column
        col_order = [
            "lv_id", "target_id", "target_name", "node_type", "b",
            "metapath", "metapath_rank", "effect_size_d",
            "n_genes_total",
            "n_genes_with_paths", "n_genes_sharing", "pct_genes_sharing",
            "median_jaccard_to_group", "mean_jaccard_to_group",
            "n_unique_intermediates",
            "n_shared_intermediates_2plus", "n_shared_intermediates_quarter",
            "n_shared_intermediates_majority", "n_shared_intermediates_all",
            "pct_intermediates_shared_2plus", "pct_intermediates_shared_quarter",
            "pct_intermediates_shared_majority", "pct_intermediates_shared_all",
            "n_intermediates_cover_50pct", "n_intermediates_cover_80pct",
            "top1_intermediate_coverage", "top5_intermediate_coverage",
        ]
        sharing_df = sharing_df[[c for c in col_order if c in sharing_df.columns]]
        sharing_df.to_csv(b_out_dir / "intermediate_sharing_by_metapath.csv", index=False)
        print(f"\nSaved: {b_out_dir / 'intermediate_sharing_by_metapath.csv'}")

        # Save top intermediates
        if top_intermediates_rows:
            top_int_df = pd.DataFrame(top_intermediates_rows)
            top_int_df["b"] = b_value
            top_int_df.to_csv(b_out_dir / "top_intermediates_by_metapath.csv", index=False)
            print(f"Saved: {b_out_dir / 'top_intermediates_by_metapath.csv'}")

        # Save per-gene per-path node sequences for downstream visualization.
        if gene_path_records:
            gene_paths_df = pd.DataFrame(gene_path_records)
            gene_paths_df["b"] = b_value
            gene_paths_df.to_csv(b_out_dir / "gene_paths.csv", index=False)
            print(f"Saved: {b_out_dir / 'gene_paths.csv'}")

        # Aggregate summary per LV
        summary = sharing_df.groupby(["lv_id", "target_id", "target_name", "node_type"]).agg(
            n_metapaths=("metapath", "count"),
            n_genes_total=("n_genes_total", "first"),
            # Effect size stats
            median_effect_size_d=("effect_size_d", "median"),
            max_effect_size_d=("effect_size_d", "max"),
            # Sharing stats
            median_pct_sharing=("pct_genes_sharing", "median"),
            mean_pct_sharing=("pct_genes_sharing", "mean"),
            max_pct_sharing=("pct_genes_sharing", "max"),
            median_jaccard=("median_jaccard_to_group", "median"),
            mean_jaccard=("mean_jaccard_to_group", "mean"),
            median_n_intermediates=("n_unique_intermediates", "median"),
            median_pct_intermediates_shared_quarter=("pct_intermediates_shared_quarter", "median"),
            median_pct_intermediates_shared_majority=("pct_intermediates_shared_majority", "median"),
            median_pct_intermediates_shared_all=("pct_intermediates_shared_all", "median"),
            median_top1_coverage=("top1_intermediate_coverage", "median"),
            median_top5_coverage=("top5_intermediate_coverage", "median"),
            median_n_for_50pct=("n_intermediates_cover_50pct", "median"),
            median_n_for_80pct=("n_intermediates_cover_80pct", "median"),
        ).reset_index()
        summary["b"] = b_value
        summary.to_csv(b_out_dir / "intermediate_sharing_summary.csv", index=False)
        print(f"Saved: {b_out_dir / 'intermediate_sharing_summary.csv'}")

        # Print summary for abstract
        print("\n" + "=" * 60)
        print(f"=== Summary for B = {b_value} ===")
        print("=" * 60)
        print("Effect size categories (Cohen's d): large >= 0.8, medium >= 0.5, small >= 0.2")
        for _, row in summary.iterrows():
            median_d = row.get('median_effect_size_d', 0)
            max_d = row.get('max_effect_size_d', 0)
            d_cat = "large" if median_d >= 0.8 else "medium" if median_d >= 0.5 else "small"
            print(f"\n{row['lv_id']} / {row['target_name']}:")
            print(f"  {row['n_metapaths']} metapaths with effect size d > {args.effect_size_threshold}")
            print(f"  Median effect size d = {median_d:.2f} ({d_cat}), max d = {max_d:.2f}")
            print(f"  {row['n_genes_total']} genes in set")
            print(f"  Median {row['median_n_intermediates']:.0f} unique intermediates per metapath")
            if pd.notna(row.get('median_top1_coverage')):
                print(f"  Median top-1 intermediate covers {row['median_top1_coverage']:.1f}% of genes")
            if pd.notna(row.get('median_top5_coverage')):
                print(f"  Median top-5 intermediates cover {row['median_top5_coverage']:.1f}% of genes")
            if pd.notna(row.get('median_pct_intermediates_shared_quarter')):
                print(f"  Median {row['median_pct_intermediates_shared_quarter']:.1f}% of intermediates used by >25% of genes")
            if pd.notna(row.get('median_pct_intermediates_shared_majority')):
                print(f"  Median {row['median_pct_intermediates_shared_majority']:.1f}% of intermediates used by >50% of genes")
            if pd.notna(row.get('median_pct_intermediates_shared_all')):
                print(f"  Median {row['median_pct_intermediates_shared_all']:.1f}% of intermediates used by ALL genes")
            if pd.notna(row.get('median_n_for_80pct')):
                print(f"  Median {row['median_n_for_80pct']:.0f} intermediates needed to cover 80% of genes")
            if pd.notna(row['median_jaccard']):
                print(f"  Median Jaccard similarity: {row['median_jaccard']:.3f}")

    print("\n" + "=" * 60)
    print("All B values processed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
