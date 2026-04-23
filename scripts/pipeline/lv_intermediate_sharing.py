#!/usr/bin/env python3
"""Compute intermediate sharing statistics for LV gene sets.

For each LV with supported metapaths:
1. Select metapaths by permutation z-statistic (z > threshold) at specified B value(s)
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

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.path_enumeration import (  # noqa: E402
    EdgeLoader,
    load_node_maps,
    load_node_names,
    parse_metapath,
)
from src.intermediate_sharing import (  # noqa: E402
    compute_dwpc_thresholds,
    compute_dwpc_z_stats,
    compute_intermediate_coverage,
    enumerate_gene_intermediates,
    load_dwpc_from_numpy,
    load_runs_at_b,
)


def _select_metapaths_by_effect_size(
    results_df: pd.DataFrame,
    z_threshold: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select metapaths by permutation z-statistic from permutation null.

    Per web_tool_discussion.md (section 1.1c, 1.2):
    - Use permutation null only (not random, not consensus)
    - Rank metapaths by permutation z
    - Select metapaths with z > threshold

    Args:
        results_df: DataFrame with columns including 'effect_size_z' and 'lv_id'
        z_threshold: Minimum permutation z to include

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
        max_d = group["effect_size_z"].max()

        # Filter to metapaths with d > threshold
        above_threshold = group[group["effect_size_z"] > z_threshold].copy()

        if above_threshold.empty:
            dropped_rows.append({
                "lv_id": lv_id,
                "target_id": target_id,
                "target_name": target_name,
                "node_type": node_type,
                "n_metapaths_total": n_metapaths_total,
                "n_metapaths_selected": 0,
                "max_effect_size_z": max_d,
                "reason": f"No metapaths with permutation z > {z_threshold}",
            })
            continue

        # Sort by effect size descending and assign ranks
        above_threshold = above_threshold.sort_values(
            "effect_size_z", ascending=False
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
    parser.add_argument("--path-top-k", type=int, default=100,
        help="Cap on paths per gene (legacy). Ignored when --path-z-threshold is set.")
    parser.add_argument("--degree-d", type=float, default=0.5)
    parser.add_argument(
        "--effect-size-threshold", type=float, default=0.5,
        help="Minimum permutation z-statistic (diff / null_std) for metapath selection. Default 0.5.",
    )
    parser.add_argument(
        "--dwpc-percentile", type=float, default=75.0,
        help="DWPC percentile threshold within (gene_set, metapath). Ignored when --dwpc-z-threshold is set. Default 75 (top 25%%).",
    )
    parser.add_argument(
        "--dwpc-z-threshold", type=float, default=None,
        help="Keep genes whose (DWPC - mean)/std >= threshold, computed over the gene universe per (gene_set, metapath). Overrides --dwpc-percentile when set.",
    )
    parser.add_argument(
        "--path-z-threshold", type=float, default=None,
        help="Keep paths whose (score - pool_mean)/pool_std >= threshold, computed over all paths for each (gene_set, metapath). Overrides --path-top-k when set.",
    )
    parser.add_argument(
        "--path-enumeration-cap", type=int, default=None,
        help="Max paths to enumerate per gene (used with --path-z-threshold). Default: no cap.",
    )
    parser.add_argument("--output-dir", default="output/lv_intermediate_sharing")
    return parser.parse_args()


SHARING_COL_ORDER = [
    "lv_id", "target_id", "target_name", "node_type", "b",
    "metapath", "metapath_rank", "effect_size_z",
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


def _process_lv(
    lv_id,
    group: pd.DataFrame,
    top_genes: pd.DataFrame,
    lv_targets: pd.DataFrame,
    edge_loader,
    maps,
    node_name_maps: dict,
    dwpc_lookup: dict,
    dwpc_thresholds: dict,
    dwpc_z_stats: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict]]:
    target_name = group["target_name"].iloc[0]
    node_type = group["node_type"].iloc[0]

    lv_genes = top_genes.loc[
        top_genes["lv_id"] == lv_id, "gene_identifier"
    ].astype(int).tolist()
    print(f"\n{lv_id} / {target_name}: {len(lv_genes)} genes, {len(group)} metapaths")

    lv_target_row = lv_targets[lv_targets["lv_id"] == lv_id]
    if lv_target_row.empty:
        print(f"  Warning: No target found for {lv_id}")
        return [], [], []
    target_position = lv_target_row["target_position"].iloc[0]
    target_id = lv_target_row["target_id"].iloc[0]

    sharing_rows: list[dict] = []
    top_intermediates_rows: list[dict] = []
    gene_path_records: list[dict] = []

    for _, mp_row in group.iterrows():
        metapath = mp_row["metapath"]
        mp_rank = mp_row["metapath_rank"]
        z_score = mp_row.get("effect_size_z", 0)

        gene_ints, n_filtered = enumerate_gene_intermediates(
            lv_genes, int(target_position), metapath, edge_loader, maps,
            path_top_k=args.path_top_k, degree_d=args.degree_d,
            debug=(mp_rank == 1),
            gene_set_id=lv_id,
            dwpc_lookup=dwpc_lookup or None,
            dwpc_thresholds=dwpc_thresholds or None,
            dwpc_z_stats=dwpc_z_stats or None,
            dwpc_z_min=args.dwpc_z_threshold if args.dwpc_z_threshold is not None else 1.65,
            path_z_min=args.path_z_threshold,
            path_enumeration_cap=args.path_enumeration_cap,
            record_paths=gene_path_records,
            record_extra={"lv_id": lv_id},
        )

        stats = _compute_sharing_stats(gene_ints)
        stats["n_genes_filtered_by_dwpc"] = n_filtered
        coverage_stats, intermediate_stats = compute_intermediate_coverage(
            gene_ints, node_name_maps=node_name_maps
        )
        stats.update(coverage_stats)
        stats.update({
            "lv_id": lv_id,
            "target_id": target_id,
            "target_name": target_name,
            "node_type": node_type,
            "metapath": metapath,
            "metapath_rank": mp_rank,
            "effect_size_z": z_score,
            "n_genes_total": len(lv_genes),
        })
        sharing_rows.append(stats)

        for rank, int_stat in enumerate(intermediate_stats[:20], 1):
            top_intermediates_rows.append({
                "lv_id": lv_id,
                "target_id": target_id,
                "target_name": target_name,
                "metapath": metapath,
                "metapath_rank": mp_rank,
                "intermediate_rank": rank,
                "intermediate_id": int_stat["intermediate_id"],
                "intermediate_name": int_stat.get("intermediate_name"),
                "n_genes_using": int_stat["n_genes_using"],
                "pct_genes_using": int_stat["pct_genes_using"],
            })

        n_with = stats["n_genes_with_paths"]
        if n_with > 0:
            pct_maj = stats.get("pct_intermediates_shared_majority", 0) or 0
            top1 = stats.get("top1_intermediate_coverage", 0) or 0
            print(f"  {metapath} (z={z_score:.2f}): "
                  f"{n_with}/{len(lv_genes)} genes, top1={top1:.0f}%, maj_shared={pct_maj:.0f}%")
        else:
            print(f"  {metapath} (z={z_score:.2f}): no paths")

    return sharing_rows, top_intermediates_rows, gene_path_records


def _process_b_value(
    b_value: int,
    runs_paths: list[Path],
    out_dir: Path,
    top_genes: pd.DataFrame,
    lv_targets: pd.DataFrame,
    edge_loader,
    maps,
    node_name_maps: dict,
    dwpc_lookup: dict,
    dwpc_thresholds: dict,
    dwpc_z_stats: dict,
    args: argparse.Namespace,
):
    print(f"\n{'='*60}\nProcessing B = {b_value}\n{'='*60}")

    b_out_dir = out_dir / f"b{b_value}"
    b_out_dir.mkdir(parents=True, exist_ok=True)

    results_frames = []
    for runs_path in runs_paths:
        try:
            results_frames.append(
                load_runs_at_b(runs_path, b_value,
                               ["lv_id", "target_id", "target_name", "node_type", "metapath"])
            )
        except ValueError as e:
            print(f"  Warning: {e}")

    if not results_frames:
        print(f"No results found for B = {b_value}, skipping")
        return maps

    results_df = pd.concat(results_frames, ignore_index=True).drop_duplicates()
    print(f"Loaded {len(results_df)} metapath results at b={b_value}")

    selected_mp, dropped_lvs = _select_metapaths_by_effect_size(
        results_df, z_threshold=args.effect_size_threshold
    )

    if not dropped_lvs.empty:
        dropped_lvs.to_csv(b_out_dir / "dropped_lvs.csv", index=False)
        print(f"Dropped {len(dropped_lvs)} LVs (no metapaths with z > {args.effect_size_threshold})")

    if selected_mp.empty:
        print(f"No metapaths with z > {args.effect_size_threshold}")
        return maps

    print(f"Selected {len(selected_mp)} metapaths across {selected_mp['lv_id'].nunique()} LVs")
    selected_mp.to_csv(b_out_dir / "selected_metapaths.csv", index=False)

    if maps is None:
        all_node_types = set()
        for mp in selected_mp["metapath"].unique():
            nodes, _ = parse_metapath(mp)
            all_node_types.update(nodes)
        maps = load_node_maps(REPO_ROOT, list(all_node_types))

    sharing_rows: list[dict] = []
    top_intermediates_rows: list[dict] = []
    gene_path_records: list[dict] = []

    for lv_id, group in selected_mp.groupby("lv_id"):
        s, t, g = _process_lv(
            lv_id, group, top_genes, lv_targets,
            edge_loader, maps, node_name_maps,
            dwpc_lookup, dwpc_thresholds, dwpc_z_stats, args,
        )
        sharing_rows.extend(s)
        top_intermediates_rows.extend(t)
        gene_path_records.extend(g)

    sharing_df = pd.DataFrame(sharing_rows)
    sharing_df["b"] = b_value
    sharing_df = sharing_df[[c for c in SHARING_COL_ORDER if c in sharing_df.columns]]
    sharing_df.to_csv(b_out_dir / "intermediate_sharing_by_metapath.csv", index=False)

    if top_intermediates_rows:
        top_int_df = pd.DataFrame(top_intermediates_rows)
        top_int_df["b"] = b_value
        top_int_df.to_csv(b_out_dir / "top_intermediates_by_metapath.csv", index=False)

    if gene_path_records:
        gene_paths_df = pd.DataFrame(gene_path_records)
        gene_paths_df["b"] = b_value
        gene_paths_df.to_csv(b_out_dir / "gene_paths.csv", index=False)

    summary = sharing_df.groupby(["lv_id", "target_id", "target_name", "node_type"]).agg(
        n_metapaths=("metapath", "count"),
        n_genes_total=("n_genes_total", "first"),
        median_effect_size_z=("effect_size_z", "median"),
        max_effect_size_z=("effect_size_z", "max"),
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

    print(f"Saved results to {b_out_dir}/")
    return maps


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_paths = []
    top_genes_frames = []
    lv_targets_frames = []
    dwpc_lookup: dict[tuple, float] = {}

    for lv_output_dir_str in args.lv_output_dirs:
        lv_output_dir = Path(lv_output_dir_str)
        runs_path = lv_output_dir / "lv_rank_stability_experiment" / "all_runs_long.csv"
        if not runs_path.exists():
            print(f"Warning: {runs_path} not found, skipping")
            continue
        runs_paths.append(runs_path)
        top_genes_frames.append(pd.read_csv(lv_output_dir / "lv_top_genes.csv"))
        lv_targets_frames.append(pd.read_csv(lv_output_dir / "lv_targets.csv"))
        lv_dwpc = load_dwpc_from_numpy(lv_output_dir, analysis_type="lv")
        dwpc_lookup.update(lv_dwpc)

    if not runs_paths:
        print("No valid LV output directories found.")
        return

    top_genes = pd.concat(top_genes_frames, ignore_index=True).drop_duplicates()
    lv_targets = pd.concat(lv_targets_frames, ignore_index=True).drop_duplicates()

    dwpc_thresholds: dict = {}
    dwpc_z_stats: dict = {}
    if dwpc_lookup:
        if args.dwpc_z_threshold is not None:
            dwpc_z_stats = compute_dwpc_z_stats(dwpc_lookup)
            print(f"DWPC z-filter active: keep genes with z >= {args.dwpc_z_threshold}")
        elif args.dwpc_percentile > 0:
            dwpc_thresholds = compute_dwpc_thresholds(dwpc_lookup, args.dwpc_percentile)

    if args.b_values:
        b_values = [int(b.strip()) for b in args.b_values.split(",")]
    elif args.b is not None:
        b_values = [args.b]
    else:
        b_values = [10]

    node_name_maps = load_node_names(REPO_ROOT)
    edge_loader = EdgeLoader(REPO_ROOT / "data" / "edges")
    maps = None

    for b_value in b_values:
        maps = _process_b_value(
            b_value, runs_paths, out_dir,
            top_genes, lv_targets,
            edge_loader, maps, node_name_maps,
            dwpc_lookup, dwpc_thresholds, dwpc_z_stats, args,
        )

    print("\nAll B values processed.")


if __name__ == "__main__":
    main()
