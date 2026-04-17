#!/usr/bin/env python3
"""Compute intermediate sharing between 2016 and 2024-added genes per GO term.

Updated version with:
- Effect size d > threshold for metapath selection (from rank stability experiments)
- Multi-B support
- DWPC percentile filtering

For each (GO term, metapath) with effect size d > threshold:
1. Identify genes annotated in 2016 vs genes added by 2024
2. Enumerate path instances for both gene sets
3. Compute:
   - % of 2016 genes sharing intermediates with other 2016 genes
   - % of 2024-added genes sharing intermediates with 2016 genes
   - Intermediate coverage metrics

Usage:
    # Single B value
    python scripts/year_intermediate_sharing.py --b 10 --output-dir output/year_int_share

    # Multiple B values
    python scripts/year_intermediate_sharing.py --b-values 2,5,10,20,30 --output-dir output/year_int_share

    # Single GO term (for HPC array jobs)
    python scripts/year_intermediate_sharing.py --b 10 --go-id GO:0001234 --output-dir output/year_int_share
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
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
) -> pd.DataFrame:
    """Select metapaths by effect size (Cohen's d) from permutation null.

    For year analysis, select metapaths that meet threshold in BOTH 2016 and 2024.
    """
    if results_df.empty:
        return pd.DataFrame()

    # Check if year column exists
    if "year" in results_df.columns:
        # Get metapaths meeting threshold in each year
        above_2016 = results_df[
            (results_df["year"] == 2016) & (results_df["effect_size_d"] > d_threshold)
        ][["go_id", "metapath"]].drop_duplicates()
        above_2016["in_2016"] = True

        above_2024 = results_df[
            (results_df["year"] == 2024) & (results_df["effect_size_d"] > d_threshold)
        ][["go_id", "metapath"]].drop_duplicates()
        above_2024["in_2024"] = True

        # Keep only those meeting threshold in both years
        consensus = above_2016.merge(above_2024, on=["go_id", "metapath"], how="inner")

        # Add effect size from 2016 for ranking
        results_2016 = results_df[results_df["year"] == 2016].copy()
        consensus = consensus.merge(
            results_2016[["go_id", "metapath", "effect_size_d"]],
            on=["go_id", "metapath"],
            how="left",
        )

        # Rank within each GO term
        consensus = consensus.sort_values(
            ["go_id", "effect_size_d"], ascending=[True, False]
        )
        consensus["metapath_rank"] = consensus.groupby("go_id").cumcount() + 1

        return consensus[["go_id", "metapath", "metapath_rank", "effect_size_d"]]

    else:
        # No year column - simple filtering
        above = results_df[results_df["effect_size_d"] > d_threshold].copy()
        above = above.sort_values(["go_id", "effect_size_d"], ascending=[True, False])
        above["metapath_rank"] = above.groupby("go_id").cumcount() + 1
        return above[["go_id", "metapath", "metapath_rank", "effect_size_d"]]


def _compute_sharing_stats(
    genes_2016_intermediates: dict[int, set[str]],
    genes_2024_intermediates: dict[int, set[str]],
    n_genes_2016_total: int,
    n_genes_2024_total: int,
) -> dict:
    """Compute sharing statistics between 2016 and 2024-added genes."""
    # All intermediates from 2016 genes
    all_2016_intermediates = set()
    for ints in genes_2016_intermediates.values():
        all_2016_intermediates.update(ints)

    # 2016-2016 sharing
    n_2016 = len(genes_2016_intermediates)
    n_2016_sharing = 0
    jaccard_2016_to_2016 = []
    for gene_id, ints in genes_2016_intermediates.items():
        other_ints = set()
        for other_gene, other_gene_ints in genes_2016_intermediates.items():
            if other_gene != gene_id:
                other_ints.update(other_gene_ints)
        if ints & other_ints:
            n_2016_sharing += 1
        union = ints | other_ints
        if union:
            jaccard_2016_to_2016.append(len(ints & other_ints) / len(union))

    # 2024-2016 sharing
    n_2024 = len(genes_2024_intermediates)
    n_2024_sharing_with_2016 = 0
    jaccard_2024_to_2016 = []
    for gene_id, ints in genes_2024_intermediates.items():
        if ints & all_2016_intermediates:
            n_2024_sharing_with_2016 += 1
        union = ints | all_2016_intermediates
        if union:
            jaccard_2024_to_2016.append(len(ints & all_2016_intermediates) / len(union))

    # 2024-2024 sharing
    n_2024_sharing_with_2024 = 0
    jaccard_2024_to_2024 = []
    for gene_id, ints in genes_2024_intermediates.items():
        other_ints = set()
        for other_gene, other_gene_ints in genes_2024_intermediates.items():
            if other_gene != gene_id:
                other_ints.update(other_gene_ints)
        if ints & other_ints:
            n_2024_sharing_with_2024 += 1
        union = ints | other_ints
        if union:
            jaccard_2024_to_2024.append(len(ints & other_ints) / len(union))

    all_2024_intermediates = set()
    for ints in genes_2024_intermediates.values():
        all_2024_intermediates.update(ints)

    return {
        "n_genes_2016": n_genes_2016_total,
        "n_genes_2016_with_paths": n_2016,
        "n_genes_2016_sharing_with_2016": n_2016_sharing,
        "pct_2016_sharing_with_2016": n_2016_sharing / n_2016 * 100 if n_2016 > 0 else 0.0,
        "median_jaccard_2016_to_2016": float(np.median(jaccard_2016_to_2016)) if jaccard_2016_to_2016 else 0.0,
        "n_genes_2024_added": n_genes_2024_total,
        "n_genes_2024_with_paths": n_2024,
        "n_genes_2024_sharing_with_2016": n_2024_sharing_with_2016,
        "pct_2024_sharing_with_2016": n_2024_sharing_with_2016 / n_2024 * 100 if n_2024 > 0 else 0.0,
        "median_jaccard_2024_to_2016": float(np.median(jaccard_2024_to_2016)) if jaccard_2024_to_2016 else 0.0,
        "n_genes_2024_sharing_with_2024": n_2024_sharing_with_2024,
        "pct_2024_sharing_with_2024": n_2024_sharing_with_2024 / n_2024 * 100 if n_2024 > 0 else 0.0,
        "median_jaccard_2024_to_2024": float(np.median(jaccard_2024_to_2024)) if jaccard_2024_to_2024 else 0.0,
        "n_unique_intermediates_2016": len(all_2016_intermediates),
        "n_unique_intermediates_2024": len(all_2024_intermediates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year-output-dir",
        default="output/year_experiment",
        help="Path to year experiment output directory",
    )
    parser.add_argument(
        "--added-pairs-path",
        default="output/intermediate/upd_go_bp_2024_added.csv",
        help="Path to CSV with 2024-added GO-gene pairs",
    )
    parser.add_argument(
        "--b",
        type=int,
        default=None,
        help="Single B value for metapath selection",
    )
    parser.add_argument(
        "--b-values",
        default=None,
        help="Comma-separated B values (e.g., '2,5,10,20,30')",
    )
    parser.add_argument(
        "--effect-size-threshold",
        type=float,
        default=0.5,
        help="Minimum effect size (Cohen's d) for metapath selection (default: 0.5, medium effect)",
    )
    parser.add_argument(
        "--dwpc-percentile",
        type=float,
        default=75.0,
        help="DWPC percentile threshold (default 75 = top 25%%)",
    )
    parser.add_argument(
        "--path-top-k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--go-id",
        default=None,
        help="Run single GO term (for HPC array jobs)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/year_intermediate_sharing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    year_output_dir = Path(args.year_output_dir)
    added_path = Path(args.added_pairs_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine B values
    if args.b_values:
        b_values = [int(b.strip()) for b in args.b_values.split(",")]
        print(f"Multi-B mode: {b_values}")
    elif args.b is not None:
        b_values = [args.b]
    else:
        b_values = [10]
        print("Using default B = 10")

    # Load 2024-added pairs
    if added_path.exists():
        added_df = pd.read_csv(added_path)
        added_pairs = set(zip(added_df["go_id"], added_df["entrez_gene_id"].astype(int)))
        print(f"Loaded {len(added_pairs):,} added GO-gene pairs")
    else:
        print(f"Warning: {added_path} not found")
        added_pairs = set()

    # Load GO-gene annotations for 2016
    go_genes_path = year_output_dir / "go_top_genes.csv"
    if go_genes_path.exists():
        go_genes_df = pd.read_csv(go_genes_path)
        print(f"Loaded {len(go_genes_df)} GO-gene annotations")
    else:
        print(f"Error: {go_genes_path} not found")
        return

    # Load node name mappings
    node_name_maps = load_node_names(REPO_ROOT)
    print(f"Loaded node name maps for: {list(node_name_maps.keys())}")

    # Setup edge loader
    edges_dir = REPO_ROOT / "data" / "edges"
    edge_loader = EdgeLoader(edges_dir)

    # Load DWPC values (placeholder - would need actual DWPC data)
    dwpc_lookup: dict[tuple, float] = {}
    dwpc_thresholds: dict[tuple, float] = {}

    if args.dwpc_percentile > 0 and dwpc_lookup:
        dwpc_thresholds = compute_dwpc_thresholds(dwpc_lookup, args.dwpc_percentile)
        print(f"Computed DWPC thresholds for {len(dwpc_thresholds)} pairs")

    # Initialize node maps (will be populated on first iteration)
    maps = None

    # Process each B value
    for b_value in b_values:
        print(f"\n{'='*60}")
        print(f"Processing B = {b_value}")
        print("=" * 60)

        # Always write to a b{value}/ subdirectory so downstream consumers
        # (generate_gene_table, plot_metapath_subgraphs, global summary) can
        # find outputs uniformly regardless of single- vs multi-B runs.
        b_out_dir = out_dir / f"b{b_value}"
        b_out_dir.mkdir(parents=True, exist_ok=True)

        # Load results at this B
        runs_path = year_output_dir / "year_rank_stability_experiment" / "all_runs_long.csv"
        if not runs_path.exists():
            print(f"Warning: {runs_path} not found, skipping B={b_value}")
            continue

        try:
            results_df = load_runs_at_b(runs_path, b_value, ["go_id", "metapath", "year"])
        except ValueError as e:
            print(f"Warning: {e}")
            continue

        # Select metapaths by effect size
        selected_mp = _select_metapaths_by_effect_size(
            results_df, d_threshold=args.effect_size_threshold
        )

        if selected_mp.empty:
            print(f"No metapaths with d > {args.effect_size_threshold}")
            continue

        # Filter to single GO term if specified
        if args.go_id:
            selected_mp = selected_mp[selected_mp["go_id"] == args.go_id]
            if selected_mp.empty:
                print(f"GO term {args.go_id} not found")
                continue

        print(f"Selected {len(selected_mp)} metapaths across {selected_mp['go_id'].nunique()} GO terms")

        # Save selected metapaths
        selected_mp.to_csv(b_out_dir / "selected_metapaths.csv", index=False)

        # Load node maps on first iteration
        if maps is None:
            all_node_types = set()
            for mp in selected_mp["metapath"].unique():
                nodes, _ = parse_metapath(mp)
                all_node_types.update(nodes)
            print(f"Loading node maps for: {sorted(all_node_types)}")
            maps = load_node_maps(REPO_ROOT, list(all_node_types))

        # Process each GO term
        sharing_rows = []
        top_intermediates_rows = []
        gene_path_records: list[dict] = []

        for go_id in selected_mp["go_id"].unique():
            go_mps = selected_mp[selected_mp["go_id"] == go_id]

            # Get genes for this GO term
            go_gene_df = go_genes_df[go_genes_df["go_id"] == go_id]
            all_genes = go_gene_df["gene_identifier"].astype(int).tolist()

            # Split into 2016 (baseline) and 2024-added
            genes_2016 = [g for g in all_genes if (go_id, g) not in added_pairs]
            genes_2024 = [g for g in all_genes if (go_id, g) in added_pairs]

            print(f"\n{go_id}: {len(genes_2016)} genes (2016), {len(genes_2024)} added (2024), {len(go_mps)} metapaths")

            bp_pos = maps.id_to_pos.get("BP", {}).get(go_id)
            if bp_pos is None:
                print(f"  Warning: BP position not found for {go_id}, skipping")
                continue

            for _, mp_row in go_mps.iterrows():
                metapath = mp_row["metapath"]
                mp_rank = mp_row["metapath_rank"]
                effect_size = mp_row["effect_size_d"]

                ints_2016 = enumerate_gene_intermediates(
                    genes_2016, bp_pos, metapath, edge_loader, maps,
                    path_top_k=args.path_top_k,
                    gene_set_id=go_id,
                    dwpc_lookup=dwpc_lookup or None,
                    dwpc_thresholds=dwpc_thresholds or None,
                    record_paths=gene_path_records,
                    record_extra={"go_id": go_id, "year": "2016"},
                )

                ints_2024 = enumerate_gene_intermediates(
                    genes_2024, bp_pos, metapath, edge_loader, maps,
                    path_top_k=args.path_top_k,
                    gene_set_id=go_id,
                    dwpc_lookup=dwpc_lookup or None,
                    dwpc_thresholds=dwpc_thresholds or None,
                    record_paths=gene_path_records,
                    record_extra={"go_id": go_id, "year": "2024"},
                )

                # Compute sharing stats
                stats = _compute_sharing_stats(
                    ints_2016, ints_2024, len(genes_2016), len(genes_2024)
                )

                # Compute coverage for combined gene set
                combined_ints = {**ints_2016, **ints_2024}
                coverage_stats, int_stats = compute_intermediate_coverage(
                    combined_ints, node_name_maps
                )
                stats.update(coverage_stats)

                stats.update({
                    "go_id": go_id,
                    "metapath": metapath,
                    "metapath_rank": mp_rank,
                    "effect_size_d": effect_size,
                    "b": b_value,
                })
                sharing_rows.append(stats)

                # Store top intermediates
                for rank, int_stat in enumerate(int_stats, 1):
                    top_intermediates_rows.append({
                        "go_id": go_id,
                        "metapath": metapath,
                        "intermediate_rank": rank,
                        "intermediate_id": int_stat["intermediate_id"],
                        "intermediate_name": int_stat.get("intermediate_name"),
                        "n_genes_using": int_stat["n_genes_using"],
                        "pct_genes_using": int_stat["pct_genes_using"],
                    })

                print(f"  {metapath} (d={effect_size:.2f}): "
                      f"2016={len(ints_2016)}/{len(genes_2016)}, "
                      f"2024={len(ints_2024)}/{len(genes_2024)}")

        # Save results
        if sharing_rows:
            sharing_df = pd.DataFrame(sharing_rows)
            sharing_df.to_csv(b_out_dir / "intermediate_sharing_by_metapath.csv", index=False)
            print(f"\nSaved: {b_out_dir / 'intermediate_sharing_by_metapath.csv'}")

        if top_intermediates_rows:
            top_int_df = pd.DataFrame(top_intermediates_rows)
            top_int_df.to_csv(b_out_dir / "top_intermediates_by_metapath.csv", index=False)
            print(f"Saved: {b_out_dir / 'top_intermediates_by_metapath.csv'}")

        # Persist per-gene per-path node sequences for downstream visualization.
        if gene_path_records:
            gene_paths_df = pd.DataFrame(gene_path_records)
            gene_paths_df["b"] = b_value
            gene_paths_df.to_csv(b_out_dir / "gene_paths.csv", index=False)
            print(f"Saved: {b_out_dir / 'gene_paths.csv'}")

        # Aggregate summary
        if sharing_rows:
            sharing_df = pd.DataFrame(sharing_rows)
            summary = sharing_df.groupby("go_id").agg(
                n_metapaths=("metapath", "count"),
                n_genes_2016=("n_genes_2016", "first"),
                n_genes_2024_added=("n_genes_2024_added", "first"),
                median_pct_2024_sharing_with_2016=("pct_2024_sharing_with_2016", "median"),
                median_jaccard_2024_to_2016=("median_jaccard_2024_to_2016", "median"),
                median_top1_coverage=("top1_intermediate_coverage", "median"),
            ).reset_index()
            summary["b"] = b_value
            summary.to_csv(b_out_dir / "intermediate_sharing_summary.csv", index=False)
            print(f"Saved: {b_out_dir / 'intermediate_sharing_summary.csv'}")

    print("\n" + "=" * 60)
    print("All B values processed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
