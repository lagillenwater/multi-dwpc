#!/usr/bin/env python3
"""Compute intermediate sharing between 2016 and 2024-added genes per GO term.

Updated version with:
- Permutation z > threshold for metapath selection (from rank stability experiments)
- Multi-B support
- DWPC percentile filtering

For each (GO term, metapath) with permutation z > threshold:
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
    z_threshold: float = 0.2,
) -> pd.DataFrame:
    """Select metapaths by permutation z-statistic from permutation null.

    For year analysis, select metapaths that meet threshold in BOTH 2016 and 2024.
    """
    if results_df.empty:
        return pd.DataFrame()

    # Check if year column exists
    if "year" in results_df.columns:
        # Get metapaths meeting threshold in each year
        above_2016 = results_df[
            (results_df["year"] == 2016) & (results_df["permutation_z"] > z_threshold)
        ][["go_id", "metapath"]].drop_duplicates()
        above_2016["in_2016"] = True

        above_2024 = results_df[
            (results_df["year"] == 2024) & (results_df["permutation_z"] > z_threshold)
        ][["go_id", "metapath"]].drop_duplicates()
        above_2024["in_2024"] = True

        # Keep only those meeting threshold in both years
        consensus = above_2016.merge(above_2024, on=["go_id", "metapath"], how="inner")

        # Add effect size from 2016 for ranking
        results_2016 = results_df[results_df["year"] == 2016].copy()
        consensus = consensus.merge(
            results_2016[["go_id", "metapath", "permutation_z"]],
            on=["go_id", "metapath"],
            how="left",
        )

        # Rank within each GO term
        consensus = consensus.sort_values(
            ["go_id", "permutation_z"], ascending=[True, False]
        )
        consensus["metapath_rank"] = consensus.groupby("go_id").cumcount() + 1

        return consensus[["go_id", "metapath", "metapath_rank", "permutation_z"]]

    else:
        # No year column - simple filtering
        above = results_df[results_df["permutation_z"] > z_threshold].copy()
        above = above.sort_values(["go_id", "permutation_z"], ascending=[True, False])
        above["metapath_rank"] = above.groupby("go_id").cumcount() + 1
        return above[["go_id", "metapath", "metapath_rank", "permutation_z"]]


def _within_group_jaccard(gene_intermediates: dict[int, set[str]]) -> tuple[int, list[float]]:
    """For each gene, compute Jaccard vs the union of all other genes' intermediates.

    Returns (n_sharing, jaccard_scores) where n_sharing is the count of genes
    that share at least one intermediate with another gene in the group.
    """
    n_sharing = 0
    jaccard_scores: list[float] = []
    all_others_cache: set[str] | None = None
    if len(gene_intermediates) > 1:
        all_ints = set()
        for v in gene_intermediates.values():
            all_ints.update(v)
        all_others_cache = all_ints

    for gene_id, ints in gene_intermediates.items():
        if all_others_cache is not None:
            other_ints = all_others_cache - ints
        else:
            other_ints = set()
        if ints & other_ints:
            n_sharing += 1
        union = ints | other_ints
        if union:
            jaccard_scores.append(len(ints & other_ints) / len(union))
    return n_sharing, jaccard_scores


def _pct(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator > 0 else 0.0


def _median_or_zero(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _compute_sharing_stats(
    genes_2016_intermediates: dict[int, set[str]],
    genes_2024_intermediates: dict[int, set[str]],
    n_genes_2016_total: int,
    n_genes_2024_total: int,
) -> dict:
    all_2016_intermediates: set[str] = set()
    for ints in genes_2016_intermediates.values():
        all_2016_intermediates.update(ints)

    all_2024_intermediates: set[str] = set()
    for ints in genes_2024_intermediates.values():
        all_2024_intermediates.update(ints)

    n_2016 = len(genes_2016_intermediates)
    n_2024 = len(genes_2024_intermediates)

    n_2016_sharing, jaccard_2016_to_2016 = _within_group_jaccard(genes_2016_intermediates)
    n_2024_sharing_2024, jaccard_2024_to_2024 = _within_group_jaccard(genes_2024_intermediates)

    n_2024_sharing_with_2016 = 0
    jaccard_2024_to_2016: list[float] = []
    for ints in genes_2024_intermediates.values():
        if ints & all_2016_intermediates:
            n_2024_sharing_with_2016 += 1
        union = ints | all_2016_intermediates
        if union:
            jaccard_2024_to_2016.append(len(ints & all_2016_intermediates) / len(union))

    return {
        "n_genes_2016": n_genes_2016_total,
        "n_genes_2016_with_paths": n_2016,
        "n_genes_2016_sharing_with_2016": n_2016_sharing,
        "pct_2016_sharing_with_2016": _pct(n_2016_sharing, n_2016),
        "median_jaccard_2016_to_2016": _median_or_zero(jaccard_2016_to_2016),
        "n_genes_2024_added": n_genes_2024_total,
        "n_genes_2024_with_paths": n_2024,
        "n_genes_2024_sharing_with_2016": n_2024_sharing_with_2016,
        "pct_2024_sharing_with_2016": _pct(n_2024_sharing_with_2016, n_2024),
        "median_jaccard_2024_to_2016": _median_or_zero(jaccard_2024_to_2016),
        "n_genes_2024_sharing_with_2024": n_2024_sharing_2024,
        "pct_2024_sharing_with_2024": _pct(n_2024_sharing_2024, n_2024),
        "median_jaccard_2024_to_2024": _median_or_zero(jaccard_2024_to_2024),
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
        help="Minimum permutation z-statistic (diff / null_std) for metapath selection (default: 0.5)",
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


def _process_go_term(
    go_id: str,
    go_mps: pd.DataFrame,
    go_genes_df: pd.DataFrame,
    added_pairs: set,
    bp_pos: int,
    b_value: int,
    edge_loader,
    maps,
    node_name_maps: dict,
    dwpc_lookup: dict,
    dwpc_thresholds: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict]]:
    all_genes = go_genes_df.loc[
        go_genes_df["go_id"] == go_id, "gene_identifier"
    ].astype(int).tolist()
    genes_2016 = [g for g in all_genes if (go_id, g) not in added_pairs]
    genes_2024 = [g for g in all_genes if (go_id, g) in added_pairs]

    print(f"\n{go_id}: {len(genes_2016)} genes (2016), {len(genes_2024)} added (2024), {len(go_mps)} metapaths")

    sharing_rows: list[dict] = []
    top_intermediates_rows: list[dict] = []
    gene_path_records: list[dict] = []

    for _, mp_row in go_mps.iterrows():
        metapath = mp_row["metapath"]
        mp_rank = mp_row["metapath_rank"]
        z_score = mp_row["permutation_z"]

        enum_kwargs = dict(
            target_pos=bp_pos, metapath=metapath,
            edge_loader=edge_loader, maps=maps,
            path_top_k=args.path_top_k,
            gene_set_id=go_id,
            dwpc_lookup=dwpc_lookup or None,
            dwpc_thresholds=dwpc_thresholds or None,
            record_paths=gene_path_records,
        )

        ints_2016 = enumerate_gene_intermediates(
            genes_2016, **enum_kwargs,
            record_extra={"go_id": go_id, "year": "2016"},
        )
        ints_2024 = enumerate_gene_intermediates(
            genes_2024, **enum_kwargs,
            record_extra={"go_id": go_id, "year": "2024"},
        )

        stats = _compute_sharing_stats(ints_2016, ints_2024, len(genes_2016), len(genes_2024))
        coverage_stats, int_stats = compute_intermediate_coverage(
            {**ints_2016, **ints_2024}, node_name_maps
        )
        stats.update(coverage_stats)
        stats.update({
            "go_id": go_id,
            "metapath": metapath,
            "metapath_rank": mp_rank,
            "permutation_z": z_score,
            "b": b_value,
        })
        sharing_rows.append(stats)

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

        print(f"  {metapath} (z={z_score:.2f}): "
              f"2016={len(ints_2016)}/{len(genes_2016)}, "
              f"2024={len(ints_2024)}/{len(genes_2024)}")

    return sharing_rows, top_intermediates_rows, gene_path_records


def _process_b_value(
    b_value: int,
    runs_path: Path,
    out_dir: Path,
    go_genes_df: pd.DataFrame,
    added_pairs: set,
    edge_loader,
    maps,
    node_name_maps: dict,
    dwpc_lookup: dict,
    dwpc_thresholds: dict,
    args: argparse.Namespace,
):
    print(f"\n{'='*60}\nProcessing B = {b_value}\n{'='*60}")

    b_out_dir = out_dir / f"b{b_value}"
    b_out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_path.exists():
        print(f"Warning: {runs_path} not found, skipping B={b_value}")
        return maps

    try:
        results_df = load_runs_at_b(runs_path, b_value, ["go_id", "metapath", "year"])
    except ValueError as e:
        print(f"Warning: {e}")
        return maps

    selected_mp = _select_metapaths_by_effect_size(
        results_df, z_threshold=args.effect_size_threshold
    )
    if selected_mp.empty:
        print(f"No metapaths with z > {args.effect_size_threshold}")
        return maps

    if args.go_id:
        selected_mp = selected_mp[selected_mp["go_id"] == args.go_id]
        if selected_mp.empty:
            print(f"GO term {args.go_id} not found")
            return maps

    print(f"Selected {len(selected_mp)} metapaths across {selected_mp['go_id'].nunique()} GO terms")
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

    for go_id in selected_mp["go_id"].unique():
        bp_pos = maps.id_to_pos.get("BP", {}).get(go_id)
        if bp_pos is None:
            print(f"  Warning: BP position not found for {go_id}, skipping")
            continue

        s, t, g = _process_go_term(
            go_id, selected_mp[selected_mp["go_id"] == go_id],
            go_genes_df, added_pairs, bp_pos, b_value,
            edge_loader, maps, node_name_maps,
            dwpc_lookup, dwpc_thresholds, args,
        )
        sharing_rows.extend(s)
        top_intermediates_rows.extend(t)
        gene_path_records.extend(g)

    if sharing_rows:
        sharing_df = pd.DataFrame(sharing_rows)
        sharing_df.to_csv(b_out_dir / "intermediate_sharing_by_metapath.csv", index=False)

    if top_intermediates_rows:
        pd.DataFrame(top_intermediates_rows).to_csv(
            b_out_dir / "top_intermediates_by_metapath.csv", index=False
        )

    if gene_path_records:
        gene_paths_df = pd.DataFrame(gene_path_records)
        gene_paths_df["b"] = b_value
        gene_paths_df.to_csv(b_out_dir / "gene_paths.csv", index=False)

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

    print(f"Saved results to {b_out_dir}/")
    return maps


def main() -> None:
    args = parse_args()
    year_output_dir = Path(args.year_output_dir)
    added_path = Path(args.added_pairs_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.b_values:
        b_values = [int(b.strip()) for b in args.b_values.split(",")]
    elif args.b is not None:
        b_values = [args.b]
    else:
        b_values = [10]

    if added_path.exists():
        added_df = pd.read_csv(added_path)
        added_pairs = set(zip(added_df["go_id"], added_df["entrez_gene_id"].astype(int)))
    else:
        print(f"Warning: {added_path} not found")
        added_pairs = set()

    go_genes_path = year_output_dir / "go_top_genes.csv"
    if not go_genes_path.exists():
        print(f"Error: {go_genes_path} not found")
        return
    go_genes_df = pd.read_csv(go_genes_path)

    node_name_maps = load_node_names(REPO_ROOT)
    edge_loader = EdgeLoader(REPO_ROOT / "data" / "edges")

    dwpc_lookup: dict[tuple, float] = {}
    dwpc_thresholds: dict[tuple, float] = {}
    if args.dwpc_percentile > 0 and dwpc_lookup:
        dwpc_thresholds = compute_dwpc_thresholds(dwpc_lookup, args.dwpc_percentile)

    runs_path = year_output_dir / "year_rank_stability_experiment" / "all_runs_long.csv"
    maps = None

    for b_value in b_values:
        maps = _process_b_value(
            b_value, runs_path, out_dir,
            go_genes_df, added_pairs,
            edge_loader, maps, node_name_maps,
            dwpc_lookup, dwpc_thresholds, args,
        )

    print("\nAll B values processed.")


if __name__ == "__main__":
    main()
