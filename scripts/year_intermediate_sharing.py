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
from typing import Dict, Tuple

import numpy as np
import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_top_paths_local import (  # noqa: E402
    EdgeLoader,
    enumerate_paths,
    load_node_maps,
    parse_metapath,
    reverse_metapath_abbrev,
    select_paths,
)


# Node type abbreviation to full name mapping
NODE_TYPE_NAMES = {
    "G": "Gene",
    "A": "Anatomy",
    "BP": "Biological Process",
    "CC": "Cellular Component",
    "C": "Compound",
    "D": "Disease",
    "MF": "Molecular Function",
    "PC": "Pharmacologic Class",
    "PW": "Pathway",
    "SE": "Side Effect",
    "S": "Symptom",
}


def _load_node_names(repo_root: Path) -> dict[str, dict[str, str]]:
    """Load node ID to name mappings for all node types."""
    nodes_dir = repo_root / "data" / "nodes"
    name_maps: dict[str, dict[str, str]] = {}

    for abbrev, full_name in NODE_TYPE_NAMES.items():
        node_file = nodes_dir / f"{full_name}.tsv"
        if not node_file.exists():
            continue
        df = pd.read_csv(node_file, sep="\t")
        name_maps[abbrev] = dict(zip(df["identifier"].astype(str), df["name"]))

    return name_maps


def _load_runs_at_b(runs_path: Path, b: int) -> pd.DataFrame:
    """Load all_runs_long.csv and get effect sizes from permutation null."""
    runs_df = pd.read_csv(runs_path)

    # Filter to specified b and permutation null only
    runs_df = runs_df[(runs_df["b"] == b) & (runs_df["control"] == "permuted")].copy()
    if runs_df.empty:
        raise ValueError(f"No permuted rows found for b={b} in {runs_path}")

    # Average effect size d across seeds
    group_cols = ["go_id", "metapath"]
    if "year" in runs_df.columns:
        group_cols.append("year")

    if "d" in runs_df.columns:
        result = runs_df.groupby(group_cols, as_index=False).agg(
            effect_size_d=("d", "mean"),
            diff_perm=("diff", "mean"),
        )
    else:
        result = runs_df.groupby(group_cols, as_index=False).agg(
            diff_perm=("diff", "mean"),
        )
        result["effect_size_d"] = result["diff_perm"]

    return result


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


def _compute_dwpc_thresholds(
    dwpc_lookup: Dict[Tuple, float],
    percentile: float,
) -> Dict[Tuple[str, str], float]:
    """Compute DWPC percentile thresholds for each (go_id, metapath)."""
    grouped: Dict[Tuple[str, str], list] = {}
    for (go_id, metapath, gene_id), dwpc in dwpc_lookup.items():
        key = (go_id, metapath)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(dwpc)

    thresholds: Dict[Tuple[str, str], float] = {}
    for key, values in grouped.items():
        if values:
            thresholds[key] = float(np.percentile(values, percentile))

    return thresholds


def _enumerate_gene_intermediates(
    genes: list[int],
    go_id: str,
    metapath: str,
    edge_loader: EdgeLoader,
    maps,
    *,
    path_top_k: int = 100,
    degree_d: float = 0.5,
    dwpc_lookup: dict[tuple, float] | None = None,
    dwpc_thresholds: dict[tuple, float] | None = None,
) -> dict[int, set[str]]:
    """Enumerate paths for genes and return {gene_id: set of intermediate_ids}."""
    # Metapath goes Gene -> ... -> BP, reverse for enumeration
    reversed_mp = reverse_metapath_abbrev(metapath)
    nodes, edges = parse_metapath(reversed_mp)

    bp_pos = maps.id_to_pos.get("BP", {}).get(go_id)
    if bp_pos is None:
        return {}

    gene_intermediates: dict[int, set[str]] = {}

    # Get DWPC threshold for this (go_id, metapath)
    dwpc_threshold = 0.0
    if dwpc_thresholds is not None:
        dwpc_threshold = dwpc_thresholds.get((go_id, metapath), 0.0)

    for gene_id in genes:
        gene_pos = maps.id_to_pos.get("G", {}).get(gene_id)
        if gene_pos is None:
            continue

        # Filter by DWPC threshold
        if dwpc_lookup is not None and dwpc_threshold > 0:
            dwpc = dwpc_lookup.get((go_id, metapath, gene_id), 0.0)
            if dwpc < dwpc_threshold:
                continue

        try:
            candidate_paths = enumerate_paths(
                bp_pos, gene_pos, nodes, edges, edge_loader,
                top_k=path_top_k, degree_d=degree_d,
            )
            paths = select_paths(
                candidate_paths,
                selection_method="effective_number",
                top_paths=path_top_k,
                path_cumulative_frac=None,
                path_min_count=1,
                path_max_count=None,
            )
        except Exception:
            continue

        intermediates = set()
        for score, pos_path in paths:
            for i, (node_type, pos) in enumerate(zip(nodes, pos_path)):
                if 0 < i < len(nodes) - 1:
                    node_id = maps.pos_to_id.get(node_type, {}).get(int(pos))
                    if node_id is not None:
                        intermediates.add(f"{node_type}:{node_id}")

        if intermediates:
            gene_intermediates[gene_id] = intermediates

    return gene_intermediates


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


def _compute_intermediate_coverage(
    gene_intermediates: dict[int, set[str]],
    node_name_maps: dict[str, dict[str, str]] | None = None,
) -> tuple[dict, list[dict]]:
    """Compute intermediate coverage statistics."""
    n_genes = len(gene_intermediates)
    if n_genes == 0:
        return {
            "top1_intermediate_coverage": 0.0,
            "top5_intermediate_coverage": 0.0,
            "pct_intermediates_shared_quarter": 0.0,
            "pct_intermediates_shared_majority": 0.0,
        }, []

    # Count genes using each intermediate
    intermediate_gene_counts: dict[str, set[int]] = {}
    for gene_id, ints in gene_intermediates.items():
        for int_id in ints:
            if int_id not in intermediate_gene_counts:
                intermediate_gene_counts[int_id] = set()
            intermediate_gene_counts[int_id].add(gene_id)

    # Build per-intermediate stats
    intermediate_stats = []
    for int_id, genes_using in intermediate_gene_counts.items():
        intermediate_name = None
        if node_name_maps and ":" in int_id:
            node_type = int_id.split(":")[0]
            node_identifier = int_id.split(":", 1)[1]
            if node_type in node_name_maps:
                intermediate_name = node_name_maps[node_type].get(node_identifier)

        intermediate_stats.append({
            "intermediate_id": int_id,
            "intermediate_name": intermediate_name,
            "n_genes_using": len(genes_using),
            "pct_genes_using": len(genes_using) / n_genes * 100,
        })

    intermediate_stats.sort(key=lambda x: x["n_genes_using"], reverse=True)

    n_total = len(intermediate_gene_counts)
    n_shared_quarter = sum(1 for genes in intermediate_gene_counts.values() if len(genes) > n_genes / 4)
    n_shared_majority = sum(1 for genes in intermediate_gene_counts.values() if len(genes) > n_genes / 2)

    top1_coverage = intermediate_stats[0]["pct_genes_using"] if intermediate_stats else 0.0
    if len(intermediate_stats) >= 5:
        top5_genes = set()
        for stat in intermediate_stats[:5]:
            top5_genes.update(intermediate_gene_counts[stat["intermediate_id"]])
        top5_coverage = len(top5_genes) / n_genes * 100
    else:
        top5_coverage = 0.0

    coverage_stats = {
        "top1_intermediate_coverage": top1_coverage,
        "top5_intermediate_coverage": top5_coverage,
        "pct_intermediates_shared_quarter": n_shared_quarter / n_total * 100 if n_total > 0 else 0.0,
        "pct_intermediates_shared_majority": n_shared_majority / n_total * 100 if n_total > 0 else 0.0,
    }

    return coverage_stats, intermediate_stats[:20]


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
        default=0.2,
        help="Minimum effect size (Cohen's d) for metapath selection",
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
    node_name_maps = _load_node_names(REPO_ROOT)
    print(f"Loaded node name maps for: {list(node_name_maps.keys())}")

    # Setup edge loader
    edges_dir = REPO_ROOT / "data" / "edges"
    edge_loader = EdgeLoader(edges_dir)

    # Load DWPC values (placeholder - would need actual DWPC data)
    dwpc_lookup: dict[tuple, float] = {}
    dwpc_thresholds: dict[tuple, float] = {}

    if args.dwpc_percentile > 0 and dwpc_lookup:
        dwpc_thresholds = _compute_dwpc_thresholds(dwpc_lookup, args.dwpc_percentile)
        print(f"Computed DWPC thresholds for {len(dwpc_thresholds)} pairs")

    # Initialize node maps (will be populated on first iteration)
    maps = None

    # Process each B value
    for b_value in b_values:
        print(f"\n{'='*60}")
        print(f"Processing B = {b_value}")
        print("=" * 60)

        # Output directory for this B
        if len(b_values) > 1:
            b_out_dir = out_dir / f"b{b_value}"
        else:
            b_out_dir = out_dir
        b_out_dir.mkdir(parents=True, exist_ok=True)

        # Load results at this B
        runs_path = year_output_dir / "year_rank_stability_experiment" / "all_runs_long.csv"
        if not runs_path.exists():
            print(f"Warning: {runs_path} not found, skipping B={b_value}")
            continue

        try:
            results_df = _load_runs_at_b(runs_path, b_value)
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

        for go_id in selected_mp["go_id"].unique():
            go_mps = selected_mp[selected_mp["go_id"] == go_id]

            # Get genes for this GO term
            go_gene_df = go_genes_df[go_genes_df["go_id"] == go_id]
            all_genes = go_gene_df["gene_identifier"].astype(int).tolist()

            # Split into 2016 (baseline) and 2024-added
            genes_2016 = [g for g in all_genes if (go_id, g) not in added_pairs]
            genes_2024 = [g for g in all_genes if (go_id, g) in added_pairs]

            print(f"\n{go_id}: {len(genes_2016)} genes (2016), {len(genes_2024)} added (2024), {len(go_mps)} metapaths")

            for _, mp_row in go_mps.iterrows():
                metapath = mp_row["metapath"]
                mp_rank = mp_row["metapath_rank"]
                effect_size = mp_row["effect_size_d"]

                # Enumerate intermediates for 2016 genes
                ints_2016 = _enumerate_gene_intermediates(
                    genes_2016, go_id, metapath, edge_loader, maps,
                    path_top_k=args.path_top_k,
                    dwpc_lookup=dwpc_lookup if dwpc_lookup else None,
                    dwpc_thresholds=dwpc_thresholds if dwpc_thresholds else None,
                )

                # Enumerate intermediates for 2024-added genes
                ints_2024 = _enumerate_gene_intermediates(
                    genes_2024, go_id, metapath, edge_loader, maps,
                    path_top_k=args.path_top_k,
                    dwpc_lookup=dwpc_lookup if dwpc_lookup else None,
                    dwpc_thresholds=dwpc_thresholds if dwpc_thresholds else None,
                )

                # Compute sharing stats
                stats = _compute_sharing_stats(
                    ints_2016, ints_2024, len(genes_2016), len(genes_2024)
                )

                # Compute coverage for combined gene set
                combined_ints = {**ints_2016, **ints_2024}
                coverage_stats, int_stats = _compute_intermediate_coverage(
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
