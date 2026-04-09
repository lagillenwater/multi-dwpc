#!/usr/bin/env python3
"""Compute intermediate sharing between 2016 and 2024-added genes per GO term.

For each (GO term, metapath) selected in BOTH 2016 and 2024:
1. Identify genes annotated in 2016 vs genes added by 2024
2. Enumerate path instances for both gene sets
3. Compute:
   - % of 2016 genes sharing intermediates with other 2016 genes
   - % of 2024-added genes sharing intermediates with 2016 genes

Outputs summary statistics only (no path instances saved) to conserve disk.

Parallelizable: use --go-id to run a single GO term (for HPC array jobs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _effective_number(scores: np.ndarray) -> float:
    """Compute effective number from score distribution."""
    vals = scores[np.isfinite(scores)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    weights = vals / vals.sum()
    entropy = float(-(weights * np.log(weights)).sum())
    return float(np.exp(entropy))


def _load_consensus_metapaths(
    support_path: Path,
    selection_col: str,
    rank_col: str = "consensus_score",
    max_rank: int = 5,
) -> pd.DataFrame:
    """Load metapaths selected in BOTH 2016 and 2024."""
    support = pd.read_csv(support_path)
    support[selection_col] = (
        support[selection_col]
        .astype(str).str.strip().str.lower()
        .isin({"1", "true", "t", "yes"})
    )

    # Get selected metapaths for each year
    selected_2016 = support[
        (support["year"] == 2016) & support[selection_col]
    ][["go_id", "metapath"]].drop_duplicates()
    selected_2016["in_2016"] = True

    selected_2024 = support[
        (support["year"] == 2024) & support[selection_col]
    ][["go_id", "metapath"]].drop_duplicates()
    selected_2024["in_2024"] = True

    # Keep only those in both years
    consensus = selected_2016.merge(selected_2024, on=["go_id", "metapath"], how="inner")

    # Add ranking based on 2016 scores
    support_2016 = support[support["year"] == 2016].copy()
    ascending = rank_col in {"consensus_rank", "fdr_sum"}
    support_2016["metapath_rank"] = (
        support_2016.groupby("go_id")[rank_col]
        .rank(method="first", ascending=ascending)
        .astype(int)
    )

    consensus = consensus.merge(
        support_2016[["go_id", "metapath", "metapath_rank"]],
        on=["go_id", "metapath"],
        how="left",
    )
    consensus = consensus[consensus["metapath_rank"] <= max_rank]

    return consensus[["go_id", "metapath", "metapath_rank"]].drop_duplicates()


def _enumerate_gene_intermediates(
    genes: list[int],
    go_id: str,
    metapath_g: str,
    edge_loader: EdgeLoader,
    maps,
    dwpc_lookup: dict[tuple, float],
    *,
    path_top_k: int = 5000,
    path_min_count: int = 1,
    path_max_count: int | None = None,
    degree_d: float = 0.5,
) -> dict[int, set[str]]:
    """Enumerate paths for genes and return {gene_id: set of intermediate_ids}."""
    metapath_bp = reverse_metapath_abbrev(metapath_g)
    nodes, edges = parse_metapath(metapath_bp)

    bp_pos = maps.id_to_pos.get("BP", {}).get(go_id)
    if bp_pos is None:
        return {}

    gene_intermediates: dict[int, set[str]] = {}

    for gene_id in genes:
        gene_pos = maps.id_to_pos.get("G", {}).get(gene_id)
        if gene_pos is None:
            continue

        # Get DWPC for this pair to determine effective n paths
        dwpc = dwpc_lookup.get((go_id, metapath_g, gene_id), 0.0)
        if dwpc <= 0:
            continue

        try:
            candidate_paths = enumerate_paths(
                bp_pos, gene_pos, nodes, edges, edge_loader,
                top_k=path_top_k, degree_d=degree_d,
            )
            paths = select_paths(
                candidate_paths,
                selection_method="effective_number",
                top_paths=5,
                path_cumulative_frac=None,
                path_min_count=path_min_count,
                path_max_count=path_max_count,
            )
        except Exception:
            continue

        intermediates = set()
        for score, pos_path in paths:
            # Intermediate nodes are positions 1 to -2 (excluding first BP and last Gene)
            for i, (node_type, pos) in enumerate(zip(nodes, pos_path)):
                if 0 < i < len(nodes) - 1:  # intermediate positions
                    node_id = maps.pos_to_id[node_type].get(int(pos))
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

    # 2016-2016 sharing: % of 2016 genes that share an intermediate with another 2016 gene
    n_2016 = len(genes_2016_intermediates)
    n_2016_sharing = 0
    for gene_id, ints in genes_2016_intermediates.items():
        other_ints = set()
        for other_gene, other_gene_ints in genes_2016_intermediates.items():
            if other_gene != gene_id:
                other_ints.update(other_gene_ints)
        if ints & other_ints:
            n_2016_sharing += 1

    # 2024-2016 sharing: % of 2024-added genes that share an intermediate with any 2016 gene
    n_2024 = len(genes_2024_intermediates)
    n_2024_sharing_with_2016 = 0
    for gene_id, ints in genes_2024_intermediates.items():
        if ints & all_2016_intermediates:
            n_2024_sharing_with_2016 += 1

    # 2024-2024 sharing: % of 2024 genes sharing with another 2024 gene
    n_2024_sharing_with_2024 = 0
    for gene_id, ints in genes_2024_intermediates.items():
        other_ints = set()
        for other_gene, other_gene_ints in genes_2024_intermediates.items():
            if other_gene != gene_id:
                other_ints.update(other_gene_ints)
        if ints & other_ints:
            n_2024_sharing_with_2024 += 1

    return {
        "n_genes_2016": n_genes_2016_total,
        "n_genes_2016_with_paths": n_2016,
        "n_genes_2016_sharing_with_2016": n_2016_sharing,
        "pct_2016_sharing_with_2016": n_2016_sharing / n_2016 * 100 if n_2016 > 0 else None,
        "n_genes_2024_added": n_genes_2024_total,
        "n_genes_2024_with_paths": n_2024,
        "n_genes_2024_sharing_with_2016": n_2024_sharing_with_2016,
        "pct_2024_sharing_with_2016": n_2024_sharing_with_2016 / n_2024 * 100 if n_2024 > 0 else None,
        "n_genes_2024_sharing_with_2024": n_2024_sharing_with_2024,
        "pct_2024_sharing_with_2024": n_2024_sharing_with_2024 / n_2024 * 100 if n_2024 > 0 else None,
        "n_unique_intermediates_2016": len(all_2016_intermediates),
        "n_unique_intermediates_2024": len(set().union(*genes_2024_intermediates.values())) if genes_2024_intermediates else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct-results-dir",
        default="output/dwpc_direct/all_GO_positive_growth/results",
    )
    parser.add_argument(
        "--support-path",
        default="output/year_direct_go_term_support_b5.csv",
    )
    parser.add_argument(
        "--added-pairs-path",
        default="output/intermediate/upd_go_bp_2024_added.csv",
    )
    parser.add_argument("--selection-col", default="selected_by_effective_n_all")
    parser.add_argument("--max-metapath-rank", type=int, default=5)
    parser.add_argument("--path-enumeration-top-k", type=int, default=5000)
    parser.add_argument("--path-min-count", type=int, default=1)
    parser.add_argument("--path-max-count", type=int, default=None)
    parser.add_argument("--degree-d", type=float, default=0.5)
    parser.add_argument(
        "--go-id", default=None,
        help="Run a single GO term (for HPC array parallelization).",
    )
    parser.add_argument("--output-dir", default="output/year_intermediate_sharing")
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct_dir = Path(args.direct_results_dir)
    support_path = Path(args.support_path)
    added_path = Path(args.added_pairs_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_dir = REPO_ROOT / "data" / "edges"
    edge_loader = EdgeLoader(edges_dir)

    # Load added pairs (2024 additions)
    added_df = pd.read_csv(added_path)
    added_pairs = set(zip(added_df["go_id"], added_df["entrez_gene_id"]))
    print(f"Loaded {len(added_pairs):,} added GO-gene pairs")

    # Load consensus metapaths (selected in both 2016 and 2024)
    consensus_mp = _load_consensus_metapaths(
        support_path, args.selection_col, max_rank=args.max_metapath_rank,
    )
    print(f"Consensus metapaths: {len(consensus_mp):,} (GO, metapath) pairs")

    # Filter to single GO term if specified
    if args.go_id:
        consensus_mp = consensus_mp[consensus_mp["go_id"] == args.go_id].copy()
        if consensus_mp.empty:
            print(f"No consensus metapaths for GO term {args.go_id}")
            sys.exit(0)

    go_ids = consensus_mp["go_id"].unique()
    print(f"Processing {len(go_ids)} GO terms")

    # Load DWPC data for 2016 (to identify 2016 genes)
    pattern_2016 = "dwpc_*_2016_real.csv"
    matches_2016 = sorted(direct_dir.glob(pattern_2016))
    if not matches_2016:
        print(f"Error: no files matching {pattern_2016}", file=sys.stderr)
        sys.exit(1)

    chunks_2016 = []
    for path in matches_2016:
        for chunk in pd.read_csv(path, chunksize=args.chunksize):
            chunk = chunk[chunk["go_id"].isin(go_ids)]
            if not chunk.empty:
                chunks_2016.append(chunk)
    dwpc_2016 = pd.concat(chunks_2016, ignore_index=True)
    print(f"Loaded 2016 DWPC: {len(dwpc_2016):,} rows")

    # Load DWPC data for 2024 (to look up pair scores for path enumeration)
    pattern_2024 = "dwpc_*_2024_real.csv"
    matches_2024 = sorted(direct_dir.glob(pattern_2024))
    if not matches_2024:
        print(f"Error: no files matching {pattern_2024}", file=sys.stderr)
        sys.exit(1)

    chunks_2024 = []
    for path in matches_2024:
        for chunk in pd.read_csv(path, chunksize=args.chunksize):
            chunk = chunk[chunk["go_id"].isin(go_ids)]
            if not chunk.empty:
                chunks_2024.append(chunk)
    dwpc_2024 = pd.concat(chunks_2024, ignore_index=True)
    print(f"Loaded 2024 DWPC: {len(dwpc_2024):,} rows")

    # Create lookup for DWPC values from BOTH files
    # 2016 genes need 2016 DWPC; 2024-added genes need 2024 DWPC
    dwpc_lookup = {}
    for _, row in dwpc_2016.iterrows():
        dwpc_lookup[(row["go_id"], row["metapath"], int(row["entrez_gene_id"]))] = row["dwpc"]
    for _, row in dwpc_2024.iterrows():
        dwpc_lookup[(row["go_id"], row["metapath"], int(row["entrez_gene_id"]))] = row["dwpc"]
    print(f"DWPC lookup: {len(dwpc_lookup):,} entries")

    # Load node maps once
    node_types = ["BP", "G", "C", "PW", "MF", "CC", "A", "D"]  # common types
    maps = load_node_maps(REPO_ROOT, node_types)

    # Process each GO term
    summary_rows = []

    for go_id in go_ids:
        go_metapaths = consensus_mp[consensus_mp["go_id"] == go_id]["metapath"].tolist()

        # Get 2016 genes from 2016 DWPC data
        go_dwpc_2016 = dwpc_2016[dwpc_2016["go_id"] == go_id]
        genes_2016 = set(go_dwpc_2016["entrez_gene_id"].unique().astype(int))

        # Get 2024-added genes from added pairs file
        genes_2024_added = {g for (gid, g) in added_pairs if gid == go_id}

        if not genes_2016:
            print(f"  {go_id}: No 2016 genes, skipping")
            continue

        print(f"  {go_id}: {len(genes_2016)} genes (2016), {len(genes_2024_added)} added (2024), {len(go_metapaths)} metapaths")

        for metapath_g in go_metapaths:
            # Enumerate intermediates for 2016 genes
            genes_2016_intermediates = _enumerate_gene_intermediates(
                list(genes_2016), go_id, metapath_g, edge_loader, maps, dwpc_lookup,
                path_top_k=args.path_enumeration_top_k,
                path_min_count=args.path_min_count,
                path_max_count=args.path_max_count,
                degree_d=args.degree_d,
            )

            # Enumerate intermediates for 2024-added genes
            genes_2024_intermediates = _enumerate_gene_intermediates(
                list(genes_2024_added), go_id, metapath_g, edge_loader, maps, dwpc_lookup,
                path_top_k=args.path_enumeration_top_k,
                path_min_count=args.path_min_count,
                path_max_count=args.path_max_count,
                degree_d=args.degree_d,
            )

            if not genes_2016_intermediates and not genes_2024_intermediates:
                continue

            # Compute sharing statistics
            stats = _compute_sharing_stats(
                genes_2016_intermediates,
                genes_2024_intermediates,
                n_genes_2016_total=len(genes_2016),
                n_genes_2024_total=len(genes_2024_added),
            )
            stats["go_id"] = go_id
            stats["metapath"] = metapath_g
            summary_rows.append(stats)

    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        # Reorder columns
        col_order = [
            "go_id", "metapath",
            "n_genes_2016", "n_genes_2016_with_paths",
            "n_genes_2016_sharing_with_2016", "pct_2016_sharing_with_2016",
            "n_genes_2024_added", "n_genes_2024_with_paths",
            "n_genes_2024_sharing_with_2016", "pct_2024_sharing_with_2016",
            "n_genes_2024_sharing_with_2024", "pct_2024_sharing_with_2024",
            "n_unique_intermediates_2016", "n_unique_intermediates_2024",
        ]
        summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

        suffix = f"_{args.go_id}" if args.go_id else ""
        out_path = out_dir / f"intermediate_sharing_summary{suffix}.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path} ({len(summary_df):,} rows)")

        # Print global stats
        print(f"\n--- Summary statistics ---")
        print(f"(GO term, metapath) groups: {len(summary_df):,}")

        valid_2016 = summary_df["pct_2016_sharing_with_2016"].dropna()
        if len(valid_2016) > 0:
            print(f"2016-2016 sharing: median {valid_2016.median():.1f}%, mean {valid_2016.mean():.1f}%")

        valid_2024 = summary_df["pct_2024_sharing_with_2016"].dropna()
        if len(valid_2024) > 0:
            print(f"2024-2016 sharing: median {valid_2024.median():.1f}%, mean {valid_2024.mean():.1f}%")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
