#!/usr/bin/env python3
"""Compute intermediate sharing statistics for LV gene sets.

For each LV with supported metapaths:
1. Select metapaths using effective number at b=10
2. Enumerate path instances for top genes
3. Compute % of genes sharing intermediates

Outputs summary statistics for abstract-level reporting.
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
    """Load node ID to name mappings for all node types.

    Returns dict mapping node_type_abbrev -> {identifier -> name}
    """
    nodes_dir = repo_root / "data" / "nodes"
    name_maps: dict[str, dict[str, str]] = {}

    for abbrev, full_name in NODE_TYPE_NAMES.items():
        node_file = nodes_dir / f"{full_name}.tsv"
        if not node_file.exists():
            continue

        df = pd.read_csv(node_file, sep="\t")
        # Map identifier to name
        name_maps[abbrev] = dict(zip(df["identifier"].astype(str), df["name"]))

    return name_maps


def _select_metapaths_by_effect_size(
    results_df: pd.DataFrame,
    d_threshold: float = 0.2,
) -> pd.DataFrame:
    """Select metapaths by effect size (Cohen's d) from permutation null.

    Per web_tool_discussion.md (section 1.1c, 1.2):
    - Use permutation null only (not random, not consensus)
    - Rank metapaths by effect size d
    - Select metapaths with d > 0.2 (small+ effect per Cohen's benchmarks)

    Args:
        results_df: DataFrame with columns including 'effect_size_d' and 'lv_id'
        d_threshold: Minimum effect size to include (default 0.2 = small effect)

    Returns:
        DataFrame of selected metapaths with metapath_rank column
    """
    if results_df.empty:
        return pd.DataFrame()

    selected_rows = []
    for lv_id, group in results_df.groupby("lv_id"):
        if group.empty:
            continue

        # Filter to metapaths with d > threshold
        above_threshold = group[group["effect_size_d"] > d_threshold].copy()

        if above_threshold.empty:
            continue

        # Sort by effect size descending and assign ranks
        above_threshold = above_threshold.sort_values(
            "effect_size_d", ascending=False
        ).reset_index(drop=True)
        above_threshold["metapath_rank"] = range(1, len(above_threshold) + 1)
        selected_rows.append(above_threshold)

    if not selected_rows:
        return pd.DataFrame()

    return pd.concat(selected_rows, ignore_index=True)


def _enumerate_gene_intermediates(
    genes: list[int],
    target_pos: int,
    metapath: str,
    edge_loader: EdgeLoader,
    maps,
    *,
    path_top_k: int = 100,
    degree_d: float = 0.5,
    debug: bool = False,
    dwpc_lookup: dict[tuple, float] | None = None,
    dwpc_threshold: float = 0.0,
    lv_id: str | None = None,
) -> dict[int, set[str]]:
    """Enumerate paths for genes and return {gene_id: set of intermediate_ids}.

    The metapath is Gene -> ... -> Target, but enumerate_paths expects
    Target -> ... -> Gene, so we reverse the metapath for enumeration
    and then reverse the resulting paths.
    """
    # Metapath goes Gene -> ... -> Target, reverse it for enumeration
    reversed_mp = reverse_metapath_abbrev(metapath)
    nodes, edges = parse_metapath(reversed_mp)

    if debug:
        print(f"      Metapath {metapath} -> reversed {reversed_mp}: nodes={nodes}, edges={edges}")

    gene_intermediates: dict[int, set[str]] = {}
    genes_found = 0
    genes_with_paths = 0

    gene_id_map = maps.id_to_pos.get("G", {})
    if debug:
        print(f"      Gene map has {len(gene_id_map)} entries, sample keys: {list(gene_id_map.keys())[:3]} (types: {[type(k) for k in list(gene_id_map.keys())[:3]]})")
        print(f"      Input genes sample: {genes[:3]} (types: {[type(g) for g in genes[:3]]})")

    for gene_id in genes:
        # Filter by DWPC threshold if lookup available
        if dwpc_lookup is not None and lv_id is not None:
            dwpc = dwpc_lookup.get((lv_id, metapath, gene_id), 0.0)
            if dwpc <= dwpc_threshold:
                continue

        # Try both int and string lookups since Gene.tsv identifiers may be int
        gene_pos = gene_id_map.get(gene_id) or gene_id_map.get(str(gene_id)) or gene_id_map.get(int(gene_id) if isinstance(gene_id, str) else gene_id)
        if gene_pos is None:
            continue
        genes_found += 1

        try:
            # enumerate_paths goes from target_pos -> gene_pos
            paths = enumerate_paths(
                target_pos, gene_pos, nodes, edges, edge_loader,
                top_k=path_top_k, degree_d=degree_d,
            )
        except Exception as e:
            if debug:
                print(f"      Exception for gene {gene_id}: {e}")
            continue

        # Select top paths by effective number
        if not paths:
            continue
        genes_with_paths += 1

        selected = select_paths(
            paths,
            selection_method="effective_number",
            top_paths=path_top_k,
            path_cumulative_frac=None,
            path_min_count=1,
            path_max_count=None,
        )

        intermediates = set()
        for score, pos_path in selected:
            # pos_path is [target_pos, intermediate1, intermediate2, ..., gene_pos]
            # Intermediate nodes are positions 1 to -2
            for i, (node_type, pos) in enumerate(zip(nodes, pos_path)):
                if 0 < i < len(nodes) - 1:  # intermediate positions
                    node_id = maps.pos_to_id.get(node_type, {}).get(int(pos))
                    if node_id is not None:
                        intermediates.add(f"{node_type}:{node_id}")

        if intermediates:
            gene_intermediates[gene_id] = intermediates

    if debug:
        print(f"      genes_found={genes_found}, genes_with_paths={genes_with_paths}, intermediates={len(gene_intermediates)}")

    return gene_intermediates


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


def _compute_intermediate_coverage(
    gene_intermediates: dict[int, set[str]],
    node_name_maps: dict[str, dict[str, str]] | None = None,
) -> tuple[dict, list[dict]]:
    """Compute intermediate coverage statistics.

    Args:
        gene_intermediates: dict mapping gene_id -> set of intermediate IDs (format: "TYPE:ID")
        node_name_maps: dict mapping node_type_abbrev -> {identifier -> name}

    Returns:
        coverage_stats: dict with summary coverage metrics
        top_intermediates: list of dicts with per-intermediate statistics
    """
    n_genes = len(gene_intermediates)
    if n_genes == 0:
        # Return 0 (not None) so metapaths without paths are included in median calculations
        return {
            "n_shared_intermediates_2plus": 0,
            "n_shared_intermediates_quarter": 0,
            "n_shared_intermediates_majority": 0,
            "n_shared_intermediates_all": 0,
            "pct_intermediates_shared_2plus": 0.0,
            "pct_intermediates_shared_quarter": 0.0,
            "pct_intermediates_shared_majority": 0.0,
            "pct_intermediates_shared_all": 0.0,
            "n_intermediates_cover_50pct": None,
            "n_intermediates_cover_80pct": None,
            "top1_intermediate_coverage": 0.0,
            "top5_intermediate_coverage": 0.0,
        }, []

    # Count how many genes use each intermediate
    intermediate_gene_counts: dict[str, set[int]] = {}
    for gene_id, ints in gene_intermediates.items():
        for int_id in ints:
            if int_id not in intermediate_gene_counts:
                intermediate_gene_counts[int_id] = set()
            intermediate_gene_counts[int_id].add(gene_id)

    # Build per-intermediate stats
    intermediate_stats = []
    for int_id, genes_using in intermediate_gene_counts.items():
        # Parse intermediate ID to get name
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
            "genes_using": sorted(genes_using),
        })

    # Sort by number of genes using (descending)
    intermediate_stats.sort(key=lambda x: x["n_genes_using"], reverse=True)

    # Compute coverage metrics
    n_total_intermediates = len(intermediate_gene_counts)

    # Count intermediates at different sharing thresholds
    n_shared_2plus = sum(1 for genes in intermediate_gene_counts.values() if len(genes) >= 2)
    n_shared_quarter = sum(1 for genes in intermediate_gene_counts.values() if len(genes) > n_genes / 4)
    n_shared_majority = sum(1 for genes in intermediate_gene_counts.values() if len(genes) > n_genes / 2)
    n_shared_all = sum(1 for genes in intermediate_gene_counts.values() if len(genes) == n_genes)

    # Greedy set cover: how many intermediates needed to cover X% of genes?
    genes_covered: set[int] = set()
    all_genes = set(gene_intermediates.keys())
    n_for_50pct = None
    n_for_80pct = None

    for i, stat in enumerate(intermediate_stats, 1):
        genes_covered.update(stat["genes_using"])
        coverage_pct = len(genes_covered) / n_genes * 100
        if n_for_50pct is None and coverage_pct >= 50:
            n_for_50pct = i
        if n_for_80pct is None and coverage_pct >= 80:
            n_for_80pct = i
        if coverage_pct >= 100:
            break

    # Top-k intermediate coverage
    top1_coverage = intermediate_stats[0]["pct_genes_using"] if intermediate_stats else None
    if len(intermediate_stats) >= 5:
        top5_genes = set()
        for stat in intermediate_stats[:5]:
            top5_genes.update(stat["genes_using"])
        top5_coverage = len(top5_genes) / n_genes * 100
    else:
        top5_coverage = None

    coverage_stats = {
        "n_shared_intermediates_2plus": n_shared_2plus,
        "n_shared_intermediates_quarter": n_shared_quarter,
        "n_shared_intermediates_majority": n_shared_majority,
        "n_shared_intermediates_all": n_shared_all,
        "pct_intermediates_shared_2plus": n_shared_2plus / n_total_intermediates * 100 if n_total_intermediates > 0 else None,
        "pct_intermediates_shared_quarter": n_shared_quarter / n_total_intermediates * 100 if n_total_intermediates > 0 else None,
        "pct_intermediates_shared_majority": n_shared_majority / n_total_intermediates * 100 if n_total_intermediates > 0 else None,
        "pct_intermediates_shared_all": n_shared_all / n_total_intermediates * 100 if n_total_intermediates > 0 else None,
        "n_intermediates_cover_50pct": n_for_50pct,
        "n_intermediates_cover_80pct": n_for_80pct,
        "top1_intermediate_coverage": top1_coverage,
        "top5_intermediate_coverage": top5_coverage,
    }

    return coverage_stats, intermediate_stats


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


def _load_runs_at_b(runs_path: Path, b: int) -> pd.DataFrame:
    """Load all_runs_long.csv and get effect sizes from permutation null.

    Per web_tool_discussion.md (section 1.2):
    - Use permutation null only (drop random null)
    - Effect size d is already computed in the data

    Args:
        runs_path: Path to all_runs_long.csv
        b: Number of replicates to use

    Returns:
        DataFrame with effect_size_d for permutation null
    """
    runs_df = pd.read_csv(runs_path)

    # Filter to specified b and permutation null only
    runs_df = runs_df[(runs_df["b"] == b) & (runs_df["control"] == "permuted")].copy()
    if runs_df.empty:
        raise ValueError(f"No permuted rows found for b={b} in {runs_path}")

    # Average effect size d across seeds for each (lv_id, metapath)
    group_cols = ["lv_id", "target_id", "target_name", "node_type", "metapath"]

    # Check if 'd' column exists (effect size), otherwise compute from diff
    if "d" in runs_df.columns:
        result = runs_df.groupby(group_cols, as_index=False).agg(
            effect_size_d=("d", "mean"),
            diff_perm=("diff", "mean"),
        )
    else:
        # Fallback: just use diff (no std available)
        result = runs_df.groupby(group_cols, as_index=False).agg(
            diff_perm=("diff", "mean"),
        )
        result["effect_size_d"] = result["diff_perm"]  # Use diff as proxy

    return result


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
        default=10,
        help="B value for metapath selection (default: 10).",
    )
    parser.add_argument("--path-top-k", type=int, default=100)
    parser.add_argument("--degree-d", type=float, default=0.5)
    parser.add_argument(
        "--effect-size-threshold", type=float, default=0.2,
        help="Minimum effect size (Cohen's d) for metapath selection. Default 0.2 (small effect).",
    )
    parser.add_argument("--output-dir", default="output/lv_intermediate_sharing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and concatenate data from all LV output directories
    results_frames = []
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

        results_frames.append(_load_runs_at_b(runs_path, args.b))
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

    if not results_frames:
        print("No valid LV output directories found.")
        return

    results_df = pd.concat(results_frames, ignore_index=True).drop_duplicates()
    top_genes = pd.concat(top_genes_frames, ignore_index=True).drop_duplicates()
    lv_targets = pd.concat(lv_targets_frames, ignore_index=True).drop_duplicates()

    if dwpc_lookup:
        print(f"\nTotal DWPC lookup entries: {len(dwpc_lookup)}")
    else:
        print("\nWarning: No DWPC data found, DWPC filtering disabled")

    # DWPC threshold: use dwpc > 0 (any non-zero connectivity)
    # Per web_tool_discussion.md section 1.1d: metapath-level d > 0.2 filtering
    # already selects meaningful metapaths; for genes we just need non-zero DWPC
    dwpc_threshold = 0.0
    print(f"DWPC threshold: {dwpc_threshold} (genes with any connectivity)")

    print(f"\nLoaded {len(results_df)} metapath results at b={args.b}")
    print(f"Loaded {len(top_genes)} top genes")
    print(f"Loaded {len(lv_targets)} LV target entries")

    # Load node name mappings for human-readable intermediate names
    node_name_maps = _load_node_names(REPO_ROOT)
    print(f"Loaded node name maps for: {list(node_name_maps.keys())}")

    # Select metapaths by effect size (per web_tool_discussion.md section 1.1c)
    # Uses permutation null only, ranks by Cohen's d
    # Default threshold 0.2 = small effect per Cohen's benchmarks
    selected_mp = _select_metapaths_by_effect_size(
        results_df, d_threshold=args.effect_size_threshold
    )

    if selected_mp.empty:
        print("No metapaths found with effect size d > 0.2")
        return

    print(f"\nSelected {len(selected_mp)} metapaths across {selected_mp['lv_id'].nunique()} LVs")

    # Save selected metapaths
    selected_mp.to_csv(out_dir / "selected_metapaths_effective_n.csv", index=False)
    print(f"Saved: {out_dir / 'selected_metapaths_effective_n.csv'}")

    # Summarize selection
    selection_summary = selected_mp.groupby("lv_id").agg(
        n_metapaths_selected=("metapath", "count"),
        effective_n_metapaths=("effective_n_metapaths", "first"),
        target_name=("target_name", "first"),
        node_type=("node_type", "first"),
    ).reset_index()
    print(f"\n=== Metapath selection summary (b={args.b}) ===")
    print(selection_summary.to_string(index=False))

    # Setup for path enumeration
    edges_dir = REPO_ROOT / "data" / "edges"
    edge_loader = EdgeLoader(edges_dir)

    # Get unique node types from selected metapaths
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

            gene_ints = _enumerate_gene_intermediates(
                lv_genes, int(target_position), metapath, edge_loader, maps,
                path_top_k=args.path_top_k, degree_d=args.degree_d,
                debug=is_first_mp,
                dwpc_lookup=dwpc_lookup if dwpc_lookup else None,
                dwpc_threshold=dwpc_threshold,
                lv_id=lv_id,
            )
            if gene_ints:
                print(f"    Found paths for {len(gene_ints)} genes")

            stats = _compute_sharing_stats(gene_ints)
            coverage_stats, intermediate_stats = _compute_intermediate_coverage(
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

    # Save detailed results
    sharing_df = pd.DataFrame(sharing_rows)
    col_order = [
        "lv_id", "target_id", "target_name", "node_type",
        "metapath", "metapath_rank", "effect_size_d",
        "n_genes_total",
        "n_genes_with_paths", "n_genes_sharing", "pct_genes_sharing",
        "median_jaccard_to_group", "mean_jaccard_to_group",
        "n_unique_intermediates",
        "n_shared_intermediates_2plus", "n_shared_intermediates_majority", "n_shared_intermediates_all",
        "pct_intermediates_shared_2plus", "pct_intermediates_shared_majority", "pct_intermediates_shared_all",
        "n_intermediates_cover_50pct", "n_intermediates_cover_80pct",
        "top1_intermediate_coverage", "top5_intermediate_coverage",
    ]
    sharing_df = sharing_df[[c for c in col_order if c in sharing_df.columns]]
    sharing_df.to_csv(out_dir / "intermediate_sharing_by_metapath.csv", index=False)
    print(f"\nSaved: {out_dir / 'intermediate_sharing_by_metapath.csv'}")

    # Save top intermediates
    if top_intermediates_rows:
        top_int_df = pd.DataFrame(top_intermediates_rows)
        top_int_df.to_csv(out_dir / "top_intermediates_by_metapath.csv", index=False)
        print(f"Saved: {out_dir / 'top_intermediates_by_metapath.csv'}")

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
    summary.to_csv(out_dir / "intermediate_sharing_summary.csv", index=False)
    print(f"Saved: {out_dir / 'intermediate_sharing_summary.csv'}")

    # Print summary for abstract
    print("\n" + "=" * 60)
    print("=== Summary for abstract ===")
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


if __name__ == "__main__":
    main()
