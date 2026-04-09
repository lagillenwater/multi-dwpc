#!/usr/bin/env python3
"""Compute intermediate sharing statistics for LV gene sets.

For each (LV, target_set) with supported metapaths:
1. Select metapaths using effective number at b=10
2. Enumerate path instances for top genes
3. Compute % of genes sharing intermediates

Outputs summary statistics for abstract-level reporting.
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


def _add_consensus_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add consensus_score column, consistent with year_go_term_support.py.

    consensus_score = 0.5 * (1/rank_perm + 1/rank_rand)
    where ranks are based on diff_perm and diff_rand within each group.
    """
    out = df.copy()
    group_cols = ["lv_id", "target_set_id"]

    out["rank_perm"] = (
        out.groupby(group_cols)["diff_perm"]
        .rank(method="average", ascending=False)
        .astype(float)
    )
    out["rank_rand"] = (
        out.groupby(group_cols)["diff_rand"]
        .rank(method="average", ascending=False)
        .astype(float)
    )
    out["consensus_rank"] = 0.5 * (out["rank_perm"] + out["rank_rand"])
    out["consensus_score"] = 0.5 * (
        (1.0 / out["rank_perm"].replace(0, np.nan))
        + (1.0 / out["rank_rand"].replace(0, np.nan))
    )
    return out


def _select_metapaths_by_effective_n(results_df: pd.DataFrame) -> pd.DataFrame:
    """Select metapaths by effective number of consensus scores.

    Consistent with year analysis (year_go_term_support.py):
    - Uses consensus_score (based on ranks of diff_perm and diff_rand)
    - Selects top ceil(effective_n) metapaths
    - If effective_n is 0 (no positive scores), no metapaths are selected
    """
    if results_df.empty:
        return pd.DataFrame()

    # Add consensus score columns
    results_df = _add_consensus_score(results_df)

    # For each (lv_id, target_set_id), select top effective_n metapaths
    selected_rows = []
    for (lv_id, ts_id), group in results_df.groupby(["lv_id", "target_set_id"]):
        if group.empty:
            continue
        # Sort by consensus_score descending
        group = group.sort_values("consensus_score", ascending=False).reset_index(drop=True)
        scores = group["consensus_score"].fillna(0.0).clip(lower=0.0).to_numpy()
        eff_n = _effective_number(scores)
        k = int(np.ceil(eff_n))
        if k == 0:
            continue
        top_k = group.head(k).copy()
        top_k["effective_n_metapaths"] = eff_n
        top_k["metapath_rank"] = range(1, len(top_k) + 1)
        selected_rows.append(top_k)

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
        return {
            "n_genes_with_paths": 0,
            "n_genes_sharing": 0,
            "pct_genes_sharing": None,
            "n_unique_intermediates": 0,
        }

    # Count genes that share at least one intermediate with another gene
    n_sharing = 0
    for gene_id, ints in gene_intermediates.items():
        other_ints = set()
        for other_gene, other_gene_ints in gene_intermediates.items():
            if other_gene != gene_id:
                other_ints.update(other_gene_ints)
        if ints & other_ints:
            n_sharing += 1

    all_intermediates = set()
    for ints in gene_intermediates.values():
        all_intermediates.update(ints)

    return {
        "n_genes_with_paths": n_genes,
        "n_genes_sharing": n_sharing,
        "pct_genes_sharing": n_sharing / n_genes * 100 if n_genes > 0 else None,
        "n_unique_intermediates": len(all_intermediates),
    }


def _load_runs_at_b(runs_path: Path, b: int) -> pd.DataFrame:
    """Load all_runs_long.csv and compute diff_perm/diff_rand at specified b.

    Averages diff across seeds for each control type, then pivots to get
    diff_perm and diff_rand columns.
    """
    runs_df = pd.read_csv(runs_path)

    # Filter to specified b
    runs_df = runs_df[runs_df["b"] == b].copy()
    if runs_df.empty:
        raise ValueError(f"No rows found for b={b} in {runs_path}")

    # Average diff across seeds for each (control, lv_id, target_set_id, metapath)
    group_cols = ["control", "lv_id", "target_set_id", "target_set_label", "node_type", "metapath"]
    mean_diff = runs_df.groupby(group_cols, as_index=False)["diff"].mean()

    # Pivot to get diff_perm and diff_rand columns
    pivot_cols = ["lv_id", "target_set_id", "target_set_label", "node_type", "metapath"]
    perm_df = mean_diff[mean_diff["control"] == "permuted"][pivot_cols + ["diff"]].copy()
    perm_df = perm_df.rename(columns={"diff": "diff_perm"})

    rand_df = mean_diff[mean_diff["control"] == "random"][pivot_cols + ["diff"]].copy()
    rand_df = rand_df.rename(columns={"diff": "diff_rand"})

    # Merge permuted and random
    result = perm_df.merge(rand_df, on=pivot_cols, how="outer")
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
    parser.add_argument("--output-dir", default="output/lv_intermediate_sharing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and concatenate data from all LV output directories
    results_frames = []
    top_genes_frames = []
    target_sets_frames = []

    for lv_output_dir_str in args.lv_output_dirs:
        lv_output_dir = Path(lv_output_dir_str)
        print(f"\nLoading from: {lv_output_dir}")

        runs_path = lv_output_dir / "lv_rank_stability_experiment" / "all_runs_long.csv"
        top_genes_path = lv_output_dir / "lv_top_genes.csv"
        target_sets_path = lv_output_dir / "target_sets.csv"

        if not runs_path.exists():
            print(f"  Warning: {runs_path} not found, skipping")
            continue

        results_frames.append(_load_runs_at_b(runs_path, args.b))
        top_genes_frames.append(pd.read_csv(top_genes_path))
        target_sets_frames.append(pd.read_csv(target_sets_path))

    if not results_frames:
        print("No valid LV output directories found.")
        return

    results_df = pd.concat(results_frames, ignore_index=True).drop_duplicates()
    top_genes = pd.concat(top_genes_frames, ignore_index=True).drop_duplicates()
    target_sets = pd.concat(target_sets_frames, ignore_index=True).drop_duplicates()

    print(f"\nLoaded {len(results_df)} metapath results at b={args.b}")
    print(f"Loaded {len(top_genes)} top genes")
    print(f"Loaded {len(target_sets)} target set entries")

    # Select metapaths by effective number (no FDR filtering, consistent with year analysis)
    selected_mp = _select_metapaths_by_effective_n(results_df)

    if selected_mp.empty:
        print("No supported metapaths found.")
        return

    print(f"\nSelected {len(selected_mp)} metapaths across {selected_mp['lv_id'].nunique()} LVs")

    # Save selected metapaths
    selected_mp.to_csv(out_dir / "selected_metapaths_effective_n.csv", index=False)
    print(f"Saved: {out_dir / 'selected_metapaths_effective_n.csv'}")

    # Summarize selection
    selection_summary = selected_mp.groupby(["lv_id", "target_set_id"]).agg(
        n_metapaths_selected=("metapath", "count"),
        effective_n_metapaths=("effective_n_metapaths", "first"),
        target_set_label=("target_set_label", "first"),
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

    # Process each (LV, target_set, metapath)
    sharing_rows = []

    for (lv_id, ts_id), group in selected_mp.groupby(["lv_id", "target_set_id"]):
        target_label = group["target_set_label"].iloc[0]
        node_type = group["node_type"].iloc[0]

        # Get genes for this LV (convert to int for node map lookup)
        lv_genes_raw = top_genes[top_genes["lv_id"] == lv_id]["gene_identifier"].tolist()
        lv_genes = [int(g) for g in lv_genes_raw]
        print(f"\n{lv_id} / {target_label}: {len(lv_genes)} genes, {len(group)} metapaths")

        # Get target positions from target_sets
        ts_targets = target_sets[target_sets["target_set_id"] == ts_id]
        target_positions = ts_targets["target_position"].unique()
        print(f"  Target positions: {target_positions[:5]}... (total: {len(target_positions)})")

        for _, mp_row in group.iterrows():
            metapath = mp_row["metapath"]
            mp_rank = mp_row["metapath_rank"]
            # Use mean of diff_perm and diff_rand as effect size
            effect_size = 0.5 * (mp_row.get("diff_perm", 0) + mp_row.get("diff_rand", 0))

            # Aggregate intermediates across all targets
            all_gene_intermediates: dict[int, set[str]] = {}

            is_first_mp = (mp_rank == 1)
            for target_pos in target_positions:
                gene_ints = _enumerate_gene_intermediates(
                    lv_genes, int(target_pos), metapath, edge_loader, maps,
                    path_top_k=args.path_top_k, degree_d=args.degree_d,
                    debug=is_first_mp,
                )
                if gene_ints:
                    print(f"    Found paths for {len(gene_ints)} genes at target_pos={target_pos}")
                for gene_id, ints in gene_ints.items():
                    if gene_id not in all_gene_intermediates:
                        all_gene_intermediates[gene_id] = set()
                    all_gene_intermediates[gene_id].update(ints)

            stats = _compute_sharing_stats(all_gene_intermediates)
            stats.update({
                "lv_id": lv_id,
                "target_set_id": ts_id,
                "target_set_label": target_label,
                "node_type": node_type,
                "metapath": metapath,
                "metapath_rank": mp_rank,
                "effect_size_d": effect_size,
                "n_genes_total": len(lv_genes),
                "n_targets": len(target_positions),
            })
            sharing_rows.append(stats)

            print(f"  {metapath} (rank {mp_rank}): {stats['n_genes_with_paths']}/{len(lv_genes)} genes, "
                  f"{stats['pct_genes_sharing']:.1f}% sharing" if stats['pct_genes_sharing'] else "  no paths")

    # Save detailed results
    sharing_df = pd.DataFrame(sharing_rows)
    col_order = [
        "lv_id", "target_set_id", "target_set_label", "node_type",
        "metapath", "metapath_rank", "effect_size_d",
        "n_genes_total", "n_targets",
        "n_genes_with_paths", "n_genes_sharing", "pct_genes_sharing",
        "n_unique_intermediates",
    ]
    sharing_df = sharing_df[[c for c in col_order if c in sharing_df.columns]]
    sharing_df.to_csv(out_dir / "intermediate_sharing_by_metapath.csv", index=False)
    print(f"\nSaved: {out_dir / 'intermediate_sharing_by_metapath.csv'}")

    # Aggregate summary per (LV, target_set)
    summary = sharing_df.groupby(["lv_id", "target_set_id", "target_set_label", "node_type"]).agg(
        n_metapaths=("metapath", "count"),
        n_genes_total=("n_genes_total", "first"),
        median_pct_sharing=("pct_genes_sharing", "median"),
        mean_pct_sharing=("pct_genes_sharing", "mean"),
        max_pct_sharing=("pct_genes_sharing", "max"),
        median_n_intermediates=("n_unique_intermediates", "median"),
    ).reset_index()
    summary.to_csv(out_dir / "intermediate_sharing_summary.csv", index=False)
    print(f"Saved: {out_dir / 'intermediate_sharing_summary.csv'}")

    # Print summary for abstract
    print("\n" + "=" * 60)
    print("=== Summary for abstract ===")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(f"\n{row['lv_id']} / {row['target_set_label']}:")
        print(f"  {row['n_metapaths']} metapaths selected by effective number")
        print(f"  {row['n_genes_total']} genes in set")
        print(f"  Median {row['median_pct_sharing']:.1f}% of genes share intermediates")
        print(f"  Max {row['max_pct_sharing']:.1f}% sharing for best metapath")


if __name__ == "__main__":
    main()
