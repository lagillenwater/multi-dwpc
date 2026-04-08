#!/usr/bin/env python3
"""Enumerate path instances for effective-number-selected pairs and analyze
shared intermediate nodes per (GO term, metapath).

Uses 2016-selected metapaths applied to 2024 DWPC data. For each (GO term,
metapath), selects effective-n top pairs by DWPC, enumerates effective-n
path instances per pair, then counts how many genes share intermediate nodes.

Parallelizable: use --go-id to run a single GO term (for HPC array jobs).

Outputs
-------
- path_instances.csv             : all enumerated path instances
- intermediate_sharing.csv       : per (GO term, metapath, intermediate node):
                                   how many genes share that intermediate
- intermediate_summary.csv       : per (GO term, metapath): summary stats
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
    vals = scores[np.isfinite(scores)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    weights = vals / vals.sum()
    entropy = float(-(weights * np.log(weights)).sum())
    return float(np.exp(entropy))


def _load_selected_metapaths(
    support_path: Path,
    selection_col: str,
    year: int,
    rank_col: str = "consensus_score",
    max_rank: int = 5,
) -> pd.DataFrame:
    support = pd.read_csv(support_path)
    support[selection_col] = (
        support[selection_col]
        .astype(str).str.strip().str.lower()
        .isin({"1", "true", "t", "yes"})
    )
    selected = support[
        (support["year"] == year) & support[selection_col]
    ].copy()
    ascending = rank_col in {"consensus_rank", "fdr_sum"}
    selected["metapath_rank"] = (
        selected.groupby("go_id")[rank_col]
        .rank(method="first", ascending=ascending)
        .astype(int)
    )
    selected = selected[selected["metapath_rank"] <= max_rank]
    return selected[["go_id", "metapath", "metapath_rank"]].drop_duplicates()


def _select_pairs_by_effective_n(
    dwpc_df: pd.DataFrame,
    *,
    min_n: int = 1,
    max_n: int | None = None,
) -> pd.DataFrame:
    rows = []
    for (go_id, metapath), group in dwpc_df.groupby(["go_id", "metapath"], sort=False):
        scores = group["dwpc"].to_numpy(dtype=float)
        eff_n = _effective_number(scores)
        k = max(min_n, int(np.ceil(eff_n)))
        if max_n is not None:
            k = min(k, max_n)
        top = group.nlargest(k, "dwpc").copy()
        top["pair_effective_n"] = float(eff_n)
        top["pair_rank"] = range(1, len(top) + 1)
        rows.append(top)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


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
    parser.add_argument("--selection-col", default="selected_by_effective_n_all")
    parser.add_argument("--max-metapath-rank", type=int, default=5)
    parser.add_argument("--effective-min-n", type=int, default=1)
    parser.add_argument("--effective-max-n", type=int, default=None)
    parser.add_argument("--path-enumeration-top-k", type=int, default=5000)
    parser.add_argument("--path-min-count", type=int, default=1)
    parser.add_argument("--path-max-count", type=int, default=None)
    parser.add_argument("--degree-d", type=float, default=0.5)
    parser.add_argument(
        "--go-id", default=None,
        help="Run a single GO term (for HPC array parallelization).",
    )
    parser.add_argument("--output-dir", default="output/year_subgraph_intermediates")
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct_dir = Path(args.direct_results_dir)
    support_path = Path(args.support_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_dir = REPO_ROOT / "data" / "edges"
    edge_loader = EdgeLoader(edges_dir)

    # Load direct-DWPC for 2024
    pattern = "dwpc_*_2024_real.csv"
    matches = sorted(direct_dir.glob(pattern))
    if not matches:
        print(f"Error: no files matching {pattern}", file=sys.stderr)
        sys.exit(1)
    chunks = []
    for path in matches:
        for chunk in pd.read_csv(path, chunksize=args.chunksize):
            chunks.append(chunk)
    direct = pd.concat(chunks, ignore_index=True)

    # 2016-selected metapaths
    selected_mp = _load_selected_metapaths(
        support_path, args.selection_col, year=2016,
        max_rank=args.max_metapath_rank,
    )

    # Filter to selected metapaths
    direct_sel = direct.merge(selected_mp[["go_id", "metapath"]], on=["go_id", "metapath"], how="inner")

    # Optional single GO term
    if args.go_id:
        direct_sel = direct_sel[direct_sel["go_id"] == args.go_id].copy()
        if direct_sel.empty:
            print(f"No data for GO term {args.go_id}")
            sys.exit(0)

    # Select effective-n pairs per (go_id, metapath)
    selected_pairs = _select_pairs_by_effective_n(
        direct_sel, min_n=args.effective_min_n, max_n=args.effective_max_n,
    )
    n_pairs = len(selected_pairs)
    n_go = selected_pairs["go_id"].nunique()
    print(f"Selected {n_pairs:,} pairs across {n_go:,} GO terms")

    # Enumerate paths
    path_rows = []
    for _, row in selected_pairs.iterrows():
        metapath_g = row["metapath"]
        metapath_bp = reverse_metapath_abbrev(metapath_g)
        nodes, edges = parse_metapath(metapath_bp)
        node_types = list(dict.fromkeys(nodes))
        maps = load_node_maps(REPO_ROOT, node_types)

        bp_id = row["go_id"]
        gene_id = int(row["entrez_gene_id"])
        bp_pos = maps.id_to_pos.get("BP", {}).get(bp_id)
        gene_pos = maps.id_to_pos.get("G", {}).get(gene_id)
        if bp_pos is None or gene_pos is None:
            continue

        try:
            candidate_k = int(args.path_enumeration_top_k)
            candidate_paths = enumerate_paths(
                bp_pos, gene_pos, nodes, edges, edge_loader,
                top_k=candidate_k, degree_d=args.degree_d,
            )
            paths = select_paths(
                candidate_paths,
                selection_method="effective_number",
                top_paths=5,
                path_cumulative_frac=None,
                path_min_count=int(args.path_min_count),
                path_max_count=args.path_max_count,
            )
        except Exception as exc:
            print(f"  Failed {metapath_bp} {bp_id} {gene_id}: {exc}")
            continue

        for path_rank, (score, pos_path) in enumerate(paths, start=1):
            id_path = []
            name_path = []
            for node_type, pos in zip(nodes, pos_path):
                node_id = maps.pos_to_id[node_type].get(int(pos))
                node_name = maps.id_to_name[node_type].get(node_id, str(node_id))
                id_path.append(str(node_id))
                name_path.append(str(node_name))

            # Intermediate nodes are everything except first (BP) and last (G)
            intermediate_ids = id_path[1:-1]
            intermediate_names = name_path[1:-1]

            path_rows.append({
                "go_id": bp_id,
                "metapath": metapath_g,
                "metapath_bp": metapath_bp,
                "entrez_gene_id": gene_id,
                "pair_rank": int(row["pair_rank"]),
                "pair_dwpc": float(row["dwpc"]),
                "path_rank": path_rank,
                "path_score": float(score),
                "intermediate_ids": "|".join(intermediate_ids),
                "intermediate_names": "|".join(intermediate_names),
                "path_nodes_ids": "|".join(id_path),
                "path_nodes_names": "|".join(name_path),
            })

    paths_df = pd.DataFrame(path_rows)
    suffix = f"_{args.go_id}" if args.go_id else ""
    paths_df.to_csv(out_dir / f"path_instances{suffix}.csv", index=False)
    print(f"Saved {len(paths_df):,} path instances to {out_dir / f'path_instances{suffix}.csv'}")

    if paths_df.empty:
        print("No paths enumerated.")
        return

    # --- Intermediate node sharing analysis ---
    # Explode intermediate nodes (for length-3 metapaths there are 2 intermediates)
    # Analyze at the level of individual intermediate positions
    sharing_rows = []
    for (go_id, metapath), group in paths_df.groupby(["go_id", "metapath"]):
        # Get unique (gene, intermediate) pairs from path instances
        gene_intermediates = []
        for _, r in group.iterrows():
            int_ids = str(r["intermediate_ids"]).split("|")
            int_names = str(r["intermediate_names"]).split("|")
            for idx, (int_id, int_name) in enumerate(zip(int_ids, int_names)):
                gene_intermediates.append({
                    "go_id": go_id,
                    "metapath": metapath,
                    "entrez_gene_id": int(r["entrez_gene_id"]),
                    "intermediate_position": idx + 1,
                    "intermediate_id": int_id,
                    "intermediate_name": int_name,
                })
        if not gene_intermediates:
            continue
        gi_df = pd.DataFrame(gene_intermediates).drop_duplicates()

        # Per intermediate node: how many distinct genes use it?
        int_sharing = gi_df.groupby(
            ["go_id", "metapath", "intermediate_position", "intermediate_id", "intermediate_name"],
            as_index=False,
        ).agg(n_genes_sharing=("entrez_gene_id", "nunique"))
        sharing_rows.append(int_sharing)

    if sharing_rows:
        sharing_df = pd.concat(sharing_rows, ignore_index=True)
        sharing_df = sharing_df.sort_values(
            ["go_id", "metapath", "intermediate_position", "n_genes_sharing"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)
        sharing_df.to_csv(out_dir / f"intermediate_sharing{suffix}.csv", index=False)
        print(f"Saved: {out_dir / f'intermediate_sharing{suffix}.csv'}")

        # Summary per (go_id, metapath)
        summary = sharing_df.groupby(["go_id", "metapath"], as_index=False).agg(
            n_unique_intermediates=("intermediate_id", "nunique"),
            max_genes_per_intermediate=("n_genes_sharing", "max"),
            mean_genes_per_intermediate=("n_genes_sharing", "mean"),
            n_shared_intermediates=("n_genes_sharing", lambda x: (x > 1).sum()),
        )
        n_genes_per_group = paths_df.groupby(["go_id", "metapath"], as_index=False).agg(
            n_genes=("entrez_gene_id", "nunique"),
        )
        summary = summary.merge(n_genes_per_group, on=["go_id", "metapath"], how="left")
        summary["frac_intermediates_shared"] = np.where(
            summary["n_unique_intermediates"] > 0,
            summary["n_shared_intermediates"] / summary["n_unique_intermediates"],
            0.0,
        )
        summary = summary.sort_values("max_genes_per_intermediate", ascending=False).reset_index(drop=True)
        summary.to_csv(out_dir / f"intermediate_summary{suffix}.csv", index=False)
        print(f"Saved: {out_dir / f'intermediate_summary{suffix}.csv'}")

        # Print global stats
        print(f"\n--- Intermediate node sharing ---")
        print(f"  (GO term, metapath) groups: {len(summary):,}")
        print(f"  Groups with shared intermediates: {(summary['n_shared_intermediates'] > 0).sum():,}")
        print(f"  Max genes sharing one intermediate: {summary['max_genes_per_intermediate'].max()}")
        print(f"  Median max_genes_per_intermediate: {summary['max_genes_per_intermediate'].median():.0f}")
        print(f"  Median frac_intermediates_shared: {summary['frac_intermediates_shared'].median():.3f}")


if __name__ == "__main__":
    main()
