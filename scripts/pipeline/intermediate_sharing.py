#!/usr/bin/env python3
"""Compute intermediate sharing for year (2016 vs 2024) or LV gene sets.

Replaces the prior `scripts/pipeline/year_intermediate_sharing.py` and
`scripts/pipeline/lv_intermediate_sharing.py`. Selects metapaths by
permutation z-statistic (z > threshold) at one or more B values, enumerates
path instances per gene set, and computes intermediate-coverage and
gene-to-group sharing statistics.

Year mode:
- Splits each GO term's genes into 2016 (real) vs 2024-added cohorts
- Selects metapaths that meet z > threshold in BOTH years (consensus)
- Reports cross-year sharing (2024->2016) plus within-year sharing

LV mode:
- Treats each LV's top genes as a single cohort
- Selects metapaths per-LV with z > threshold
- Reports within-group sharing and intermediate coverage to a target node

Usage:
    python scripts/pipeline/intermediate_sharing.py --analysis-type year \\
        --b 10 --output-dir output/year_intermediate_sharing

    python scripts/pipeline/intermediate_sharing.py --analysis-type lv \\
        --b-values 2,5,10,20 --output-dir output/lv_intermediate_sharing
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _pct(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator > 0 else 0.0


def _median_or_zero(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _within_group_jaccard(gene_intermediates: dict[int, set[str]]) -> tuple[int, list[float]]:
    """Per gene, Jaccard of its intermediates vs the union of other genes' intermediates.

    Returns (n_sharing, jaccard_scores).
    """
    n_sharing = 0
    jaccard_scores: list[float] = []
    all_others_cache: set[str] | None = None
    if len(gene_intermediates) > 1:
        all_ints: set[str] = set()
        for v in gene_intermediates.values():
            all_ints.update(v)
        all_others_cache = all_ints

    for _gene_id, ints in gene_intermediates.items():
        other_ints = (all_others_cache - ints) if all_others_cache is not None else set()
        if ints & other_ints:
            n_sharing += 1
        union = ints | other_ints
        if union:
            jaccard_scores.append(len(ints & other_ints) / len(union))
    return n_sharing, jaccard_scores


# ---------------------------------------------------------------------------
# Year-specific
# ---------------------------------------------------------------------------


def _select_metapaths_year(
    results_df: pd.DataFrame, z_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select metapaths meeting z > threshold in BOTH 2016 and 2024.

    Returns (selected_df, dropped_df). Dropped tracks GO terms whose metapaths
    failed consensus selection.
    """
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    has_year = "year" in results_df.columns
    if has_year:
        above_2016 = results_df[
            (results_df["year"] == 2016) & (results_df["effect_size_z"] > z_threshold)
        ][["go_id", "metapath"]].drop_duplicates()
        above_2016["in_2016"] = True

        above_2024 = results_df[
            (results_df["year"] == 2024) & (results_df["effect_size_z"] > z_threshold)
        ][["go_id", "metapath"]].drop_duplicates()
        above_2024["in_2024"] = True

        consensus = above_2016.merge(above_2024, on=["go_id", "metapath"], how="inner")

        results_2016 = results_df[results_df["year"] == 2016].copy()
        consensus = consensus.merge(
            results_2016[["go_id", "metapath", "effect_size_z"]],
            on=["go_id", "metapath"],
            how="left",
        )
        consensus = consensus.sort_values(["go_id", "effect_size_z"], ascending=[True, False])
        consensus["metapath_rank"] = consensus.groupby("go_id").cumcount() + 1
        selected = consensus[["go_id", "metapath", "metapath_rank", "effect_size_z"]]
    else:
        above = results_df[results_df["effect_size_z"] > z_threshold].copy()
        above = above.sort_values(["go_id", "effect_size_z"], ascending=[True, False])
        above["metapath_rank"] = above.groupby("go_id").cumcount() + 1
        selected = above[["go_id", "metapath", "metapath_rank", "effect_size_z"]]

    selected_go = set(selected["go_id"].unique()) if not selected.empty else set()
    all_go = set(results_df["go_id"].unique())
    dropped_go = sorted(all_go - selected_go)
    if dropped_go:
        dropped_df = pd.DataFrame({
            "go_id": dropped_go,
            "reason": [f"No metapaths with z > {z_threshold} in both years"] * len(dropped_go),
        })
    else:
        dropped_df = pd.DataFrame()
    return selected, dropped_df


def _compute_sharing_stats_year(
    genes_2016_intermediates: dict[int, set[str]],
    genes_2024_intermediates: dict[int, set[str]],
    n_genes_2016_total: int,
    n_genes_2024_total: int,
) -> dict:
    all_2016: set[str] = set()
    for ints in genes_2016_intermediates.values():
        all_2016.update(ints)

    all_2024: set[str] = set()
    for ints in genes_2024_intermediates.values():
        all_2024.update(ints)

    n_2016 = len(genes_2016_intermediates)
    n_2024 = len(genes_2024_intermediates)

    n_2016_sharing, jaccard_2016_to_2016 = _within_group_jaccard(genes_2016_intermediates)
    n_2024_sharing_2024, jaccard_2024_to_2024 = _within_group_jaccard(genes_2024_intermediates)

    n_2024_sharing_with_2016 = 0
    jaccard_2024_to_2016: list[float] = []
    for ints in genes_2024_intermediates.values():
        if ints & all_2016:
            n_2024_sharing_with_2016 += 1
        union = ints | all_2016
        if union:
            jaccard_2024_to_2016.append(len(ints & all_2016) / len(union))

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
        "n_unique_intermediates_2016": len(all_2016),
        "n_unique_intermediates_2024": len(all_2024),
    }


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
    dwpc_z_stats: dict,
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
        z_score = mp_row["effect_size_z"]

        enum_kwargs = dict(
            target_pos=bp_pos, metapath=metapath,
            edge_loader=edge_loader, maps=maps,
            path_top_k=args.path_top_k,
            degree_d=args.degree_d,
            gene_set_id=go_id,
            dwpc_lookup=dwpc_lookup or None,
            dwpc_thresholds=dwpc_thresholds or None,
            dwpc_z_stats=dwpc_z_stats or None,
            dwpc_z_min=args.dwpc_z_threshold if args.dwpc_z_threshold is not None else 1.65,
            path_z_min=args.path_z_threshold,
            path_enumeration_cap=args.path_enumeration_cap,
            record_paths=gene_path_records,
        )

        ints_2016, n_filtered_2016 = enumerate_gene_intermediates(
            genes_2016, **enum_kwargs,
            record_extra={"go_id": go_id, "year": "2016"},
        )
        ints_2024, n_filtered_2024 = enumerate_gene_intermediates(
            genes_2024, **enum_kwargs,
            record_extra={"go_id": go_id, "year": "2024"},
        )

        stats = _compute_sharing_stats_year(ints_2016, ints_2024, len(genes_2016), len(genes_2024))
        stats["n_genes_filtered_by_dwpc_2016"] = n_filtered_2016
        stats["n_genes_filtered_by_dwpc_2024"] = n_filtered_2024
        coverage_stats, int_stats = compute_intermediate_coverage(
            {**ints_2016, **ints_2024}, node_name_maps
        )
        stats.update(coverage_stats)
        stats.update({
            "go_id": go_id,
            "metapath": metapath,
            "metapath_rank": mp_rank,
            "effect_size_z": z_score,
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


def _process_b_value_year(
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
    dwpc_z_stats: dict,
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

    selected_mp, dropped_go = _select_metapaths_year(
        results_df, z_threshold=args.effect_size_threshold
    )
    if not dropped_go.empty:
        dropped_go.to_csv(b_out_dir / "dropped_go_terms.csv", index=False)
        print(f"Dropped {len(dropped_go)} GO terms (no metapaths with z > {args.effect_size_threshold})")

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
        all_node_types: set[str] = set()
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
            dwpc_lookup, dwpc_thresholds, dwpc_z_stats, args,
        )
        sharing_rows.extend(s)
        top_intermediates_rows.extend(t)
        gene_path_records.extend(g)

    if sharing_rows:
        sharing_df = pd.DataFrame(sharing_rows)
        sharing_df.to_csv(b_out_dir / "intermediate_sharing_by_metapath.csv", index=False)

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

    if top_intermediates_rows:
        pd.DataFrame(top_intermediates_rows).to_csv(
            b_out_dir / "top_intermediates_by_metapath.csv", index=False
        )

    if gene_path_records:
        gene_paths_df = pd.DataFrame(gene_path_records)
        gene_paths_df["b"] = b_value
        gene_paths_df.to_csv(b_out_dir / "gene_paths.csv", index=False)

    print(f"Saved results to {b_out_dir}/")
    return maps


def _run_year(args: argparse.Namespace) -> None:
    year_output_dir = Path(args.year_output_dir)
    added_path = Path(args.added_pairs_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    try:
        dwpc_lookup = load_dwpc_from_numpy(year_output_dir, analysis_type="year")
    except FileNotFoundError as e:
        print(f"Note: {e}; --dwpc-percentile and --dwpc-z-threshold flags will be no-ops")

    dwpc_thresholds: dict[tuple, float] = {}
    dwpc_z_stats: dict[tuple, tuple[float, float]] = {}
    if dwpc_lookup:
        if args.dwpc_z_threshold is not None:
            dwpc_z_stats = compute_dwpc_z_stats(dwpc_lookup)
            print(f"DWPC z-filter active: keep genes with z >= {args.dwpc_z_threshold}")
        elif args.dwpc_percentile > 0:
            dwpc_thresholds = compute_dwpc_thresholds(dwpc_lookup, args.dwpc_percentile)

    runs_path = year_output_dir / "year_rank_stability_experiment" / "all_runs_long.csv"
    maps = None
    b_values = _resolve_b_values(args)

    for b_value in b_values:
        maps = _process_b_value_year(
            b_value, runs_path, out_dir,
            go_genes_df, added_pairs,
            edge_loader, maps, node_name_maps,
            dwpc_lookup, dwpc_thresholds, dwpc_z_stats, args,
        )

    print("\nAll B values processed.")


# ---------------------------------------------------------------------------
# LV-specific
# ---------------------------------------------------------------------------


SHARING_COL_ORDER_LV = [
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


def _select_metapaths_lv(
    results_df: pd.DataFrame, z_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-LV metapath selection by z > threshold. Returns (selected, dropped_lvs)."""
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
        max_z = group["effect_size_z"].max()

        above_threshold = group[group["effect_size_z"] > z_threshold].copy()

        if above_threshold.empty:
            dropped_rows.append({
                "lv_id": lv_id,
                "target_id": target_id,
                "target_name": target_name,
                "node_type": node_type,
                "n_metapaths_total": n_metapaths_total,
                "n_metapaths_selected": 0,
                "max_effect_size_z": max_z,
                "reason": f"No metapaths with permutation z > {z_threshold}",
            })
            continue

        above_threshold = above_threshold.sort_values(
            "effect_size_z", ascending=False
        ).reset_index(drop=True)
        above_threshold["metapath_rank"] = range(1, len(above_threshold) + 1)
        selected_rows.append(above_threshold)

    if not selected_rows:
        dropped_df = pd.DataFrame(dropped_rows) if dropped_rows else pd.DataFrame()
        return pd.DataFrame(), dropped_df

    return pd.concat(selected_rows, ignore_index=True), pd.DataFrame(dropped_rows)


def _compute_sharing_stats_lv(gene_intermediates: dict[int, set[str]]) -> dict:
    """Within-group sharing stats for one LV's gene set. Always returns 0.0 for empty data."""
    n_genes = len(gene_intermediates)
    if n_genes == 0:
        return {
            "n_genes_with_paths": 0,
            "n_genes_sharing": 0,
            "pct_genes_sharing": 0.0,
            "n_unique_intermediates": 0,
            "median_jaccard_to_group": 0.0,
            "mean_jaccard_to_group": 0.0,
        }

    n_sharing, jaccard_scores = _within_group_jaccard(gene_intermediates)

    all_intermediates: set[str] = set()
    for ints in gene_intermediates.values():
        all_intermediates.update(ints)

    return {
        "n_genes_with_paths": n_genes,
        "n_genes_sharing": n_sharing,
        "pct_genes_sharing": _pct(n_sharing, n_genes),
        "n_unique_intermediates": len(all_intermediates),
        "median_jaccard_to_group": _median_or_zero(jaccard_scores),
        "mean_jaccard_to_group": _mean_or_zero(jaccard_scores),
    }


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

        stats = _compute_sharing_stats_lv(gene_ints)
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


def _process_b_value_lv(
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

    selected_mp, dropped_lvs = _select_metapaths_lv(
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
        all_node_types: set[str] = set()
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
    sharing_df = sharing_df[[c for c in SHARING_COL_ORDER_LV if c in sharing_df.columns]]
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


def _run_lv(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_paths: list[Path] = []
    top_genes_frames: list[pd.DataFrame] = []
    lv_targets_frames: list[pd.DataFrame] = []
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

    node_name_maps = load_node_names(REPO_ROOT)
    edge_loader = EdgeLoader(REPO_ROOT / "data" / "edges")
    maps = None
    b_values = _resolve_b_values(args)

    for b_value in b_values:
        maps = _process_b_value_lv(
            b_value, runs_paths, out_dir,
            top_genes, lv_targets,
            edge_loader, maps, node_name_maps,
            dwpc_lookup, dwpc_thresholds, dwpc_z_stats, args,
        )

    print("\nAll B values processed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _resolve_b_values(args: argparse.Namespace) -> list[int]:
    if args.b_values:
        return [int(b.strip()) for b in args.b_values.split(",")]
    if args.b is not None:
        return [args.b]
    return [10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--analysis-type", required=True, choices=["year", "lv"],
        help="year (2016 vs 2024 GO-gene cohorts) or lv (per-LV gene sets).",
    )
    parser.add_argument("--output-dir", required=True)

    # Shared selection / filter knobs
    parser.add_argument("--b", type=int, default=None,
                        help="Single B value for metapath selection.")
    parser.add_argument("--b-values", default=None,
                        help="Comma-separated B values (e.g., '2,5,10,20'). Overrides --b. Default 10 if neither set.")
    parser.add_argument("--effect-size-threshold", type=float, default=0.5,
                        help="Minimum permutation z (diff / null_std) for metapath selection. Default 0.5.")
    parser.add_argument("--dwpc-percentile", type=float, default=75.0,
                        help="DWPC percentile threshold per (gene_set, metapath). Ignored when --dwpc-z-threshold is set. Default 75 (top 25%%).")
    parser.add_argument("--dwpc-z-threshold", type=float, default=None,
                        help="Keep genes whose (DWPC - mean) / std >= threshold. Overrides --dwpc-percentile.")
    parser.add_argument("--path-top-k", type=int, default=100,
                        help="Cap on paths per gene (legacy). Ignored when --path-z-threshold is set.")
    parser.add_argument("--path-z-threshold", type=float, default=None,
                        help="Keep paths whose z >= threshold per (gene_set, metapath). Overrides --path-top-k.")
    parser.add_argument("--path-enumeration-cap", type=int, default=None,
                        help="Max paths to enumerate per gene (used with --path-z-threshold). Default: no cap.")
    parser.add_argument("--degree-d", type=float, default=0.5,
                        help="Degree-weight exponent for path enumeration (passed to enumerate_gene_intermediates). Default 0.5.")

    # Year-only flags
    parser.add_argument("--year-output-dir", default="output/year_experiment",
                        help="[year only] Year experiment output directory.")
    parser.add_argument("--added-pairs-path",
                        default="output/intermediate/upd_go_bp_2024_added.csv",
                        help="[year only] CSV with 2024-added GO-gene pairs.")
    parser.add_argument("--go-id", default=None,
                        help="[year only] Single GO term (for HPC array jobs).")

    # LV-only flags
    parser.add_argument("--lv-output-dirs", nargs="+", default=["output/lv_experiment"],
                        help="[lv only] One or more LV experiment output directories.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis_type == "year":
        _run_year(args)
    else:
        _run_lv(args)


if __name__ == "__main__":
    main()
