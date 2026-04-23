#!/usr/bin/env python3
"""Sweep DWPC percentile thresholds and measure sharing metric sensitivity.

Runs intermediate sharing at multiple percentile cutoffs for a subset of
gene sets.  Reports how sharing metrics change as a function of the
percentile, revealing whether the default (75th) is on a plateau or a
steep slope.

Usage:
    python scripts/experiments/dwpc_filter_sensitivity.py \
        --analysis-type lv \
        --lv-output-dirs output/lv_single_target_refactor \
        --b 10 \
        --percentiles 0,25,50,75,90 \
        --output-dir output/dwpc_filter_sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.path_enumeration import EdgeLoader, load_node_maps, load_node_names, parse_metapath  # noqa: E402
from src.intermediate_sharing import (  # noqa: E402
    compute_dwpc_thresholds,
    compute_intermediate_coverage,
    enumerate_gene_intermediates,
    load_dwpc_from_numpy,
    load_runs_at_b,
)
from src.lv_dwpc import NODE_TYPE_TO_ABBREV  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-type", choices=["lv", "year"], required=True)
    p.add_argument("--lv-output-dirs", nargs="+", required=True)
    p.add_argument("--b", type=int, required=True)
    p.add_argument("--percentiles", default="0,25,50,75,90")
    p.add_argument("--gene-set-ids", default=None, help="Comma-separated subset of gene set IDs to test")
    p.add_argument("--z-threshold", type=float, default=0.5)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    percentiles = [float(x) for x in args.percentiles.split(",")]

    id_col = "lv_id" if args.analysis_type == "lv" else "go_id"
    lv_dirs = [Path(d) for d in args.lv_output_dirs]

    runs_paths = []
    for d in lv_dirs:
        subdir = "lv_rank_stability_experiment" if args.analysis_type == "lv" else "year_rank_stability_experiment"
        candidate = d / subdir / "all_runs_long.csv"
        if candidate.exists():
            runs_paths.append(candidate)

    if not runs_paths:
        print("No all_runs_long.csv found in any output directory")
        return

    group_cols = (
        ["lv_id", "target_id", "target_name", "node_type", "metapath"]
        if args.analysis_type == "lv"
        else ["go_id", "metapath", "year"]
    )

    results_frames = []
    for rp in runs_paths:
        results_frames.append(load_runs_at_b(rp, args.b, group_cols))
    results_df = pd.concat(results_frames, ignore_index=True)

    selected = results_df[results_df["effect_size_z"] > args.z_threshold].copy()
    if selected.empty:
        print(f"No metapaths with z > {args.z_threshold}")
        return

    if args.gene_set_ids:
        ids = [x.strip() for x in args.gene_set_ids.split(",")]
        selected = selected[selected[id_col].isin(ids)]

    dwpc_lookup = load_dwpc_from_numpy(lv_dirs, analysis_type=args.analysis_type)
    if not dwpc_lookup:
        print("No DWPC data found -- cannot run percentile sensitivity")
        return

    edges_dir = REPO_ROOT / "data" / "edges"
    edge_loader = EdgeLoader(edges_dir)
    node_name_maps = load_node_names(REPO_ROOT)

    all_node_types = set()
    for mp in selected["metapath"].unique():
        nodes, _ = parse_metapath(mp)
        all_node_types.update(nodes)
    maps = load_node_maps(REPO_ROOT, list(all_node_types))

    top_genes_path = None
    for d in lv_dirs:
        candidate = d / ("lv_top_genes.csv" if args.analysis_type == "lv" else "go_top_genes.csv")
        if candidate.exists():
            top_genes_path = candidate
            break

    if top_genes_path is None:
        print("No top_genes.csv found")
        return

    top_genes_df = pd.read_csv(top_genes_path)
    gene_id_col = "gene_identifier"

    rows = []
    gene_set_ids = sorted(selected[id_col].unique())
    print(f"Running sensitivity for {len(gene_set_ids)} gene sets, {len(percentiles)} percentiles")

    for gs_id in gene_set_ids:
        gs_selected = selected[selected[id_col] == gs_id]
        gs_genes = top_genes_df[top_genes_df[id_col] == gs_id][gene_id_col].astype(int).tolist()
        if not gs_genes:
            continue

        target_pos = None
        if args.analysis_type == "lv" and "target_id" in gs_selected.columns:
            tid = gs_selected["target_id"].iloc[0]
            node_type = gs_selected["node_type"].iloc[0] if "node_type" in gs_selected.columns else "BP"
            nt_key = NODE_TYPE_TO_ABBREV.get(node_type, node_type)
            target_pos = maps.id_to_pos.get(nt_key, {}).get(tid)
        elif args.analysis_type == "year":
            target_pos = maps.id_to_pos.get("BP", {}).get(gs_id)

        if target_pos is None:
            continue

        for metapath in gs_selected["metapath"].unique():
            z_val = float(gs_selected[gs_selected["metapath"] == metapath]["effect_size_z"].iloc[0])

            for pct in percentiles:
                if pct > 0:
                    thresholds = compute_dwpc_thresholds(dwpc_lookup, pct)
                else:
                    thresholds = None

                t0 = time.time()
                gene_ints, n_filtered = enumerate_gene_intermediates(
                    gs_genes, target_pos, metapath, edge_loader, maps,
                    gene_set_id=gs_id,
                    dwpc_lookup=dwpc_lookup if pct > 0 else None,
                    dwpc_thresholds=thresholds,
                )
                elapsed = time.time() - t0

                n_with_paths = len(gene_ints)
                if n_with_paths > 0:
                    all_ints = set()
                    for v in gene_ints.values():
                        all_ints.update(v)
                    n_sharing = sum(
                        1 for g, ints in gene_ints.items()
                        if ints & (all_ints - ints)
                    )
                    coverage_stats, _ = compute_intermediate_coverage(gene_ints, node_name_maps)
                else:
                    n_sharing = 0
                    coverage_stats = {}

                rows.append({
                    "gene_set_id": gs_id,
                    "metapath": metapath,
                    "effect_size_z": z_val,
                    "percentile": pct,
                    "n_genes_total": len(gs_genes),
                    "n_genes_filtered": n_filtered,
                    "n_genes_with_paths": n_with_paths,
                    "pct_genes_sharing": n_sharing / n_with_paths * 100 if n_with_paths > 0 else 0,
                    "top1_coverage": coverage_stats.get("top1_intermediate_coverage", 0),
                    "top5_coverage": coverage_stats.get("top5_intermediate_coverage", 0),
                    "n_unique_intermediates": coverage_stats.get("n_shared_intermediates_2plus", 0),
                    "wall_time_s": elapsed,
                })

            print(f"  {gs_id} / {metapath}: done ({len(percentiles)} percentiles)")

    result_df = pd.DataFrame(rows)
    out_path = out_dir / "dwpc_filter_sensitivity.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(result_df)} rows)")


if __name__ == "__main__":
    main()
