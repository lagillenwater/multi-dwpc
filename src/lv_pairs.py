"""
Pair construction utilities for LV multi-DWPC analysis.

Builds gene-target pairs where each LV's genes are paired with the LV's single target.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_lv_gene_target_pairs(
    top_genes_path: Path,
    lv_targets_path: Path,
    output_pairs_path: Path,
) -> pd.DataFrame:
    """
    Build LV-specific gene-target pairs from top genes and LV targets.

    Each LV has a single target. This creates the cross-product of
    (lv_genes) x (lv_target) for each LV.

    Args:
        top_genes_path: Path to lv_top_genes.csv
        lv_targets_path: Path to lv_targets.csv
        output_pairs_path: Path to write output pairs CSV

    Returns:
        DataFrame with one row per (lv_id, gene_identifier, target_id) pair.
        Columns: lv_id, gene_identifier, gene_symbol, loading, gene_rank,
                 target_id, target_name, node_type, target_position
    """
    top_genes = pd.read_csv(top_genes_path)
    lv_targets = pd.read_csv(lv_targets_path)

    required_genes = {"lv_id", "gene_identifier", "gene_symbol", "loading", "rank"}
    required_targets = {"lv_id", "target_id", "target_name", "node_type", "target_position"}

    missing_genes = required_genes - set(top_genes.columns)
    missing_targets = required_targets - set(lv_targets.columns)

    if missing_genes:
        raise ValueError(f"Missing columns in top genes file: {sorted(missing_genes)}")
    if missing_targets:
        raise ValueError(f"Missing columns in targets file: {sorted(missing_targets)}")

    top_genes["gene_identifier"] = top_genes["gene_identifier"].astype(str)

    # Merge genes with their LV's target
    pairs = top_genes.merge(lv_targets, on="lv_id", how="inner")
    if pairs.empty:
        raise ValueError("No gene-target pairs generated. Check LV ID alignment.")

    pairs = pairs.rename(columns={"rank": "gene_rank"})
    pairs = pairs.sort_values(["lv_id", "gene_rank", "target_id"]).reset_index(drop=True)

    # Select and order columns
    output_cols = [
        "lv_id",
        "gene_identifier",
        "gene_symbol",
        "loading",
        "gene_rank",
        "target_id",
        "target_name",
        "node_type",
        "target_position",
    ]
    pairs = pairs[output_cols]

    output_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_pairs_path, index=False)
    return pairs
