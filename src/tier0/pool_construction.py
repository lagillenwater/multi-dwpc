"""Builds (scores, pools, counts, observed) for one (lv_id, feature_idx) row,
mirroring exactly what src.bipartite_nulls.generate_promiscuity_matched_samples
does -- fixed-bin (n_bins=10) stratified SRSWOR by LV-membership promiscuity,
excluding the LV's own real genes from their own candidate pools.

Deliberately recomputes bins from lv_top_genes.csv rather than reusing
gene_degree_table.csv's degree_bin column -- see
tests/tier0/test_pool_construction.py::test_freshly_computed_bins_differ_from_gene_degree_table_degree_bin
for why those are not interchangeable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.bipartite_nulls import calculate_target_membership_counts, _assign_rank_bins

N_BINS = 10


def _load_substrate(substrate_dir: Path):
    gene_ids = np.load(substrate_dir / "gene_ids.npy", allow_pickle=True)
    gene_scores = np.load(substrate_dir / "gene_feature_scores.npy")
    top_genes = pd.read_csv(substrate_dir / "lv_top_genes.csv")
    real_scores = pd.read_csv(substrate_dir / "real_feature_scores.csv")
    return gene_ids, gene_scores, top_genes, real_scores


def build_pools_and_counts(
    lv_id: str, feature_idx: int, substrate_dir: Path
) -> tuple[np.ndarray, list[np.ndarray], list[int], float]:
    substrate_dir = Path(substrate_dir)
    gene_ids, gene_scores, top_genes, real_scores = _load_substrate(substrate_dir)

    scores = gene_scores[:, feature_idx].astype(float)

    membership = top_genes.rename(columns={"lv_id": "source"})
    promiscuity = calculate_target_membership_counts(
        membership,
        source_col="source",
        target_col="gene_identifier",
        target_universe=gene_ids,
    )
    promiscuity["bin_id"] = _assign_rank_bins(promiscuity["promiscuity"], N_BINS)
    # Row order of `promiscuity` follows `target_universe`, i.e. gene_ids order.
    bin_of_row = promiscuity["bin_id"].to_numpy()

    this_lv_genes = set(top_genes[top_genes["lv_id"] == lv_id]["gene_identifier"])
    real_row_idx = np.flatnonzero(np.isin(gene_ids, list(this_lv_genes)))
    real_bins = bin_of_row[real_row_idx]

    pools: list[np.ndarray] = []
    counts: list[int] = []
    for b in range(N_BINS):
        candidate_rows = np.flatnonzero(bin_of_row == b)
        candidate_rows = candidate_rows[~np.isin(candidate_rows, real_row_idx)]
        pools.append(candidate_rows)
        counts.append(int((real_bins == b).sum()))

    row = real_scores[
        (real_scores["lv_id"] == lv_id) & (real_scores["feature_idx"] == feature_idx)
    ].iloc[0]
    observed = float(row["real_mean"])

    return scores, pools, counts, observed
