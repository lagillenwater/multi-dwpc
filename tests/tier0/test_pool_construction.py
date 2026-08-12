from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tier0.pool_construction import build_pools_and_counts

SUBSTRATE = Path("output/end_to_end_2026_4_23/lv_experiment (1)")

pytestmark = pytest.mark.skipif(
    not SUBSTRATE.exists(), reason="requires the local end_to_end_2026_4_23 substrate"
)


def test_lv246_feature0_pools_partition_matches_sampled_gene_count():
    scores, pools, counts, observed = build_pools_and_counts("LV246", 0, SUBSTRATE)
    assert scores.shape == (20945,)
    # LV246 has 34 real genes (lv_top_genes.csv); every real gene lands in
    # exactly one of the 10 promiscuity bins, so counts must sum to 34.
    assert sum(counts) == 34
    assert len(pools) == len(counts) == 10
    # No real LV246 gene may appear in its own candidate pool.
    real_genes = pd.read_csv(SUBSTRATE / "lv_top_genes.csv")
    real_genes = real_genes[real_genes["lv_id"] == "LV246"]["gene_identifier"]
    gene_ids = np.load(SUBSTRATE / "gene_ids.npy", allow_pickle=True)
    real_idx = set(np.flatnonzero(np.isin(gene_ids, real_genes.to_numpy())))
    for pool in pools:
        assert real_idx.isdisjoint(set(pool.tolist()))


def test_observed_matches_real_feature_scores_csv():
    _, _, _, observed = build_pools_and_counts("LV246", 0, SUBSTRATE)
    real = pd.read_csv(SUBSTRATE / "real_feature_scores.csv")
    row = real[(real["lv_id"] == "LV246") & (real["feature_idx"] == 0)].iloc[0]
    assert observed == pytest.approx(float(row["real_mean"]))


def test_freshly_computed_bins_differ_from_gene_degree_table_degree_bin():
    """Documents why this task recomputes bins instead of reusing
    gene_degree_table.csv's degree_bin column: that column is raw hetnet
    graph degree, a different quantity from LV-membership promiscuity.
    """
    from src.bipartite_nulls import calculate_target_membership_counts, _assign_rank_bins

    top_genes = pd.read_csv(SUBSTRATE / "lv_top_genes.csv")
    gene_ids = np.load(SUBSTRATE / "gene_ids.npy", allow_pickle=True)
    universe = pd.DataFrame({"gene_identifier": gene_ids})

    promiscuity = calculate_target_membership_counts(
        top_genes.rename(columns={"lv_id": "source", "gene_identifier": "gene_identifier"}),
        source_col="source",
        target_col="gene_identifier",
        target_universe=universe["gene_identifier"],
    )
    fresh_bins = _assign_rank_bins(promiscuity["promiscuity"], 10)

    old = pd.read_csv(SUBSTRATE / "gene_degree_table.csv")
    # Both are valid qcut-into-10 partitions, but over different underlying
    # quantities -- they should disagree on a large fraction of genes.
    agreement = (fresh_bins.to_numpy() == old["degree_bin"].to_numpy()).mean()
    assert agreement < 0.5, (
        "fresh promiscuity bins unexpectedly match gene_degree_table.csv's "
        "degree_bin -- re-check whether reusing it is actually safe"
    )
