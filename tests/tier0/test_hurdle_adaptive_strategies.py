# tests/tier0/test_hurdle_adaptive_strategies.py
import numpy as np
import pandas as pd
import pytest

from src.tier0.hurdle_adaptive_pool_construction import (
    CapacityHurdleAdaptiveStrategy,
    MetaedgeDegreeHurdleAdaptiveStrategy,
)

N_GENES = 40


@pytest.fixture
def substrate(tmp_path):
    gene_ids = np.arange(101, 101 + N_GENES)
    np.save(tmp_path / "gene_ids.npy", gene_ids)
    rng = np.random.default_rng(0)
    scores = rng.gamma(1.0, 1.0, size=(N_GENES, 2)).astype(np.float32)
    np.save(tmp_path / "gene_feature_scores.npy", scores)
    pd.DataFrame(
        {"lv_id": ["LV1"] * 4, "gene_identifier": [101, 102, 110, 120]}
    ).to_csv(tmp_path / "lv_top_genes.csv", index=False)
    pd.DataFrame(
        {"lv_id": ["LV1", "LV1"], "feature_idx": [0, 1],
         "metapath": ["GaDlA", "GiGaD"], "length": [2, 2]}
    ).to_csv(tmp_path / "feature_manifest.csv", index=False)
    pd.DataFrame(
        {"lv_id": ["LV1", "LV1"], "feature_idx": [0, 1],
         "real_mean": [1.5, 2.5]}
    ).to_csv(tmp_path / "real_feature_scores.csv", index=False)
    pd.DataFrame(
        {"lv_id": ["LV1"], "target_id": ["UBERON:1"], "node_type": ["Anatomy"]}
    ).to_csv(tmp_path / "lv_targets.csv", index=False)
    return tmp_path


def _stubbed(strategy_cls, keys, gene_ids):
    s = strategy_cls(data_dir="data", min_stratum_size=5)
    s._keys = lambda metapath, lv_id, substrate_dir: np.asarray(keys, dtype=float)
    s._hetmat_gene_ids = lambda: np.asarray(gene_ids)
    return s


@pytest.mark.parametrize(
    "cls", [CapacityHurdleAdaptiveStrategy, MetaedgeDegreeHurdleAdaptiveStrategy]
)
def test_poolfn_contract_and_hurdle_exactness(cls, substrate):
    keys = np.zeros(N_GENES)
    keys[10:] = np.arange(1, N_GENES - 9)  # genes 0-9 are zero-key
    s = _stubbed(cls, keys, np.arange(101, 101 + N_GENES))
    scores, pools, counts, observed = s("LV1", 0, substrate)
    assert observed == pytest.approx(1.5)
    assert len(scores) == N_GENES
    assert sum(counts) == 4  # the LV's real genes
    # real genes excluded from their own pools
    real_rows = {0, 1, 9, 19}  # positions of 101,102,110,120
    for pool in pools:
        assert real_rows.isdisjoint(set(pool.tolist()))
    # hurdle stratum (bin 0) contains only zero-key rows
    zero_rows = set(range(10))
    assert set(pools[0].tolist()) <= zero_rows


@pytest.mark.parametrize(
    "cls", [CapacityHurdleAdaptiveStrategy, MetaedgeDegreeHurdleAdaptiveStrategy]
)
def test_fallback_merge_is_applied_and_logged(cls, substrate):
    # Put all four real genes on a unique maximal key so their stratum's
    # pool (after self-exclusion) is smaller than count=4.
    keys = np.ones(N_GENES)
    for pos in (0, 1, 9, 19):
        keys[pos] = 99.0
    keys[25] = 99.0  # one lone candidate at the same key
    s = _stubbed(cls, keys, np.arange(101, 101 + N_GENES))
    scores, pools, counts, observed = s("LV1", 0, substrate)
    assert all(c <= len(p) for p, c in zip(pools, counts) if c > 0)
    assert len(s.merge_log) == 1
    assert s.merge_log[0]["lv_id"] == "LV1"
    assert s.merge_log[0]["feature_idx"] == 0


def test_gene_order_mismatch_raises(substrate):
    s = _stubbed(
        CapacityHurdleAdaptiveStrategy,
        np.ones(N_GENES),
        np.arange(500, 500 + N_GENES),  # hetmat order != substrate order
    )
    with pytest.raises(ValueError, match="order"):
        s("LV1", 0, substrate)
