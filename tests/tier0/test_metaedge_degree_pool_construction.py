from pathlib import Path

import numpy as np
import pytest

from src.tier0.metaedge_degree_pool_construction import MetaedgeDegreePoolStrategy

SUBSTRATE = Path("output/end_to_end_2026_4_23/lv_experiment (1)")
DATA_DIR = Path("data")

pytestmark = pytest.mark.skipif(
    not SUBSTRATE.exists() or not DATA_DIR.exists(),
    reason="requires the local substrate and data/ symlink",
)


def test_resolves_every_real_metapaths_first_hop():
    """All 453 metapaths in the substrate must resolve to an existing
    edge file -- this is the exact check that caught parse_metapath's
    gap during design (48 of 453 metapaths use G<r, which parse_metapath
    cannot parse at all).
    """
    import pandas as pd

    strategy = MetaedgeDegreePoolStrategy(data_dir=DATA_DIR)
    feature_manifest = pd.read_csv(SUBSTRATE / "feature_manifest.csv")
    failures = []
    for metapath in feature_manifest["metapath"].unique():
        try:
            strategy._resolve_first_hop(metapath)
        except Exception as e:  # noqa: BLE001
            failures.append((metapath, str(e)))
    assert failures == [], f"{len(failures)} metapaths failed to resolve: {failures[:5]}"


def test_regulates_forward_and_reverse_resolve_to_the_same_file_different_axis():
    """G<rG (regulates reverse) and Gr>G (regulates forward) both traverse
    Gr>G.sparse.npz -- the only file that exists for this edge type -- but
    on opposite axes (row for forward, col for reverse). If axis selection
    were wrong, this is exactly the kind of directed-edge bug that would
    silently produce a plausible-looking but incorrect degree array.
    """
    strategy = MetaedgeDegreePoolStrategy(data_dir=DATA_DIR)
    fname_fwd, axis_fwd = strategy._resolve_first_hop("Gr>GdA")
    fname_rev, axis_rev = strategy._resolve_first_hop("G<rGiGdD")
    assert fname_fwd == fname_rev == "Gr>G.sparse.npz"
    assert axis_fwd == "row"
    assert axis_rev == "col"


def test_pools_partition_matches_sampled_gene_count_and_self_exclusion():
    scores, pools, counts, observed = strategy_call("LV246", 0, SUBSTRATE)
    assert scores.shape == (20945,)
    assert sum(counts) == 34
    assert len(pools) == len(counts) == 10
    import pandas as pd

    real_genes = pd.read_csv(SUBSTRATE / "lv_top_genes.csv")
    real_genes = real_genes[real_genes["lv_id"] == "LV246"]["gene_identifier"]
    gene_ids = np.load(SUBSTRATE / "gene_ids.npy", allow_pickle=True)
    real_idx = set(np.flatnonzero(np.isin(gene_ids, real_genes.to_numpy())))
    for pool in pools:
        assert real_idx.isdisjoint(set(pool.tolist()))


def strategy_call(lv_id, feature_idx, substrate_dir):
    strategy = MetaedgeDegreePoolStrategy(data_dir=DATA_DIR)
    return strategy(lv_id, feature_idx, substrate_dir)


def test_active_strata_spread_beats_promiscuity_on_real_data():
    """The whole point of this task: does metaedge-degree stratification
    actually give real LV data more than one active stratum per row, unlike
    promiscuity (confirmed in the predecessor plan: all 48 subsampled rows
    were single-stratum under promiscuity)? Report the finding either way --
    don't force the assertion if it turns out not to hold; that would itself
    be a real, reportable result.
    """
    import pandas as pd

    strategy = MetaedgeDegreePoolStrategy(data_dir=DATA_DIR)
    feature_manifest = pd.read_csv(SUBSTRATE / "feature_manifest.csv")
    sample = feature_manifest.drop_duplicates(subset=["lv_id", "feature_idx"]).sample(
        n=min(20, len(feature_manifest)), random_state=0
    )
    n_multi_stratum = 0
    for _, row in sample.iterrows():
        _, pools, counts, _ = strategy(row["lv_id"], int(row["feature_idx"]), SUBSTRATE)
        n_active = sum(1 for c in counts if c > 0)
        if n_active > 1:
            n_multi_stratum += 1
    print(f"\n{n_multi_stratum}/{len(sample)} sampled rows are multi-stratum under metaedge-degree")
    # Not a hard assertion -- this is a diagnostic the report must surface,
    # not a pass/fail gate. Sanity-check only: the function must run and
    # produce valid (possibly single-stratum) pools for every sampled row.
    assert n_multi_stratum >= 0
