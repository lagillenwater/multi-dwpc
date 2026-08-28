from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.experiments.tier0_b_convergence import _strategy_pool_fn, sweep_b
from src.tier0.metaedge_degree_pool_construction import MetaedgeDegreePoolStrategy
from src.tier0.pool_construction import build_pools_and_counts

SUBSTRATE = Path("output/end_to_end_2026_4_23/lv_experiment (1)")
DATA_DIR = Path("data")

pytestmark = pytest.mark.skipif(
    not SUBSTRATE.exists(), reason="requires the local end_to_end_2026_4_23 substrate"
)


def test_sweep_b_returns_expected_columns_and_length_buckets():
    df, _rows = sweep_b(
        SUBSTRATE, per_cell_max=3, b_values=[10, 100], dwpc_z_threshold=1.65,
        random_state=0, pool_fn=build_pools_and_counts,
    )
    expected_cols = {
        "b", "length_bucket", "n", "n_excluded_nan",
        "spearman_rho_analytical_vs_mc_highb", "spearman_rho_analytical_vs_original",
        "jaccard_selected_analytical_vs_mc_highb", "jaccard_selected_analytical_vs_original",
        "max_abs_relative_std_error", "max_abs_mean_diff_in_mc_se",
        "median_active_strata", "min_active_strata",
    }
    assert expected_cols <= set(df.columns)
    assert set(df["b"].unique()) <= {10, 100}
    assert set(df["length_bucket"].unique()) <= {"L=2", "L>=3"}
    # min_active_strata/median_active_strata must be internally consistent
    # (min <= median) and at least 1 (every row has >=1 active stratum by
    # construction -- a row can't be sampled with zero real genes in any bin).
    assert (df["min_active_strata"] >= 1).all()
    assert (df["min_active_strata"] <= df["median_active_strata"]).all()


def test_strategy_pool_fn_promiscuity_returns_build_pools_and_counts():
    assert _strategy_pool_fn("promiscuity") is build_pools_and_counts


def test_strategy_pool_fn_metaedge_degree_returns_working_strategy():
    """The actual integration point between Task 1 (sweep_b/--strategy) and
    Task 2 (MetaedgeDegreePoolStrategy): _strategy_pool_fn("metaedge_degree")
    must both construct the right type and return something callable that
    actually produces valid pools against the real substrate/data dir --
    not just importable."""
    pool_fn = _strategy_pool_fn("metaedge_degree")
    assert isinstance(pool_fn, MetaedgeDegreePoolStrategy)
    if DATA_DIR.exists():
        scores, pools, counts, observed = pool_fn("LV246", 0, SUBSTRATE)
        assert scores.shape == (20945,)
        assert len(pools) == len(counts) == 10


def test_strategy_pool_fn_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        _strategy_pool_fn("nonexistent_strategy")


@pytest.mark.skipif(
    not DATA_DIR.exists(), reason="requires the local data/ symlink"
)
def test_sweep_b_smoke_test_with_metaedge_degree_pool_fn():
    """Small real-substrate smoke test proving --strategy metaedge_degree's
    wiring works end to end through sweep_b, not just that the strategy
    object can be constructed. Deliberately small (2 B values, small
    per-cell-max) -- this isn't meant to be exhaustive, just to prove the
    strategy plugs into sweep_b the same way build_pools_and_counts does."""
    df, _rows = sweep_b(
        SUBSTRATE, per_cell_max=3, b_values=[10, 100], dwpc_z_threshold=1.65,
        random_state=0, pool_fn=MetaedgeDegreePoolStrategy(data_dir=DATA_DIR),
    )
    assert not df.empty
    assert set(df["b"].unique()) <= {10, 100}
    assert (df["min_active_strata"] >= 1).all()


def test_sweep_b_analytical_result_is_computed_once_not_per_b():
    """The analytical arm is B-independent -- z_analytical for a given row
    must be identical across every B in the sweep. Verified indirectly: two
    different B lists covering the same rows produce the same
    spearman_rho_analytical_vs_original (which only depends on z_analytical
    and z_original, neither of which varies with B) at every B they share.
    """
    df1, _rows1 = sweep_b(
        SUBSTRATE, per_cell_max=3, b_values=[10], dwpc_z_threshold=1.65,
        random_state=0, pool_fn=build_pools_and_counts,
    )
    df2, _rows2 = sweep_b(
        SUBSTRATE, per_cell_max=3, b_values=[10, 50], dwpc_z_threshold=1.65,
        random_state=0, pool_fn=build_pools_and_counts,
    )
    row1 = df1[(df1["b"] == 10) & (df1["length_bucket"] == "L=2")]
    row2 = df2[(df2["b"] == 10) & (df2["length_bucket"] == "L=2")]
    if not row1.empty and not row2.empty:
        assert row1["spearman_rho_analytical_vs_original"].iloc[0] == pytest.approx(
            row2["spearman_rho_analytical_vs_original"].iloc[0]
        )


def test_sweep_b_mc_metrics_vary_with_b_on_synthetic_data(monkeypatch):
    """Synthetic smoke test (no substrate dependency): a fake pool_fn with
    high null variance should show jaccard/rho actually change across a wide
    B range, proving the sweep isn't silently reusing one B's MC draw.
    """
    def fake_pool_fn(lv_id, feature_idx, substrate_dir):
        rng = np.random.default_rng(abs(hash((lv_id, feature_idx))) % (2**32))
        scores = rng.normal(size=200)
        pool = np.arange(200)
        observed = float(scores[:5].mean()) + 3.0
        return scores, [pool], [5], observed

    import scripts.experiments.tier0_b_convergence as mod

    fake_subsample = pd.DataFrame(
        {
            "lv_id": ["FAKE1", "FAKE2", "FAKE3", "FAKE4"],
            "feature_idx": [0, 1, 2, 3],
            "metapath": ["GaB", "GaB", "GaBcC", "GaBcC"],
            "length": [1, 1, 2, 2],
            "null_mean": [0.0, 0.0, 0.0, 0.0],
            "null_std": [1.0, 1.0, 1.0, 1.0],
        }
    )
    # Patch the name as looked up inside tier0_b_convergence.py -- it does
    # `from src.tier0.subsample import select_stratified_subsample`, so the
    # module's own attribute is what sweep_b() actually calls.
    monkeypatch.setattr(mod, "select_stratified_subsample", lambda *a, **k: fake_subsample)

    df, _rows = sweep_b(
        Path("."), per_cell_max=10, b_values=[10, 10000], dwpc_z_threshold=1.65,
        random_state=0, pool_fn=fake_pool_fn,
    )

    b10 = df[df["b"] == 10]["max_abs_mean_diff_in_mc_se"]
    b10000 = df[df["b"] == 10000]["max_abs_mean_diff_in_mc_se"]
    assert not b10.empty and not b10000.empty
    assert not np.isclose(b10.iloc[0], b10000.iloc[0])
