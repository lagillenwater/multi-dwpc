import numpy as np
import pandas as pd

from scripts.experiments.tier0_runtime_benchmark import benchmark_rows


def test_benchmark_rows_shape_and_positivity(monkeypatch, tmp_path):
    from scripts.experiments import tier0_runtime_benchmark as mod

    def stub_subsample(substrate_dir, per_cell_max, random_state):
        return pd.DataFrame(
            {"lv_id": ["LV1", "LV1"], "feature_idx": [0, 1],
             "length": [2, 3], "null_mean": [0.0, 0.0], "null_std": [1.0, 1.0]}
        )

    rng = np.random.default_rng(0)
    scores = rng.normal(size=300)

    def stub_pool_fn(lv_id, feature_idx, substrate_dir):
        return scores, [np.arange(0, 150), np.arange(150, 300)], [3, 2], 0.5

    monkeypatch.setattr(mod, "select_stratified_subsample", stub_subsample)
    monkeypatch.setattr(mod, "_strategy_pool_fn", lambda s: stub_pool_fn)

    df = benchmark_rows(tmp_path, "capacity_hurdle_adaptive", 15, [50, 100], 0)
    assert len(df) == 2 * 2  # rows x b_values
    assert (df["t_analytical_ms"] > 0).all()
    assert (df["t_mc_ms"] > 0).all()
    assert (df["speedup"] > 0).all()
    assert (df["t_poolfn_ms"] >= 0).all()  # spec diagnostic 4: pool-construction wall time
