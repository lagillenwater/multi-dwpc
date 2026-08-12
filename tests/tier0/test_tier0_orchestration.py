import numpy as np
import pandas as pd
import pytest

from scripts.experiments.tier0_offline_recompute import (
    compute_concordance_row,
    join_and_score,
)


def test_compute_concordance_row_matches_when_analytical_close_to_original():
    row = compute_concordance_row(
        z_original=2.0, z_mc_highb=2.05, z_analytical=2.02, dwpc_z_threshold=1.65
    )
    assert row["selected_original"] is True
    assert row["selected_mc_highb"] is True
    assert row["selected_analytical"] is True


def test_compute_concordance_row_flags_disagreement_below_threshold():
    row = compute_concordance_row(
        z_original=1.70, z_mc_highb=1.60, z_analytical=1.50, dwpc_z_threshold=1.65
    )
    assert row["selected_original"] is True
    assert row["selected_mc_highb"] is False
    assert row["selected_analytical"] is False


def test_join_and_score_spearman_and_jaccard():
    df = pd.DataFrame(
        {
            "z_original": [3.0, 1.0, 0.2, 2.5],
            "z_mc_highb": [2.9, 1.1, 0.1, 2.6],
            "z_analytical": [2.8, 0.9, 0.3, 2.4],
            "selected_original": [True, False, False, True],
            "selected_mc_highb": [True, False, False, True],
            "selected_analytical": [True, False, False, True],
        }
    )
    metrics = join_and_score(df)
    assert metrics["spearman_rho_analytical_vs_mc_highb"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["jaccard_selected_analytical_vs_mc_highb"] == pytest.approx(1.0)
