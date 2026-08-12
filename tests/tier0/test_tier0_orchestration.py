import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.experiments import tier0_offline_recompute
from scripts.experiments.tier0_offline_recompute import (
    _mc_seed_for_row,
    _stratum_coverage_caveat,
    compute_concordance_row,
    join_and_score,
    main,
)

SUBSTRATE = Path("output/end_to_end_2026_4_23/lv_experiment (1)")


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


def _base_fixture():
    return pd.DataFrame(
        {
            "z_original": [3.0, 1.0, 0.2, 2.5],
            "z_mc_highb": [2.9, 1.1, 0.1, 2.6],
            "z_analytical": [2.8, 0.9, 0.3, 2.4],
            "selected_original": [True, False, False, True],
            "selected_mc_highb": [True, False, False, True],
            "selected_analytical": [True, False, False, True],
            "mean_analytical": [0.0, 0.0, 0.0, 0.0],
            "mean_mc_highb": [0.0, 0.0, 0.0, 0.0],
            "std_analytical": [1.0, 1.0, 1.0, 1.0],
            "std_mc_highb": [1.0, 1.0, 1.0, 1.0],
            "b_mc_highb": [10_000, 10_000, 10_000, 10_000],
        }
    )


def test_join_and_score_spearman_and_jaccard():
    df = _base_fixture()
    metrics = join_and_score(df)
    assert metrics["spearman_rho_analytical_vs_mc_highb"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["jaccard_selected_analytical_vs_mc_highb"] == pytest.approx(1.0)
    assert metrics["n"] == 4
    assert metrics["n_excluded_nan"] == 0
    # Matched std/mean across arms in this fixture -> both magnitude checks are ~0.
    assert metrics["max_abs_relative_std_error"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["max_abs_mean_diff_in_mc_se"] == pytest.approx(0.0, abs=1e-9)


def test_magnitude_metrics_detect_std_scale_error_that_rank_metrics_miss():
    """Rank/threshold metrics (Spearman rho, Jaccard) stay at 1.0 even under a
    large multiplicative variance error, because scaling every z by the same
    factor preserves rank order, and when every row's z is already well clear
    of the selection threshold (as in the real 48-row substrate, where all
    real genes fall in a single promiscuity bin), scaling doesn't flip any
    selection outcome either. This is the actual Finding-1 defect: the
    headline metrics can't detect magnitude errors. max_abs_relative_std_error
    and max_abs_mean_diff_in_mc_se must catch what Spearman/Jaccard can't.
    """
    df = pd.DataFrame(
        {
            "z_original": [10.0, 20.0, 5.0, 15.0],
            "z_mc_highb": [10.0, 20.0, 5.0, 15.0],
            "z_analytical": [10.0, 20.0, 5.0, 15.0],
            "selected_original": [True, True, True, True],
            "selected_mc_highb": [True, True, True, True],
            "selected_analytical": [True, True, True, True],
            "mean_analytical": [0.0, 0.0, 0.0, 0.0],
            "mean_mc_highb": [0.0, 0.0, 0.0, 0.0],
            "std_analytical": [1.0, 1.0, 1.0, 1.0],
            "std_mc_highb": [1.0, 1.0, 1.0, 1.0],
            "b_mc_highb": [10_000, 10_000, 10_000, 10_000],
        }
    )
    # Simulate a 200% relative std error: analytical std is 3x the MC std,
    # and z_analytical scales along with it (same underlying bug: an
    # overstated null variance inflates both the null's std and the
    # resulting z). Every row stays far above the threshold either way, so
    # selection/rank never flip.
    df["z_analytical"] = df["z_mc_highb"] * 3.0
    df["std_analytical"] = df["std_mc_highb"] * 3.0
    df["selected_analytical"] = df["z_analytical"] >= 1.65

    metrics = join_and_score(df)
    assert metrics["spearman_rho_analytical_vs_mc_highb"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["jaccard_selected_analytical_vs_mc_highb"] == pytest.approx(1.0)
    # The magnitude check correctly flags the 200% std error the rank
    # metrics above missed entirely.
    assert metrics["max_abs_relative_std_error"] == pytest.approx(2.0, abs=1e-6)


def test_join_and_score_excludes_nan_rows_from_jaccard_and_reports_n_excluded():
    """Before the fix, compute_concordance_row's `bool(z >= threshold)`
    silently evaluated to False for NaN z, so a degenerate row (zero-
    variance null) was counted as "not selected" in Jaccard instead of being
    excluded -- diluting the Jaccard score even though Spearman already
    omitted it. Row 1 here has a NaN z_mc_highb (degenerate MC arm) but a
    real, large z_analytical -- under the old bug it would count as a
    Jaccard disagreement (selected_analytical=True, selected_mc_highb=False
    from the NaN); after the fix it's excluded from the comparison entirely.
    """
    df = pd.DataFrame(
        {
            "z_original": [3.0, 2.0, 0.2],
            "z_mc_highb": [2.9, np.nan, 0.1],
            "z_analytical": [2.8, 5.0, 0.3],
            "selected_original": [True, True, False],
            "selected_mc_highb": [True, False, False],  # row 1: False only via NaN-coercion
            "selected_analytical": [True, True, False],
            "mean_analytical": [0.0, 0.0, 0.0],
            "mean_mc_highb": [0.0, 0.0, 0.0],
            "std_analytical": [1.0, 1.0, 1.0],
            "std_mc_highb": [1.0, np.nan, 1.0],
            "b_mc_highb": [10_000, 10_000, 10_000],
        }
    )
    metrics = join_and_score(df)
    assert metrics["n_excluded_nan"] == 1
    assert metrics["n"] == 2
    # Only rows 0 and 2 are compared; both agree (row 0 selected in both,
    # row 2 selected in neither) -> perfect concordance, not diluted to 0.5
    # by treating row 1's NaN-derived False as a real disagreement.
    assert metrics["jaccard_selected_analytical_vs_mc_highb"] == pytest.approx(1.0)


def test_stratum_coverage_caveat_tracks_actual_counts():
    """New breakage B (regression fix): the caveat text previously asserted
    a fixed "every row is single-stratum" scenario regardless of the actual
    n_single_stratum count, which would self-contradict a computed "0 of N"
    on a substrate where rows aren't all single-stratum. Each branch's
    prose must match the counts passed in.
    """
    all_single = _stratum_coverage_caveat(n_rows=48, n_single_stratum=48)
    assert "all 48 subsampled rows use" in all_single
    assert "none of them exercises" in all_single

    partial = _stratum_coverage_caveat(n_rows=48, n_single_stratum=10)
    assert "10 of 48 subsampled rows" in partial
    assert "remaining 38 row(s) exercise 2+ strata" in partial
    # Must not claim every row is single-stratum when it isn't.
    assert "all 48 subsampled rows use" not in partial

    none_single = _stratum_coverage_caveat(n_rows=48, n_single_stratum=0)
    assert "Multi-stratum coverage" in none_single
    assert "all 48 subsampled rows exercise 2+ active" in none_single

    empty = _stratum_coverage_caveat(n_rows=0, n_single_stratum=0)
    assert "No subsampled rows" in empty


def test_mc_seed_varies_by_row_and_is_deterministic():
    """Finding 9: run_tier0 previously passed the same random_state to every
    row's montecarlo_reference call, so MC error was perfectly correlated
    across the table. _mc_seed_for_row must produce a distinct seed per
    (lv_id, feature_idx) while staying deterministic given the same inputs.
    """
    s_a = _mc_seed_for_row(0, "LV246", 7)
    s_b = _mc_seed_for_row(0, "LV246", 8)
    s_c = _mc_seed_for_row(0, "LV57", 7)
    assert s_a != s_b
    assert s_a != s_c
    assert _mc_seed_for_row(0, "LV246", 7) == s_a


@pytest.mark.skipif(
    not SUBSTRATE.exists(), reason="requires the local end_to_end_2026_4_23 substrate"
)
def test_main_writes_length_separated_concordance_metrics(monkeypatch, tmp_path):
    # Isolate from the real production output -- main() must not clobber
    # output/tier0_offline_recompute/ (the actual Step-5 deliverable).
    monkeypatch.setattr(tier0_offline_recompute, "OUTPUT_DIR", tmp_path / "tier0_offline_recompute")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tier0_offline_recompute.py",
            "--substrate-dir", str(SUBSTRATE),
            # per-cell-max matches the real Step-5 run so both length buckets
            # keep >=3 non-degenerate rows (spearmanr(nan_policy="omit")
            # raises below that) -- a small toy sample risked a crash here.
            "--per-cell-max", "15",
            "--b", "200",
            "--dwpc-z-threshold", "1.65",
        ],
    )
    main()
    summary = (tmp_path / "tier0_offline_recompute" / "summary.md").read_text()

    assert "## Concordance by metapath length" in summary
    assert "### L=2" in summary
    assert "### L>=3" in summary

    def rho_values(section: str) -> list[float]:
        return [
            float(v)
            for v in re.findall(r"spearman_rho_\S+\*\*:\s*(\S+)", section)
        ]

    l2_section = summary.split("### L=2", 1)[1].split("### L>=3", 1)[0]
    lge3_section = summary.split("### L>=3", 1)[1]
    for section in (l2_section, lge3_section):
        rhos = rho_values(section)
        assert rhos, "expected spearman_rho_* lines in this length bucket"
        assert all(math.isfinite(r) for r in rhos), (
            f"expected finite Spearman rho values, got {rhos} -- "
            "nan_policy='omit' regression?"
        )
        # Finding 4/5: n / n_excluded_nan must sit right next to the
        # correlation numbers in each length-bucket section, not just as a
        # misleading total-row count elsewhere in the file.
        assert "**n**:" in section
        assert "**n_excluded_nan**:" in section

    # Finding 1: magnitude-agreement checks must appear alongside the
    # rank-based metrics.
    assert "max_abs_relative_std_error" in summary
    assert "max_abs_mean_diff_in_mc_se" in summary

    # Finding 5: run parameters recorded for provenance.
    assert "## Run parameters" in summary
    assert f"substrate_dir**: {SUBSTRATE}" in summary
    assert "per_cell_max**: 15" in summary
    assert "b**: 200" in summary
    assert "dwpc_z_threshold**: 1.65" in summary
    assert "random_state**:" in summary

    # Finding 2: the single-promiscuity-stratum caveat must be explicit for
    # a human making the Tier-1 go/no-go call.
    assert "## Caveats" in summary
    assert "single active promiscuity stratum" in summary
    assert "tests/tier0/test_hetnetex_md_import.py" in summary

    # Finding 8: the fourth degenerate-input category must show up when it
    # fires (it always has on this real substrate as of this fix).
    degenerate_csv = (tmp_path / "tier0_offline_recompute" / "degenerate_inputs.csv").read_text()
    assert "zero_variance_null" in degenerate_csv
