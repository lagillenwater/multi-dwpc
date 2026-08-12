import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.experiments.tier0_offline_recompute import (
    OUTPUT_DIR,
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


@pytest.mark.skipif(
    not SUBSTRATE.exists(), reason="requires the local end_to_end_2026_4_23 substrate"
)
def test_main_writes_length_separated_concordance_metrics(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tier0_offline_recompute.py",
            "--substrate-dir", str(SUBSTRATE),
            "--per-cell-max", "2",
            "--b", "200",
            "--dwpc-z-threshold", "1.65",
        ],
    )
    main()
    summary = (OUTPUT_DIR / "summary.md").read_text()

    assert "## Concordance by metapath length" in summary
    assert "### L=2" in summary
    assert "### L>=3" in summary
    # The length-specific subsections must carry actual concordance metrics
    # (Spearman rho / Jaccard), not just the pre-existing row counts.
    l2_section = summary.split("### L=2", 1)[1].split("### L>=3", 1)[0]
    lge3_section = summary.split("### L>=3", 1)[1]
    for section in (l2_section, lge3_section):
        assert (
            "spearman_rho_analytical_vs_mc_highb" in section
            or "No rows in this bucket" in section
        )
