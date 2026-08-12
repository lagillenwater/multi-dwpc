from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.experiments.tier0_offline_recompute import check_degenerate_inputs, run_tier0

SUBSTRATE = Path("output/end_to_end_2026_4_23/lv_experiment (1)")

pytestmark = pytest.mark.skipif(
    not SUBSTRATE.exists(), reason="requires the local end_to_end_2026_4_23 substrate"
)


def test_check_degenerate_inputs_returns_expected_columns():
    result = check_degenerate_inputs(SUBSTRATE)
    assert list(result.columns) == [
        "lv_id", "feature_idx", "metapath", "issue",
    ]
    # Every flagged issue must be one of the three column-level categories
    # (the fourth, pool-level category only fires when a concordance_df is
    # passed in -- see test_check_degenerate_inputs_flags_zero_variance_null).
    assert result["issue"].isin(
        {"zero_variance_scores", "single_gene_stratum", "few_genes_with_paths"}
    ).all()


def test_check_degenerate_inputs_without_concordance_df_has_no_zero_variance_null():
    result = check_degenerate_inputs(SUBSTRATE)
    assert "zero_variance_null" not in set(result["issue"])


def test_check_degenerate_inputs_flags_zero_variance_null():
    """Finding 8: a pool-level zero-variance null (analytical_null(...).var
    == 0) is a different, pool-specific defect from the three column-level
    checks above -- a row's full score column can have nonzero variance
    while every nonzero-scoring gene sits outside the sampled pool for that
    row's stratum. This can only be checked for rows actually run through
    run_tier0 (needs the pool/analytical-null context), not the full
    feature_manifest.
    """
    concordance = run_tier0(
        SUBSTRATE, per_cell_max=15, b=200, dwpc_z_threshold=1.65, random_state=0
    )
    result = check_degenerate_inputs(SUBSTRATE, concordance_df=concordance)
    zero_var_null = result[result["issue"] == "zero_variance_null"]

    # Every row run_tier0 flagged as having a zero-variance analytical null
    # (NaN z_analytical) must show up as zero_variance_null here, and only
    # those rows.
    expected_n = int((concordance["var_analytical"] == 0.0).sum())
    assert expected_n > 0, "expected at least one degenerate row on this real substrate"
    assert len(zero_var_null) == expected_n

    nan_z_n = int(concordance["z_analytical"].isna().sum())
    # Document whether this fourth category explains all NaN-z rows on this
    # substrate; do not force the number if it doesn't line up exactly.
    assert nan_z_n == expected_n, (
        f"zero_variance_null ({expected_n}) does not exactly match "
        f"NaN z_analytical rows ({nan_z_n}) -- some other cause of NaN z "
        "exists and should be investigated separately."
    )
