from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.experiments.tier0_offline_recompute import check_degenerate_inputs

SUBSTRATE = Path("output/end_to_end_2026_4_23/lv_experiment (1)")

pytestmark = pytest.mark.skipif(
    not SUBSTRATE.exists(), reason="requires the local end_to_end_2026_4_23 substrate"
)


def test_check_degenerate_inputs_returns_expected_columns():
    result = check_degenerate_inputs(SUBSTRATE)
    assert list(result.columns) == [
        "lv_id", "feature_idx", "metapath", "issue",
    ]
    # Every flagged issue must be one of the three documented categories.
    assert result["issue"].isin(
        {"zero_variance_scores", "single_gene_stratum", "few_genes_with_paths"}
    ).all()
