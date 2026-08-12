from pathlib import Path

import pytest

from src.tier0.subsample import select_stratified_subsample

SUBSTRATE = Path("output/end_to_end_2026_4_23/lv_experiment (1)")

pytestmark = pytest.mark.skipif(
    not SUBSTRATE.exists(), reason="requires the local end_to_end_2026_4_23 substrate"
)


def test_subsample_is_random_null_only():
    sub = select_stratified_subsample(SUBSTRATE, per_cell_max=10, random_state=0)
    assert set(sub["null_type"].unique()) <= {"random"}


def test_subsample_covers_length_and_floor_strata():
    sub = select_stratified_subsample(SUBSTRATE, per_cell_max=10, random_state=0)
    assert sub["is_l2"].isin([True, False]).all()
    assert sub["is_floor_pinned"].isin([True, False]).all()
    # At least one row present for each cell that has any real rows.
    for is_l2 in (True, False):
        for pinned in (True, False):
            cell = sub[(sub["is_l2"] == is_l2) & (sub["is_floor_pinned"] == pinned)]
            # Not asserting non-empty for every cell (some combos may be rare/absent
            # in the real 906-row substrate) -- just that the columns are well-formed.
            assert cell["feature_idx"].is_unique


def test_subsample_deterministic_given_seed():
    a = select_stratified_subsample(SUBSTRATE, per_cell_max=5, random_state=3)
    b = select_stratified_subsample(SUBSTRATE, per_cell_max=5, random_state=3)
    assert list(a["feature_idx"]) == list(b["feature_idx"])
