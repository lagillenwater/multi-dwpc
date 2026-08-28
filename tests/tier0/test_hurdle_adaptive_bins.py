import numpy as np
import pytest

from src.tier0.hurdle_adaptive_bins import hurdle_adaptive_bins


def test_zero_keys_form_exclusive_hurdle_stratum():
    keys = np.array([0, 0, 0, 1, 1, 2, 3, 5, 8, 8])
    bins = hurdle_adaptive_bins(keys, min_stratum_size=3)
    assert set(bins[keys == 0]) == {0}
    assert 0 not in set(bins[keys > 0])


def test_equal_keys_never_split():
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 6, size=200)
    bins = hurdle_adaptive_bins(keys, min_stratum_size=7)
    for v in np.unique(keys):
        assert len(set(bins[keys == v])) == 1, f"key {v} split across strata"


def test_min_stratum_size_met_except_none():
    rng = np.random.default_rng(1)
    keys = np.concatenate([np.zeros(50), rng.integers(1, 40, size=300)])
    bins = hurdle_adaptive_bins(keys, min_stratum_size=25)
    sizes = np.bincount(bins)
    assert (sizes[1:] >= 25).all(), f"non-hurdle stratum below min size: {sizes}"


def test_bins_ascend_with_key_value():
    keys = np.array([0.0, 0.5, 0.5, 1.2, 1.2, 3.0, 3.0, 9.9, 9.9, 9.9])
    bins = hurdle_adaptive_bins(keys, min_stratum_size=2)
    order = np.argsort(keys)
    assert (np.diff(bins[order]) >= 0).all()


def test_all_zero_keys_single_stratum():
    bins = hurdle_adaptive_bins(np.zeros(10), min_stratum_size=5)
    assert set(bins) == {0}


def test_no_zero_keys_no_empty_hurdle():
    keys = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    bins = hurdle_adaptive_bins(keys, min_stratum_size=2)
    sizes = np.bincount(bins)
    assert (sizes > 0).all()


def test_continuous_keys_work():
    rng = np.random.default_rng(2)
    keys = np.concatenate([np.zeros(20), rng.gamma(2.0, 3.0, size=500)])
    bins = hurdle_adaptive_bins(keys, min_stratum_size=50)
    sizes = np.bincount(bins)
    assert sizes[0] == 20
    assert (sizes[1:] >= 50).all()


def test_negative_keys_rejected():
    with pytest.raises(ValueError):
        hurdle_adaptive_bins(np.array([-1.0, 0.0, 1.0]), min_stratum_size=2)
