import numpy as np

from src.tier0.analytical_null import analytical_null
from src.tier0.montecarlo_reference import sample_null_sets
from scripts.experiments.tier0_capacity_controls import plant_signal


def _setup():
    rng = np.random.default_rng(0)
    scores = rng.normal(5.0, 1.0, size=400)
    pools = [np.arange(0, 200), np.arange(200, 400)]
    counts = [3, 2]
    return rng, scores, pools, counts


def test_sample_null_sets_respects_strata_and_counts():
    rng, scores, pools, counts = _setup()
    idx = sample_null_sets(rng, pools, counts)
    assert len(idx) == 5
    assert len(np.intersect1d(idx, pools[0])) == 3
    assert len(np.intersect1d(idx, pools[1])) == 2
    assert len(set(idx.tolist())) == 5


def test_negative_control_z_is_standard_normal_shaped():
    rng, scores, pools, counts = _setup()
    res = analytical_null(scores, pools, counts, observed=0.0)
    zs = []
    for _ in range(2000):
        idx = sample_null_sets(rng, pools, counts)
        zs.append((scores[idx].mean() - res.mean) / res.std)
    zs = np.asarray(zs)
    assert abs(zs.mean()) < 0.1
    assert abs(zs.std() - 1.0) < 0.05


def test_plant_signal_preserves_stratum_profile_and_raises_score():
    rng, scores, pools, counts = _setup()
    drawn = sample_null_sets(rng, pools, counts)
    planted = plant_signal(scores, pools, counts, drawn, fraction=0.5, rng=rng)
    assert len(planted) == len(drawn)
    assert len(np.intersect1d(planted, pools[0])) == counts[0]
    assert len(np.intersect1d(planted, pools[1])) == counts[1]
    assert scores[planted].mean() > scores[drawn].mean()


def test_plant_signal_fraction_one_uses_top_scoring_candidates():
    rng, scores, pools, counts = _setup()
    drawn = sample_null_sets(rng, pools, counts)
    planted = plant_signal(scores, pools, counts, drawn, fraction=1.0, rng=rng)
    # Top scorers among candidates NOT already drawn -- a drawn member can
    # itself be a pool-wide top scorer, so compare against the candidate set.
    cand = pools[0][~np.isin(pools[0], drawn)]
    top0 = cand[np.argsort(scores[cand])[-counts[0]:]]
    assert set(np.intersect1d(planted, pools[0]).tolist()) == set(top0.tolist())
