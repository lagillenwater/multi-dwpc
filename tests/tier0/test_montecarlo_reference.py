import math
import numpy as np
import pytest

from src.tier0.analytical_null import analytical_null
from src.tier0.montecarlo_reference import montecarlo_reference


def test_mc_reference_converges_toward_analytical_moments():
    """At B=20,000 on a synthetic 2-stratum case, the MC reference's mean/std
    should land within Monte Carlo noise of the exact analytical moments --
    this is Tier 0's actual validation logic, exercised here on a case with
    a known-correct analytical answer.
    """
    rng = np.random.default_rng(7)
    scores = rng.gamma(2.0, 1.0, size=40)
    pools = [np.arange(20), np.arange(20, 40)]
    counts = [3, 2]
    observed = float(scores[:5].mean())

    exact = analytical_null(scores, pools, counts, observed)
    mc = montecarlo_reference(scores, pools, counts, observed, b=20_000, random_state=0)

    # SE of the MC mean estimate is sqrt(exact.var / b); allow a 6-sigma band.
    se = math.sqrt(exact.var / mc.b)
    assert abs(mc.mean - exact.mean) < 6 * se
    assert mc.std == pytest.approx(exact.std, rel=0.05)


def test_mc_reference_deterministic_given_seed():
    scores = np.arange(10, dtype=float)
    pools = [np.arange(10)]
    counts = [3]
    a = montecarlo_reference(scores, pools, counts, observed=5.0, b=500, random_state=1)
    b = montecarlo_reference(scores, pools, counts, observed=5.0, b=500, random_state=1)
    assert a.mean == b.mean
    assert a.std == b.std
