"""Empirical high-B stratified-SRSWOR resampling reference: for each of B
draws, sample counts[r] genes without replacement from pools[r] per stratum,
union across strata, compute T = mean(scores[union]). This is the actual
Monte Carlo process the analytical kernel (Task 1) claims to describe
exactly in the B -> infinity limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import norm


@dataclass
class McReferenceResult:
    mean: float
    std: float
    z: float
    p: float
    b: int


def montecarlo_reference(
    scores: np.ndarray,
    pools: Sequence[np.ndarray],
    counts: Sequence[int],
    observed: float,
    b: int,
    random_state: int,
) -> McReferenceResult:
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(random_state)
    k_total = int(sum(counts))
    draws = np.empty(b, dtype=float)

    for i in range(b):
        sampled_idx = np.empty(k_total, dtype=np.int64)
        pos = 0
        for pool, k in zip(pools, counts):
            if k == 0:
                continue
            sampled_idx[pos : pos + k] = rng.choice(pool, size=k, replace=False)
            pos += k
        draws[i] = scores[sampled_idx].mean()

    mean = float(draws.mean())
    std = float(draws.std(ddof=1))
    if std <= 0.0 or not np.isfinite(std):
        return McReferenceResult(mean=mean, std=float("nan"), z=float("nan"), p=float("nan"), b=b)
    z = (observed - mean) / std
    p = float(norm.sf(z))
    return McReferenceResult(mean=mean, std=std, z=float(z), p=p, b=b)
