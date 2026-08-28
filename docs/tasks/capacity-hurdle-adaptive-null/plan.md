# Capacity Hurdle+Adaptive Null Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate two hurdle+adaptive stratification strategies for the HetNetEX-MD analytical null — capacity-keyed (primary) and first-hop-degree-keyed (ablation) — through the existing Tier 0 three-arm harness, with calibration and planted-signal controls and a runtime benchmark, run on Alpine.

**Architecture:** New pure modules (`hurdle_adaptive_bins`, `capacity`) feed two new pool-construction strategies that satisfy the existing `PoolFn` contract `(lv_id, feature_idx, substrate_dir) -> (scores, pools, counts, observed)`, so the existing `sweep_b` / `montecarlo_reference` / `analytical_null` machinery runs unchanged on identical partitions in both arms. New control and benchmark scripts reuse the same strategies. The full campaign runs as two chained Alpine jobs (prewarm DWPC cache subset, then validation).

**Tech Stack:** Python (numpy, scipy, pandas), pytest, HetNetEX-MD (`exact_resampling_moments` only), Slurm on Alpine.

**Spec:** `docs/tasks/capacity-hurdle-adaptive-null/design.md` (same folder — read it first; this plan implements it).

## Global Constraints

- Only `exact_resampling_moments` may be imported from HetNetEX-MD, via `src/tier0/hetnetex_md_import.py`; z/p are always derived locally (`src/tier0/analytical_null.py`).
- Hurdle stratum = keys exactly 0. Adaptive bins are value-respecting: equal keys are NEVER split across strata. `min_stratum_size` default 50.
- Feasibility fallback: merge a deficient stratum (candidate pool smaller than its count) into its lower-key neighbor (next-higher for the lowest), deterministically, and record every merge.
- Capacity key is computed on the RAW DWPC scale (row sum minus the feature's target column); observed statistics stay on the transformed (arcsinh) scale.
- Partitions depend only on (metapath, target, graph) — never on the tested gene set. Self-exclusion happens only in `pools_from_bins`.
- Seeds: `random_state=0` top-level; per-row MC seeds via `_mc_seed_for_row` (from `scripts/experiments/tier0_offline_recompute.py`).
- B sweep: 10, 30, 100, 300, 1000, 3000, 10000. Negative control: 200 draws/row. Positive control fractions: 0.1, 0.25, 0.5. Benchmark B: 1000 and 10000.
- Campaign outputs land under `output/tier0_capacity_hurdle/`.
- Every python/pytest call: `conda activate multi_dwpc && ...`. Run `conda activate multi_dwpc && python -m pytest tests/tier0 -q` before any push.
- Stage files explicitly by path (`git add -A` / `.` is denied). Push only `origin` on this branch (`fix/random-null-stratified-srswor`). Alpine: sbatch/single-id scancel/git pull are granted; everything else prompts.
- Work happens in the worktree `.worktrees/fix-random-null-stratified-srswor` (this file's repo root).

---

### Task 1: Hurdle+adaptive binning core

**Files:**
- Create: `src/tier0/hurdle_adaptive_bins.py`
- Test: `tests/tier0/test_hurdle_adaptive_bins.py`

**Interfaces:**
- Consumes: nothing (pure numpy).
- Produces: `hurdle_adaptive_bins(keys: np.ndarray, min_stratum_size: int = 50) -> np.ndarray` — dense int bin ids (0..M-1) per element, in key-ascending order; if any key == 0, bin 0 is exactly the zero-key elements and contains nothing else.

- [ ] **Step 1: Write the failing tests**

```python
# tests/tier0/test_hurdle_adaptive_bins.py
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


def test_min_stratum_size_met_except_possibly_none():
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_hurdle_adaptive_bins.py -q`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `src.tier0.hurdle_adaptive_bins`.

- [ ] **Step 3: Write the implementation**

```python
# src/tier0/hurdle_adaptive_bins.py
"""Hurdle + value-respecting adaptive binning for null stratification keys.

Replaces ``_assign_rank_bins`` for the hurdle+adaptive strategies: rank
binning with positional tie-breaking can split equal keys into different
strata (the Entrez-ID-block artifact behind the vacuous Tier 0 promiscuity
result -- see docs/tasks/capacity-hurdle-adaptive-null/design.md). Here
strata are contiguous in key value, equal keys always share a stratum, and
key == 0 (an exact exchangeability class for both first-hop degree and
leave-target-out capacity) is isolated as its own hurdle stratum.
"""

from __future__ import annotations

import numpy as np


def hurdle_adaptive_bins(keys: np.ndarray, min_stratum_size: int = 50) -> np.ndarray:
    """Return a dense bin id (0..M-1) per element, ascending in key value.

    Bin ids ascend with key value. If any key is exactly 0, those elements
    form their own stratum (the lowest bin) regardless of its size. Positive
    keys are grouped greedily in ascending distinct-value order, closing a
    stratum once it holds >= min_stratum_size elements; a trailing stratum
    below the minimum is merged into its predecessor (or stands alone when
    it is the only positive stratum).
    """
    keys = np.asarray(keys, dtype=float)
    if keys.ndim != 1:
        raise ValueError("keys must be 1-D")
    if (keys < 0).any():
        raise ValueError("keys must be non-negative")

    bins = np.empty(len(keys), dtype=int)
    has_hurdle = bool((keys == 0).any())
    if has_hurdle:
        bins[keys == 0] = 0

    pos_mask = keys > 0
    if not pos_mask.any():
        return bins

    values, counts = np.unique(keys[pos_mask], return_counts=True)
    # Greedy pass over distinct values in ascending order: a stratum closes
    # once it reaches min_stratum_size. Equal keys enter together, so they
    # can never straddle a boundary.
    boundaries = []  # index into `values` where each stratum starts
    acc = 0
    for i, c in enumerate(counts):
        if acc == 0:
            boundaries.append(i)
        acc += c
        if acc >= min_stratum_size:
            acc = 0
    if acc > 0 and len(boundaries) > 1:
        # Trailing under-filled stratum: merge into its predecessor.
        boundaries.pop()

    value_bin = np.empty(len(values), dtype=int)
    first = 1 if has_hurdle else 0
    for b, start in enumerate(boundaries):
        end = boundaries[b + 1] if b + 1 < len(boundaries) else len(values)
        value_bin[start:end] = first + b

    idx = np.searchsorted(values, keys[pos_mask])
    bins[pos_mask] = value_bin[idx]
    return bins
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_hurdle_adaptive_bins.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tier0/hurdle_adaptive_bins.py tests/tier0/test_hurdle_adaptive_bins.py
git commit -m "Add hurdle + value-respecting adaptive binning for null strata"
```

---

### Task 2: Feasibility fallback merge

**Files:**
- Modify: `src/tier0/hurdle_adaptive_bins.py` (append function)
- Test: `tests/tier0/test_hurdle_adaptive_bins.py` (append tests)

**Interfaces:**
- Consumes: `(pools, counts)` as produced by `src.tier0._pool_assembly.pools_from_bins` — `pools: list[np.ndarray]`, `counts: list[int]`, index-aligned, key-ascending order (Task 1 guarantees ascending bin ids).
- Produces: `merge_deficient_strata(pools, counts) -> tuple[list[np.ndarray], list[int], list[tuple[int, int]]]` — merged pools/counts (deficient strata absorbed) plus `merges` as `(from_idx, into_idx)` pairs in original indices. Raises `ValueError` if total pool < total count.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/tier0/test_hurdle_adaptive_bins.py
from src.tier0.hurdle_adaptive_bins import merge_deficient_strata


def _pools(*sizes):
    start = 0
    out = []
    for s in sizes:
        out.append(np.arange(start, start + s))
        start += s
    return out


def test_no_merge_when_feasible():
    pools, counts, merges = merge_deficient_strata(_pools(5, 5, 5), [2, 3, 1])
    assert merges == []
    assert [len(p) for p in pools] == [5, 5, 5]
    assert counts == [2, 3, 1]


def test_deficient_stratum_merges_into_lower_neighbor():
    # stratum 2 needs 4 but has 2 candidates
    pools, counts, merges = merge_deficient_strata(_pools(5, 5, 2), [1, 1, 4])
    assert merges == [(2, 1)]
    assert len(pools) == 2
    assert counts == [1, 5]
    assert len(pools[1]) == 7


def test_lowest_stratum_merges_upward():
    pools, counts, merges = merge_deficient_strata(_pools(1, 6), [3, 1])
    assert merges == [(0, 1)]
    assert counts == [4]
    assert len(pools[0]) == 7


def test_cascading_merge():
    # After (2 -> 1), stratum 1 holds pool 3+2=5 vs count 2+4=6: still short,
    # so it merges into 0.
    pools, counts, merges = merge_deficient_strata(_pools(9, 3, 2), [1, 2, 4])
    assert merges == [(2, 1), (1, 0)]
    assert counts == [7]
    assert len(pools[0]) == 14


def test_globally_infeasible_raises():
    with pytest.raises(ValueError):
        merge_deficient_strata(_pools(2, 2), [3, 3])


def test_zero_count_strata_untouched():
    pools, counts, merges = merge_deficient_strata(_pools(4, 0, 4), [2, 0, 2])
    assert merges == []
    assert counts == [2, 0, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_hurdle_adaptive_bins.py -q`
Expected: new tests FAIL with `ImportError: cannot import name 'merge_deficient_strata'`.

- [ ] **Step 3: Write the implementation**

```python
# append to src/tier0/hurdle_adaptive_bins.py


def merge_deficient_strata(
    pools: list[np.ndarray], counts: list[int]
) -> tuple[list[np.ndarray], list[int], list[tuple[int, int]]]:
    """Deterministic feasibility fallback (design doc: binning contract).

    A stratum whose candidate pool (post self-exclusion) is smaller than its
    real-gene count cannot be drawn from; it is merged into its lower-key
    neighbor (next-higher for the lowest stratum), repeatedly, until every
    stratum with count > 0 satisfies count <= pool size. Merge events are
    returned in original stratum indices so callers can log them -- part of
    the null's definition, applied identically to analytical and MC arms.
    A stratum with count == 0 is never deficient (nothing is drawn from it).
    """
    if sum(int(c) for c in counts) > sum(len(p) for p in pools):
        raise ValueError(
            "total real-gene count exceeds total candidate pool; "
            "no partition of these pools is feasible"
        )
    pools = [np.asarray(p) for p in pools]
    counts = [int(c) for c in counts]
    orig_idx = list(range(len(pools)))
    merges: list[tuple[int, int]] = []

    while True:
        deficient = next(
            (i for i, (p, c) in enumerate(zip(pools, counts)) if c > 0 and c > len(p)),
            None,
        )
        if deficient is None:
            return pools, counts, merges
        into = deficient - 1 if deficient > 0 else deficient + 1
        merges.append((orig_idx[deficient], orig_idx[into]))
        lo, hi = sorted((deficient, into))
        pools[lo] = np.concatenate([pools[lo], pools[hi]])
        counts[lo] += counts[hi]
        orig_idx[lo] = orig_idx[into]
        del pools[hi], counts[hi], orig_idx[hi]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_hurdle_adaptive_bins.py -q`
Expected: all PASS. If `test_cascading_merge` fails on the `orig_idx` bookkeeping, print `merges` and fix the index tracking, not the test.

- [ ] **Step 5: Commit**

```bash
git add src/tier0/hurdle_adaptive_bins.py tests/tier0/test_hurdle_adaptive_bins.py
git commit -m "Add deterministic deficient-stratum fallback merge"
```

---

### Task 3: Leave-target-out capacity

**Files:**
- Create: `src/tier0/capacity.py`
- Test: `tests/tier0/test_capacity.py`

**Interfaces:**
- Consumes: `src.dwpc_direct.HetMat` (only `compute_dwpc_matrix_csc(metapath, damping=0.5)` returning a raw-DWPC `scipy.sparse.csc_matrix` with gene rows) — injected, so tests use a stub.
- Produces:
  - `leave_target_out_capacity(matrix_csc, target_position: int) -> np.ndarray` (pure)
  - `CapacityProvider(hetmat, damping=0.5)` with `.capacity(metapath: str, target_position: int) -> np.ndarray`, caching one row-sum vector per metapath.

- [ ] **Step 1: Write the failing tests**

```python
# tests/tier0/test_capacity.py
import numpy as np
from scipy import sparse

from src.tier0.capacity import CapacityProvider, leave_target_out_capacity


def _matrix():
    # 4 genes x 3 targets, raw DWPC values
    dense = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 2.0, 0.0],
         [0.5, 0.0, 4.0],
         [3.0, 3.0, 3.0]]
    )
    return sparse.csc_matrix(dense)


def test_capacity_is_rowsum_minus_target_column():
    cap = leave_target_out_capacity(_matrix(), target_position=1)
    np.testing.assert_allclose(cap, [0.0, 1.0, 4.5, 6.0])


def test_zero_row_gene_has_zero_capacity_for_any_target():
    m = _matrix()
    for t in range(3):
        assert leave_target_out_capacity(m, t)[0] == 0.0


class _StubHetMat:
    def __init__(self):
        self.calls = []

    def compute_dwpc_matrix_csc(self, metapath, damping=0.5):
        self.calls.append(metapath)
        return _matrix()


def test_provider_caches_rowsums_per_metapath():
    hetmat = _StubHetMat()
    provider = CapacityProvider(hetmat)
    a = provider.capacity("GaDlA", target_position=0)
    b = provider.capacity("GaDlA", target_position=2)
    np.testing.assert_allclose(a, [0.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(b, [0.0, 3.0, 0.5, 6.0])
    assert hetmat.calls == ["GaDlA", "GaDlA"]  # matrix reloaded, rowsum cached is optional
    c = provider.capacity("GiGaD", target_position=0)
    assert len(c) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_capacity.py -q`
Expected: FAIL with `ModuleNotFoundError` for `src.tier0.capacity`.

- [ ] **Step 3: Write the implementation**

```python
# src/tier0/capacity.py
"""Leave-target-out metapath capacity: the stratification key for the
primary hurdle+adaptive strategy (design doc, strategy S1).

Capacity is computed on the RAW DWPC scale -- the per-entry arcsinh
transform used for observed scores is nonlinear, so summing transformed
entries would not be a walk-capacity quantity. Excluding the feature's own
target column keeps the key a generic "reach along this metapath"
covariate rather than the outcome; capacity 0 is an exact exchangeability
class (a zero row means zero DWPC to every target of the metapath).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def leave_target_out_capacity(matrix_csc: sparse.csc_matrix, target_position: int) -> np.ndarray:
    """Row sums of the raw DWPC matrix minus the target column's entries."""
    row_sums = np.asarray(matrix_csc.sum(axis=1)).ravel()
    target_col = np.asarray(matrix_csc[:, target_position].todense()).ravel()
    return row_sums - target_col


class CapacityProvider:
    """Per-metapath capacity vectors from a HetMat's cached DWPC matrices.

    The row-sum vector is cached per metapath (the expensive part is the
    matrix load); the per-feature target-column subtraction is done per
    call. ``hetmat`` must expose ``compute_dwpc_matrix_csc(metapath,
    damping)`` returning raw DWPC with gene rows -- the same source
    ``precompute_gene_feature_scores`` builds the observed scores from, so
    key and statistic describe the same matrix.
    """

    def __init__(self, hetmat, damping: float = 0.5):
        self._hetmat = hetmat
        self._damping = damping
        self._rowsums: dict[str, np.ndarray] = {}

    def capacity(self, metapath: str, target_position: int) -> np.ndarray:
        matrix = self._hetmat.compute_dwpc_matrix_csc(metapath, damping=self._damping)
        if metapath not in self._rowsums:
            self._rowsums[metapath] = np.asarray(matrix.sum(axis=1)).ravel()
        target_col = np.asarray(matrix[:, target_position].todense()).ravel()
        return self._rowsums[metapath] - target_col
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_capacity.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tier0/capacity.py tests/tier0/test_capacity.py
git commit -m "Add leave-target-out metapath capacity provider (raw DWPC scale)"
```

---

### Task 4: The two hurdle+adaptive strategies

**Files:**
- Modify: `src/tier0/metaedge_degree_pool_construction.py` (extract `_degree_for_first_hop`)
- Create: `src/tier0/hurdle_adaptive_pool_construction.py`
- Test: `tests/tier0/test_hurdle_adaptive_strategies.py`

**Interfaces:**
- Consumes: `hurdle_adaptive_bins`, `merge_deficient_strata` (Tasks 1–2); `CapacityProvider` (Task 3); `pools_from_bins(bin_of_row, real_row_idx, n_bins)` from `src.tier0._pool_assembly`; `_map_target_positions(hetmat, lv_targets)` from `src.lv_precompute`; `MetaedgeDegreePoolStrategy._resolve_first_hop` (existing).
- Produces: `CapacityHurdleAdaptiveStrategy(data_dir, min_stratum_size=50)` and `MetaedgeDegreeHurdleAdaptiveStrategy(data_dir, min_stratum_size=50)`, both callable with the harness `PoolFn` contract `(lv_id: str, feature_idx: int, substrate_dir: Path) -> (scores, pools, counts, observed)`, each exposing `.merge_log: list[dict]` (keys: `lv_id`, `feature_idx`, `from_stratum`, `into_stratum`). Test-override points: `_keys(self, metapath, lv_id, substrate_dir) -> np.ndarray` and `_hetmat_gene_ids(self) -> np.ndarray`.

- [ ] **Step 1: Extract `_degree_for_first_hop` in the existing metaedge module**

In `src/tier0/metaedge_degree_pool_construction.py`, split `_bins_for_first_hop` so degree extraction is reusable (behavior of the existing strategy unchanged):

```python
    def _degree_for_first_hop(self, metapath: str) -> np.ndarray:
        fname, axis = self._resolve_first_hop(metapath)
        cache_key = f"deg:{fname}:{axis}"
        if cache_key in self._bins_by_file:
            return self._bins_by_file[cache_key]
        matrix = sparse.load_npz(self.data_dir / "edges" / fname).tocsr()
        if axis == "row":
            degree = np.asarray(matrix.sum(axis=1)).flatten()
        else:
            degree = np.asarray(matrix.sum(axis=0)).flatten()
        self._bins_by_file[cache_key] = degree
        return degree

    def _bins_for_first_hop(self, metapath: str) -> np.ndarray:
        fname, axis = self._resolve_first_hop(metapath)
        cache_key = f"{fname}:{axis}"
        if cache_key in self._bins_by_file:
            return self._bins_by_file[cache_key]
        degree = self._degree_for_first_hop(metapath)
        bins = _assign_rank_bins(pd.Series(degree), N_BINS).to_numpy()
        self._bins_by_file[cache_key] = bins
        return bins
```

Run: `conda activate multi_dwpc && python -m pytest tests/tier0 -q`
Expected: existing suite still PASSES (pure refactor).

- [ ] **Step 2: Write the failing strategy tests**

```python
# tests/tier0/test_hurdle_adaptive_strategies.py
import numpy as np
import pandas as pd
import pytest

from src.tier0.hurdle_adaptive_pool_construction import (
    CapacityHurdleAdaptiveStrategy,
    MetaedgeDegreeHurdleAdaptiveStrategy,
)

N_GENES = 40


@pytest.fixture
def substrate(tmp_path):
    gene_ids = np.arange(101, 101 + N_GENES)
    np.save(tmp_path / "gene_ids.npy", gene_ids)
    rng = np.random.default_rng(0)
    scores = rng.gamma(1.0, 1.0, size=(N_GENES, 2)).astype(np.float32)
    np.save(tmp_path / "gene_feature_scores.npy", scores)
    pd.DataFrame(
        {"lv_id": ["LV1"] * 4, "gene_identifier": [101, 102, 110, 120]}
    ).to_csv(tmp_path / "lv_top_genes.csv", index=False)
    pd.DataFrame(
        {"lv_id": ["LV1", "LV1"], "feature_idx": [0, 1],
         "metapath": ["GaDlA", "GiGaD"], "length": [2, 2]}
    ).to_csv(tmp_path / "feature_manifest.csv", index=False)
    pd.DataFrame(
        {"lv_id": ["LV1", "LV1"], "feature_idx": [0, 1],
         "real_mean": [1.5, 2.5]}
    ).to_csv(tmp_path / "real_feature_scores.csv", index=False)
    pd.DataFrame(
        {"lv_id": ["LV1"], "target_id": ["UBERON:1"], "node_type": ["Anatomy"]}
    ).to_csv(tmp_path / "lv_targets.csv", index=False)
    return tmp_path


def _stubbed(strategy_cls, keys, gene_ids):
    s = strategy_cls(data_dir="data", min_stratum_size=5)
    s._keys = lambda metapath, lv_id, substrate_dir: np.asarray(keys, dtype=float)
    s._hetmat_gene_ids = lambda: np.asarray(gene_ids)
    return s


@pytest.mark.parametrize(
    "cls", [CapacityHurdleAdaptiveStrategy, MetaedgeDegreeHurdleAdaptiveStrategy]
)
def test_poolfn_contract_and_hurdle_exactness(cls, substrate):
    keys = np.zeros(N_GENES)
    keys[10:] = np.arange(1, N_GENES - 9)  # genes 0-9 are zero-key
    s = _stubbed(cls, keys, np.arange(101, 101 + N_GENES))
    scores, pools, counts, observed = s("LV1", 0, substrate)
    assert observed == pytest.approx(1.5)
    assert len(scores) == N_GENES
    assert sum(counts) == 4  # the LV's real genes
    # real genes excluded from their own pools
    real_rows = {0, 1, 9, 19}  # positions of 101,102,110,120
    for pool in pools:
        assert real_rows.isdisjoint(set(pool.tolist()))
    # hurdle stratum (bin 0) contains only zero-key rows
    zero_rows = set(range(10))
    assert set(pools[0].tolist()) <= zero_rows


@pytest.mark.parametrize(
    "cls", [CapacityHurdleAdaptiveStrategy, MetaedgeDegreeHurdleAdaptiveStrategy]
)
def test_fallback_merge_is_applied_and_logged(cls, substrate):
    # Put all four real genes on a unique maximal key so their stratum's
    # pool (after self-exclusion) is smaller than count=4.
    keys = np.ones(N_GENES)
    for pos in (0, 1, 9, 19):
        keys[pos] = 99.0
    keys[25] = 99.0  # one lone candidate at the same key
    s = _stubbed(cls, keys, np.arange(101, 101 + N_GENES))
    scores, pools, counts, observed = s("LV1", 0, substrate)
    assert all(c <= len(p) for p, c in zip(pools, counts) if c > 0)
    assert len(s.merge_log) == 1
    assert s.merge_log[0]["lv_id"] == "LV1"
    assert s.merge_log[0]["feature_idx"] == 0


def test_gene_order_mismatch_raises(substrate):
    s = _stubbed(
        CapacityHurdleAdaptiveStrategy,
        np.ones(N_GENES),
        np.arange(500, 500 + N_GENES),  # hetmat order != substrate order
    )
    with pytest.raises(ValueError, match="order"):
        s("LV1", 0, substrate)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_hurdle_adaptive_strategies.py -q`
Expected: FAIL with `ModuleNotFoundError` for `src.tier0.hurdle_adaptive_pool_construction`.

- [ ] **Step 4: Write the implementation**

```python
# src/tier0/hurdle_adaptive_pool_construction.py
"""Hurdle+adaptive stratified-SRSWOR pool construction (design doc S1/S2).

Both strategies satisfy the harness PoolFn contract
``(lv_id, feature_idx, substrate_dir) -> (scores, pools, counts, observed)``
and differ only in the stratification key: leave-target-out raw-DWPC
capacity (S1, primary) or first-hop metaedge degree (S2, ablation). The
partition is a function of (metapath, target, graph) only; the tested gene
set enters solely through self-exclusion in pools_from_bins and the
feasibility fallback, both shared with the MC arm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.dwpc_direct import HetMat
from src.lv_precompute import _map_target_positions
from src.tier0._pool_assembly import pools_from_bins
from src.tier0.capacity import CapacityProvider
from src.tier0.hurdle_adaptive_bins import hurdle_adaptive_bins, merge_deficient_strata
from src.tier0.metaedge_degree_pool_construction import MetaedgeDegreePoolStrategy

PIPELINE_DAMPING = 0.5


class _HurdleAdaptiveStrategyBase:
    def __init__(self, data_dir: Path, min_stratum_size: int = 50):
        self.data_dir = Path(data_dir)
        self.min_stratum_size = min_stratum_size
        self.merge_log: list[dict] = []
        self._hetmat = None
        self._gene_order_checked: set[str] = set()

    # -- test-override points -------------------------------------------
    def _keys(self, metapath: str, lv_id: str, substrate_dir: Path) -> np.ndarray:
        raise NotImplementedError

    def _hetmat_gene_ids(self) -> np.ndarray:
        return self._get_hetmat().get_nodes("Gene")["identifier"].to_numpy()

    # -------------------------------------------------------------------
    def _get_hetmat(self) -> HetMat:
        if self._hetmat is None:
            self._hetmat = HetMat(
                data_dir=self.data_dir, damping=PIPELINE_DAMPING,
                use_disk_cache=True, write_disk_cache=False,
            )
        return self._hetmat

    def _check_gene_order(self, gene_ids: np.ndarray, substrate_dir: Path) -> None:
        # Same guard, same reason as MetaedgeDegreePoolStrategy._check_gene_order:
        # keys are indexed by hetmat Gene position; a reordered substrate would
        # silently scramble every stratum.
        key = str(substrate_dir)
        if key in self._gene_order_checked:
            return
        if not np.array_equal(np.asarray(gene_ids), self._hetmat_gene_ids()):
            raise ValueError(
                f"gene_ids.npy order in {substrate_dir} does not match this "
                "hetmat's Gene node position order; stratification keys cannot "
                "be safely aligned to substrate rows."
            )
        self._gene_order_checked.add(key)

    def __call__(self, lv_id: str, feature_idx: int, substrate_dir: Path):
        substrate_dir = Path(substrate_dir)
        gene_ids = np.load(substrate_dir / "gene_ids.npy", allow_pickle=True)
        gene_scores = np.load(substrate_dir / "gene_feature_scores.npy")
        top_genes = pd.read_csv(substrate_dir / "lv_top_genes.csv")
        real_scores = pd.read_csv(substrate_dir / "real_feature_scores.csv")
        manifest = pd.read_csv(substrate_dir / "feature_manifest.csv")

        self._check_gene_order(gene_ids, substrate_dir)

        scores = gene_scores[:, feature_idx].astype(float)
        metapath = manifest[manifest["feature_idx"] == feature_idx].iloc[0]["metapath"]

        keys = self._keys(metapath, lv_id, substrate_dir)
        if len(keys) != len(gene_ids):
            raise ValueError(
                f"key vector length {len(keys)} != gene universe {len(gene_ids)}"
            )
        bins = hurdle_adaptive_bins(keys, self.min_stratum_size)

        this_lv = set(top_genes[top_genes["lv_id"] == lv_id]["gene_identifier"])
        real_row_idx = np.flatnonzero(np.isin(gene_ids, list(this_lv)))
        pools, counts = pools_from_bins(bins, real_row_idx, int(bins.max()) + 1)
        pools, counts, merges = merge_deficient_strata(pools, counts)
        for from_s, into_s in merges:
            self.merge_log.append(
                dict(lv_id=lv_id, feature_idx=int(feature_idx),
                     from_stratum=int(from_s), into_stratum=int(into_s))
            )

        row = real_scores[
            (real_scores["lv_id"] == lv_id)
            & (real_scores["feature_idx"] == feature_idx)
        ].iloc[0]
        return scores, pools, counts, float(row["real_mean"])


class CapacityHurdleAdaptiveStrategy(_HurdleAdaptiveStrategyBase):
    """S1: leave-target-out raw-DWPC capacity key."""

    def __init__(self, data_dir: Path, min_stratum_size: int = 50):
        super().__init__(data_dir, min_stratum_size)
        self._provider = None
        self._target_pos_by_lv: dict[str, dict[str, int]] = {}

    def _target_position(self, lv_id: str, substrate_dir: Path) -> int:
        key = str(substrate_dir)
        if key not in self._target_pos_by_lv:
            lv_targets = pd.read_csv(Path(substrate_dir) / "lv_targets.csv")
            self._target_pos_by_lv[key] = _map_target_positions(
                hetmat=self._get_hetmat(), lv_targets=lv_targets
            )
        return self._target_pos_by_lv[key][lv_id]

    def _keys(self, metapath: str, lv_id: str, substrate_dir: Path) -> np.ndarray:
        if self._provider is None:
            self._provider = CapacityProvider(self._get_hetmat(), PIPELINE_DAMPING)
        return self._provider.capacity(metapath, self._target_position(lv_id, substrate_dir))


class MetaedgeDegreeHurdleAdaptiveStrategy(_HurdleAdaptiveStrategyBase):
    """S2 (ablation): first-hop metaedge degree under the same binning."""

    def __init__(self, data_dir: Path, min_stratum_size: int = 50):
        super().__init__(data_dir, min_stratum_size)
        self._degree_source = MetaedgeDegreePoolStrategy(data_dir=Path(data_dir))

    def _keys(self, metapath: str, lv_id: str, substrate_dir: Path) -> np.ndarray:
        return self._degree_source._degree_for_first_hop(metapath)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_hurdle_adaptive_strategies.py tests/tier0 -q`
Expected: new tests PASS and the full tier0 suite stays green.

- [ ] **Step 6: Commit**

```bash
git add src/tier0/metaedge_degree_pool_construction.py src/tier0/hurdle_adaptive_pool_construction.py tests/tier0/test_hurdle_adaptive_strategies.py
git commit -m "Add capacity and metaedge-degree hurdle+adaptive strategies"
```

---

### Task 5: Wire strategies into the B-convergence sweep, persist per-row z

**Files:**
- Modify: `scripts/experiments/tier0_b_convergence.py`
- Test: `tests/tier0/test_tier0_orchestration.py` (extend)

**Interfaces:**
- Consumes: `sweep_b(substrate_dir, per_cell_max, b_values, dwpc_z_threshold, random_state, pool_fn)`; `_strategy_pool_fn(strategy)`; the two Task 4 strategies.
- Produces: `sweep_b(...) -> tuple[pd.DataFrame, pd.DataFrame]` — `(metrics, rows_at_max_b)`; CLI `--strategy` gains `capacity_hurdle_adaptive` and `metaedge_degree_hurdle_adaptive`; CLI gains `--output-dir` (default `output/tier0_b_convergence`); outputs per strategy: `curve_data.csv`, `rows_at_max_b.csv` (columns: `lv_id, feature_idx, length, n_active_strata, z_original, z_mc_highb, z_analytical, selected_*, mean_analytical, std_analytical, mean_mc_highb, std_mc_highb, b_mc_highb`), `merge_log.csv` (may be empty). The 17/36 lesson: pass rates must be recomputable from a persisted per-row artifact, never session archaeology.

- [ ] **Step 1: Extend the dispatch + return-shape tests**

In `tests/tier0/test_tier0_orchestration.py`, find the existing `--strategy` dispatch test (near `_strategy_pool_fn`) and add:

```python
def test_strategy_dispatch_knows_hurdle_adaptive_strategies():
    from scripts.experiments.tier0_b_convergence import _strategy_pool_fn
    from src.tier0.hurdle_adaptive_pool_construction import (
        CapacityHurdleAdaptiveStrategy,
        MetaedgeDegreeHurdleAdaptiveStrategy,
    )

    assert isinstance(
        _strategy_pool_fn("capacity_hurdle_adaptive"), CapacityHurdleAdaptiveStrategy
    )
    assert isinstance(
        _strategy_pool_fn("metaedge_degree_hurdle_adaptive"),
        MetaedgeDegreeHurdleAdaptiveStrategy,
    )


def test_sweep_b_returns_metrics_and_per_row_frame(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    from scripts.experiments import tier0_b_convergence as mod

    def stub_subsample(substrate_dir, per_cell_max, random_state):
        return pd.DataFrame(
            {"lv_id": ["LV1", "LV1"], "feature_idx": [0, 1], "length": [2, 3],
             "null_mean": [0.0, 0.0], "null_std": [1.0, 1.0]}
        )

    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)

    def stub_pool_fn(lv_id, feature_idx, substrate_dir):
        return scores, [np.arange(0, 100), np.arange(100, 200)], [3, 2], 0.4

    monkeypatch.setattr(mod, "select_stratified_subsample", stub_subsample)
    metrics, rows = mod.sweep_b(
        substrate_dir=tmp_path, per_cell_max=15, b_values=[10, 30],
        dwpc_z_threshold=1.65, random_state=0, pool_fn=stub_pool_fn,
    )
    assert {"z_analytical", "z_mc_highb", "lv_id", "feature_idx"} <= set(rows.columns)
    assert (rows["b_mc_highb"] == 30).all()  # per-row frame is at max B
    assert len(rows) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_tier0_orchestration.py -q`
Expected: FAIL — `_strategy_pool_fn` raises `ValueError: unknown strategy`, and `sweep_b` returns a single DataFrame.

- [ ] **Step 3: Implement**

In `scripts/experiments/tier0_b_convergence.py`:

1. `_strategy_pool_fn` gains:

```python
    if strategy == "capacity_hurdle_adaptive":
        from src.tier0.hurdle_adaptive_pool_construction import CapacityHurdleAdaptiveStrategy
        return CapacityHurdleAdaptiveStrategy(data_dir=Path("data"))
    if strategy == "metaedge_degree_hurdle_adaptive":
        from src.tier0.hurdle_adaptive_pool_construction import MetaedgeDegreeHurdleAdaptiveStrategy
        return MetaedgeDegreeHurdleAdaptiveStrategy(data_dir=Path("data"))
```

2. In `sweep_b`, capture the per-row frame when `b == max(b_values)`: inside the `for b in b_values` loop, after building `df_b`, add identifiers to each row dict (`lv_id=ctx["lv_id"], feature_idx=ctx["feature_idx"]` — add them into `row.update(...)` alongside `length`), and after the loop set `rows_at_max_b = df_b` when `b == max(b_values)`. Return `pd.DataFrame(records), rows_at_max_b`.

3. In `main()`: `--strategy` choices become the four names; add `parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)`; unpack `metrics, rows = sweep_b(...)`; write `rows.to_csv(out_dir / "rows_at_max_b.csv", index=False)`; and dump the merge log:

```python
    merge_log = getattr(pool_fn, "merge_log", [])
    pd.DataFrame(merge_log, columns=["lv_id", "feature_idx", "from_stratum", "into_stratum"]).to_csv(
        out_dir / "merge_log.csv", index=False
    )
```

- [ ] **Step 4: Run the full tier0 suite**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0 -q`
Expected: all PASS (the existing main-writes test may need its `sweep_b` unpacking updated — that is part of this task, update it).

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/tier0_b_convergence.py tests/tier0/test_tier0_orchestration.py
git commit -m "Dispatch hurdle+adaptive strategies; persist per-row z and merge log"
```

---

### Task 6: Calibration and planted-signal controls

**Files:**
- Modify: `src/tier0/montecarlo_reference.py` (extract `sample_null_sets`)
- Create: `scripts/experiments/tier0_capacity_controls.py`
- Test: `tests/tier0/test_capacity_controls.py`

**Interfaces:**
- Consumes: `analytical_null`, strategies via `_strategy_pool_fn`, `select_stratified_subsample`, `_mc_seed_for_row`.
- Produces:
  - `sample_null_sets(rng, pools, counts) -> np.ndarray` in `montecarlo_reference.py` (indices of one stratified draw; `montecarlo_reference` refactored to call it — behavior identical, verified by the existing suite plus seeds).
  - `plant_signal(scores, pools, counts, drawn_idx, fraction, rng) -> np.ndarray` and `run_controls(...)` in the script; outputs `negative_control.csv` (per-draw: `lv_id, feature_idx, z_draw`), `negative_summary.csv` (pooled tail fraction + 95% CI bounds), `positive_control.csv` (per row x fraction: `lv_id, feature_idx, fraction, z_planted, recovered`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/tier0/test_capacity_controls.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_capacity_controls.py -q`
Expected: FAIL with ImportError for `sample_null_sets` / the controls script.

- [ ] **Step 3: Implement**

In `src/tier0/montecarlo_reference.py`, extract the inner sampling loop:

```python
def sample_null_sets(rng: np.random.Generator, pools, counts) -> np.ndarray:
    """One stratified-SRSWOR draw: counts[r] indices from pools[r], concatenated."""
    k_total = int(sum(counts))
    sampled_idx = np.empty(k_total, dtype=np.int64)
    pos = 0
    for pool, k in zip(pools, counts):
        if k == 0:
            continue
        sampled_idx[pos : pos + k] = rng.choice(pool, size=k, replace=False)
        pos += k
    return sampled_idx
```

and make `montecarlo_reference`'s loop body `draws[i] = scores[sample_null_sets(rng, pools, counts)].mean()` — identical RNG call sequence, so seeded results are unchanged (the existing suite verifies).

Create `scripts/experiments/tier0_capacity_controls.py` (mirror the path/bootstrap header of `tier0_b_convergence.py`, including the two `sys.path.insert` lines and their reason):

```python
"""Negative (calibration) and positive (planted-signal) controls for the
hurdle+adaptive analytical null (design doc, validation experiments 2-3).

Usage:
    conda activate multi_dwpc
    python scripts/experiments/tier0_capacity_controls.py \\
        --substrate-dir "output/end_to_end_2026_4_23/lv_experiment (1)" \\
        --strategy capacity_hurdle_adaptive \\
        --output-dir output/tier0_capacity_hurdle
"""

from __future__ import annotations

import argparse
import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.tier0.analytical_null import analytical_null  # noqa: E402
from src.tier0.montecarlo_reference import sample_null_sets  # noqa: E402
from src.tier0.subsample import select_stratified_subsample  # noqa: E402
from scripts.experiments.tier0_b_convergence import _strategy_pool_fn  # noqa: E402
from scripts.experiments.tier0_offline_recompute import _mc_seed_for_row  # noqa: E402


def plant_signal(scores, pools, counts, drawn_idx, fraction, rng):
    """Replace the ceil(fraction*k_r) lowest-scoring drawn members per
    stratum with the highest-scoring pool candidates not already drawn --
    the stratum profile is preserved exactly while target-specific signal
    is injected (design doc, experiment 3)."""
    drawn = np.asarray(drawn_idx).copy()
    for pool, k in zip(pools, counts):
        if k == 0:
            continue
        in_stratum = drawn[np.isin(drawn, pool)]
        m = min(ceil(fraction * k), k)
        keep_out = in_stratum[np.argsort(scores[in_stratum])[:m]]
        candidates = pool[~np.isin(pool, drawn)]
        top_in = candidates[np.argsort(scores[candidates])[-m:]]
        replace_map = dict(zip(keep_out.tolist(), top_in.tolist()))
        drawn = np.array([replace_map.get(int(i), int(i)) for i in drawn])
    return drawn


def run_controls(substrate_dir, strategy, per_cell_max, n_draws, fractions,
                 dwpc_z_threshold, random_state, output_dir):
    pool_fn = _strategy_pool_fn(strategy)
    subsample = select_stratified_subsample(substrate_dir, per_cell_max, random_state)
    neg_rows, pos_rows = [], []
    for _, r in subsample.iterrows():
        lv_id, feature_idx = r["lv_id"], int(r["feature_idx"])
        scores, pools, counts, observed = pool_fn(lv_id, feature_idx, substrate_dir)
        res = analytical_null(scores, pools, counts, observed)
        if not np.isfinite(res.std):
            continue  # degenerate rows are reported by the sweep, not here
        rng = np.random.default_rng(_mc_seed_for_row(random_state, lv_id, feature_idx))
        for _ in range(n_draws):
            idx = sample_null_sets(rng, pools, counts)
            neg_rows.append(dict(
                lv_id=lv_id, feature_idx=feature_idx,
                z_draw=(scores[idx].mean() - res.mean) / res.std,
            ))
        base = sample_null_sets(rng, pools, counts)
        for f in fractions:
            planted = plant_signal(scores, pools, counts, base, f, rng)
            z = (scores[planted].mean() - res.mean) / res.std
            pos_rows.append(dict(
                lv_id=lv_id, feature_idx=feature_idx, fraction=f,
                z_planted=z, recovered=bool(z >= dwpc_z_threshold),
            ))

    output_dir.mkdir(parents=True, exist_ok=True)
    neg = pd.DataFrame(neg_rows)
    neg.to_csv(output_dir / f"negative_control_{strategy}.csv", index=False)
    k = int((neg["z_draw"] >= dwpc_z_threshold).sum())
    n = len(neg)
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95)
    pd.DataFrame([dict(strategy=strategy, n_draws=n, n_tail=k,
                       tail_fraction=k / n, ci_low=ci.low, ci_high=ci.high,
                       nominal=1 - 0.95)]).to_csv(
        output_dir / f"negative_summary_{strategy}.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(
        output_dir / f"positive_control_{strategy}.csv", index=False)
    print(f"{strategy}: tail {k}/{n} = {k/n:.4f}  95% CI [{ci.low:.4f}, {ci.high:.4f}]")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--substrate-dir", required=True, type=Path)
    p.add_argument("--strategy", required=True,
                   choices=["promiscuity", "metaedge_degree",
                            "capacity_hurdle_adaptive", "metaedge_degree_hurdle_adaptive"])
    p.add_argument("--per-cell-max", type=int, default=15)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--fractions", type=str, default="0.1,0.25,0.5")
    p.add_argument("--dwpc-z-threshold", type=float, default=1.65)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("output/tier0_capacity_hurdle"))
    a = p.parse_args()
    run_controls(a.substrate_dir, a.strategy, a.per_cell_max, a.n_draws,
                 [float(x) for x in a.fractions.split(",")],
                 a.dwpc_z_threshold, a.random_state, a.output_dir)


if __name__ == "__main__":
    main()
```

Note: the negative-control tail is checked against the nominal 0.05 by whether 0.05 falls inside the draw fraction's CI — the spec's criterion; the summary CSV carries both so the report can state it either way.

- [ ] **Step 4: Run tests**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_capacity_controls.py tests/tier0 -q`
Expected: all PASS, including the pre-existing `montecarlo_reference` tests (refactor must not change seeded draws).

- [ ] **Step 5: Commit**

```bash
git add src/tier0/montecarlo_reference.py scripts/experiments/tier0_capacity_controls.py tests/tier0/test_capacity_controls.py
git commit -m "Add calibration and planted-signal controls for hurdle+adaptive null"
```

---

### Task 7: Runtime benchmark script

**Files:**
- Create: `scripts/experiments/tier0_runtime_benchmark.py`
- Test: `tests/tier0/test_runtime_benchmark.py`

**Interfaces:**
- Consumes: `analytical_null`, `montecarlo_reference`, `_strategy_pool_fn`, `select_stratified_subsample`, `_mc_seed_for_row`.
- Produces: `benchmark_rows(substrate_dir, strategy, per_cell_max, b_values, random_state) -> pd.DataFrame` (columns `lv_id, feature_idx, t_analytical_ms, t_mc_ms, b, speedup`); CLI writes `runtime_benchmark_<strategy>.csv` under `--output-dir` and prints median speedup per B. Implements spec deliverable 5 — measured on the validation rows themselves.

- [ ] **Step 1: Write the failing test**

```python
# tests/tier0/test_runtime_benchmark.py
import numpy as np
import pandas as pd

from scripts.experiments.tier0_runtime_benchmark import benchmark_rows


def test_benchmark_rows_shape_and_positivity(monkeypatch, tmp_path):
    from scripts.experiments import tier0_runtime_benchmark as mod

    def stub_subsample(substrate_dir, per_cell_max, random_state):
        return pd.DataFrame(
            {"lv_id": ["LV1", "LV1"], "feature_idx": [0, 1],
             "length": [2, 3], "null_mean": [0.0, 0.0], "null_std": [1.0, 1.0]}
        )

    rng = np.random.default_rng(0)
    scores = rng.normal(size=300)

    def stub_pool_fn(lv_id, feature_idx, substrate_dir):
        return scores, [np.arange(0, 150), np.arange(150, 300)], [3, 2], 0.5

    monkeypatch.setattr(mod, "select_stratified_subsample", stub_subsample)
    monkeypatch.setattr(mod, "_strategy_pool_fn", lambda s: stub_pool_fn)

    df = benchmark_rows(tmp_path, "capacity_hurdle_adaptive", 15, [50, 100], 0)
    assert len(df) == 2 * 2  # rows x b_values
    assert (df["t_analytical_ms"] > 0).all()
    assert (df["t_mc_ms"] > 0).all()
    assert (df["speedup"] > 0).all()
    assert (df["t_poolfn_ms"] >= 0).all()  # spec diagnostic 4: pool-construction wall time
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_runtime_benchmark.py -q`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# scripts/experiments/tier0_runtime_benchmark.py
"""Per-row runtime of the analytical kernel vs the MC reference (design
doc, validation experiment 5 -- the speedup claim, measured on the
validation rows themselves rather than a synthetic setup).

Usage:
    conda activate multi_dwpc
    python scripts/experiments/tier0_runtime_benchmark.py \\
        --substrate-dir "output/end_to_end_2026_4_23/lv_experiment (1)" \\
        --strategy capacity_hurdle_adaptive \\
        --output-dir output/tier0_capacity_hurdle
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.tier0.analytical_null import analytical_null  # noqa: E402
from src.tier0.montecarlo_reference import montecarlo_reference  # noqa: E402
from src.tier0.subsample import select_stratified_subsample  # noqa: E402
from scripts.experiments.tier0_b_convergence import _strategy_pool_fn  # noqa: E402
from scripts.experiments.tier0_offline_recompute import _mc_seed_for_row  # noqa: E402

N_ANALYTICAL_REPS = 20  # sub-ms operation: repeat and average for a stable time


def benchmark_rows(substrate_dir, strategy, per_cell_max, b_values, random_state):
    pool_fn = _strategy_pool_fn(strategy)
    subsample = select_stratified_subsample(substrate_dir, per_cell_max, random_state)
    rows = []
    for _, r in subsample.iterrows():
        lv_id, feature_idx = r["lv_id"], int(r["feature_idx"])
        # Pool-construction time is per-row as experienced in a real sweep:
        # strategy-internal caches (capacity row-sums, degree vectors) make
        # later rows on the same metapath cheaper, and that amortization is
        # part of the honest number (spec diagnostic 4).
        t0 = time.perf_counter()
        scores, pools, counts, observed = pool_fn(lv_id, feature_idx, substrate_dir)
        t_pf = time.perf_counter() - t0
        try:
            t0 = time.perf_counter()
            for _ in range(N_ANALYTICAL_REPS):
                analytical_null(scores, pools, counts, observed)
            t_an = (time.perf_counter() - t0) / N_ANALYTICAL_REPS
        except ValueError:
            continue  # infeasible/degenerate rows are the sweep's business
        seed = _mc_seed_for_row(random_state, lv_id, feature_idx)
        for b in b_values:
            t0 = time.perf_counter()
            montecarlo_reference(scores, pools, counts, observed, b=b, random_state=seed)
            t_mc = time.perf_counter() - t0
            rows.append(dict(
                lv_id=lv_id, feature_idx=feature_idx, b=b,
                t_poolfn_ms=t_pf * 1e3,
                t_analytical_ms=t_an * 1e3, t_mc_ms=t_mc * 1e3,
                speedup=t_mc / t_an,
            ))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--substrate-dir", required=True, type=Path)
    p.add_argument("--strategy", default="capacity_hurdle_adaptive")
    p.add_argument("--per-cell-max", type=int, default=15)
    p.add_argument("--b-values", type=str, default="1000,10000")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("output/tier0_capacity_hurdle"))
    a = p.parse_args()
    df = benchmark_rows(a.substrate_dir, a.strategy, a.per_cell_max,
                        [int(x) for x in a.b_values.split(",")], a.random_state)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out = a.output_dir / f"runtime_benchmark_{a.strategy}.csv"
    df.to_csv(out, index=False)
    for b, grp in df.groupby("b"):
        print(f"B={b}: median MC {grp.t_mc_ms.median():.1f} ms/row, "
              f"analytical {grp.t_analytical_ms.median():.3f} ms/row, "
              f"median speedup {grp.speedup.median():.0f}x")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_runtime_benchmark.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/tier0_runtime_benchmark.py tests/tier0/test_runtime_benchmark.py
git commit -m "Add per-row analytical-vs-MC runtime benchmark script"
```

---

### Task 8: Four-strategy comparison report

**Files:**
- Modify: `scripts/experiments/tier0_b_comparison.py`
- Test: `tests/tier0/test_tier0_orchestration.py` (extend)

**Interfaces:**
- Consumes: per-strategy `curve_data.csv` and `rows_at_max_b.csv` (Task 5), `merge_log.csv`.
- Produces: `stack_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame` (generalized from the current two-strategy signature — the `strategy` tag comes from the dict key); new `pass_rate_table(rows_by_strategy: dict[str, pd.DataFrame], threshold: float) -> pd.DataFrame` with columns `strategy, n_valid, n_pass, pass_rate, n_near_threshold` (near = `|z_analytical - threshold| <= 0.5`); CLI `--strategies` (comma list, default all four) and `--convergence-dir`; `comparison.md` gains a "Pass rates and near-threshold density" section and a "Fallback merges" count per strategy.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/tier0/test_tier0_orchestration.py
def test_pass_rate_table_counts_pass_and_near_threshold():
    import pandas as pd
    from scripts.experiments.tier0_b_comparison import pass_rate_table

    rows = pd.DataFrame({"z_analytical": [0.2, 1.4, 1.7, 2.0, 25.0, float("nan")]})
    table = pass_rate_table({"s1": rows}, threshold=1.65)
    r = table.iloc[0]
    assert r["strategy"] == "s1"
    assert r["n_valid"] == 5
    assert r["n_pass"] == 3
    assert r["pass_rate"] == 3 / 5
    assert r["n_near_threshold"] == 3  # 1.4, 1.7, 2.0 within 0.5 of 1.65
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0/test_tier0_orchestration.py -q`
Expected: FAIL with ImportError for `pass_rate_table`.

- [ ] **Step 3: Implement**

In `scripts/experiments/tier0_b_comparison.py`:

```python
def pass_rate_table(rows_by_strategy: dict, threshold: float) -> "pd.DataFrame":
    """Per-strategy pass rate at the decision threshold plus near-threshold
    density -- the vacuousness diagnostic: a concordance result is only as
    informative as the number of rows near the boundary."""
    out = []
    for strategy, rows in rows_by_strategy.items():
        z = rows["z_analytical"].astype(float)
        valid = z.dropna()
        out.append(dict(
            strategy=strategy,
            n_valid=int(len(valid)),
            n_pass=int((valid >= threshold).sum()),
            pass_rate=float((valid >= threshold).mean()) if len(valid) else float("nan"),
            n_near_threshold=int(((valid - threshold).abs() <= 0.5).sum()),
        ))
    return pd.DataFrame(out)
```

Generalize `stack_frames` to take `dict[str, pd.DataFrame]` (tag each frame with its dict key as the `strategy` column, keep the existing column order constant), update `main()` to loop `--strategies` (default `promiscuity,metaedge_degree,capacity_hurdle_adaptive,metaedge_degree_hurdle_adaptive`), read each strategy's `curve_data.csv` / `rows_at_max_b.csv` / `merge_log.csv` from `--convergence-dir/<strategy>/`, render the pass-rate table and per-strategy merge counts as new markdown sections in `comparison.md`, and write `pass_rates.csv` beside it. Skip (with a printed warning) strategies whose directory is missing so partial runs still report.

- [ ] **Step 4: Run the tier0 suite**

Run: `conda activate multi_dwpc && python -m pytest tests/tier0 -q`
Expected: all PASS, including any existing `stack_frames` test updated to the dict signature (update it in this task).

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/tier0_b_comparison.py tests/tier0/test_tier0_orchestration.py
git commit -m "Generalize comparison report to N strategies; add pass-rate and near-threshold table"
```

---

### Task 9: Local real-data smoke (shape and sign before cluster time)

**Files:**
- No new files; runs the shipped scripts against the real substrate locally, on two metapaths whose DWPC matrices exist in the local cache.

- [ ] **Step 1: Run the S2 (degree) strategy sweep on a tiny B set**

```bash
conda activate multi_dwpc && python scripts/experiments/tier0_b_convergence.py \
  --substrate-dir "output/end_to_end_2026_4_23/lv_experiment (1)" \
  --strategy metaedge_degree_hurdle_adaptive \
  --b-values 10,30 --output-dir /tmp/tier0_smoke
```

Expected: completes; `/tmp/tier0_smoke/metaedge_degree_hurdle_adaptive/rows_at_max_b.csv` exists, has ~48 rows, `n_active_strata` median > 1, finite `z_analytical` for most rows.

- [ ] **Step 2: Run the S1 (capacity) strategy the same way**

Same command with `--strategy capacity_hurdle_adaptive`. Expected: rows whose metapaths are among the 6 missing from the local DWPC cache fail matrix load — confirm the error is a clean per-metapath failure, not a crash of the whole sweep. If the sweep aborts entirely, wrap the `pool_fn` call in `sweep_b`'s row loop with a try/except that records the row as skipped with a reason column (add that handling now — Alpine will hit the same rows if prewarm misses one).

- [ ] **Step 3: Sanity-check the smoke numbers**

```bash
conda activate multi_dwpc && python -c "
import pandas as pd
df = pd.read_csv('/tmp/tier0_smoke/capacity_hurdle_adaptive/rows_at_max_b.csv')
print(df[['z_analytical','z_mc_highb','n_active_strata']].describe())
print('sign agreement:', ((df.z_analytical>0)==(df.z_mc_highb>0)).mean())
"
```

Expected: z_analytical and z_mc_highb same order of magnitude and mostly same sign even at B=30; median `n_active_strata` > 1.

- [ ] **Step 4: Full local suite, then commit any smoke-motivated fixes**

```bash
conda activate multi_dwpc && python -m pytest tests/tier0 -q
git add <exact files changed by fixes, if any>
git commit -m "Harden sweep against per-row strategy failures found in local smoke"
```

(Skip the commit if the smoke motivated no changes.)

---

### Task 10: Alpine campaign — submit script, run, retrieve

**Files:**
- Create: `scripts/experiments/tier0_list_subsample_metapaths.py`
- Create: `hpc/submit_capacity_validation.sh`

**Interfaces:**
- Consumes: everything above, `scripts/prewarm_dwpc_cache.py --single-metapath`, the Slurm conventions of `hpc/submit_end_to_end.sh` (account `amc-general`, partition `acpu`, conda activation that bypasses Lmod: `source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh && conda activate multi_dwpc`).
- Produces: two chained jobs — prewarm (the subsample's metapaths) then validation (4 strategy sweeps + controls for S1 and S2 + benchmark + comparison) — writing to `output/tier0_capacity_hurdle/` on the Alpine checkout.

- [ ] **Step 1: Write the metapath-list helper script**

```python
# scripts/experiments/tier0_list_subsample_metapaths.py
"""Write the distinct metapaths used by the Tier 0 stratified subsample --
the exact set the Alpine prewarm stage must have DWPC matrices for.
feature_idx is globally unique across the manifest, so filtering on it
alone selects the subsample's features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.tier0.subsample import select_stratified_subsample  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--substrate-dir", required=True, type=Path)
    p.add_argument("--per-cell-max", type=int, default=15)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    sub = select_stratified_subsample(a.substrate_dir, a.per_cell_max, a.random_state)
    manifest = pd.read_csv(a.substrate_dir / "feature_manifest.csv")
    mps = sorted(
        manifest[manifest["feature_idx"].isin(sub["feature_idx"])]["metapath"].unique()
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(mps) + "\n")
    print(f"{len(mps)} metapaths -> {a.out}")


if __name__ == "__main__":
    main()
```

Sanity-check it locally (read-only; uses the real substrate):

```bash
conda activate multi_dwpc && python scripts/experiments/tier0_list_subsample_metapaths.py \
  --substrate-dir "output/end_to_end_2026_4_23/lv_experiment (1)" \
  --out /tmp/prewarm_metapaths.txt && wc -l /tmp/prewarm_metapaths.txt
```

Expected: ~48 lines.

- [ ] **Step 2: Write the submit script**

```bash
# hpc/submit_capacity_validation.sh
#!/bin/bash
# Two-stage Alpine campaign for the capacity-hurdle-adaptive-null
# validation (docs/tasks/capacity-hurdle-adaptive-null/). Stage 1 prewarms
# the DWPC cache for exactly the metapaths the stratified subsample uses;
# stage 2 (afterok) runs the four-strategy sweep, both controls, the
# runtime benchmark, and the comparison report.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
ACCOUNT="${ACCOUNT:-amc-general}"
PARTITION="${PARTITION:-acpu}"
QOS="${QOS:-normal}"
SUBSTRATE="${SUBSTRATE:-output/end_to_end_2026_4_23/lv_experiment (1)}"
OUT="output/tier0_capacity_hurdle"
ACTIVATE='source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh && conda activate multi_dwpc'
mkdir -p hpc/logs "$OUT"

# Stage 1: derive the subsample's metapath list, prewarm only those.
PREWARM_CMD="$ACTIVATE && python scripts/experiments/tier0_list_subsample_metapaths.py \
    --substrate-dir \"$SUBSTRATE\" --out $OUT/prewarm_metapaths.txt \
  && while read -r mp; do
       python scripts/prewarm_dwpc_cache.py --single-metapath \"\$mp\"
     done < $OUT/prewarm_metapaths.txt"

PREWARM_ID=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
  --job-name=cap-prewarm --time=04:00:00 --mem=32G --cpus-per-task=4 \
  --output=hpc/logs/cap-prewarm-%j.out \
  --wrap="bash -c 'cd \"$REPO_ROOT\" && set -e && $PREWARM_CMD'")
echo "prewarm job: $PREWARM_ID"

# Stage 2: the validation proper.
VALIDATE_CMD="$ACTIVATE && set -e
for s in promiscuity metaedge_degree capacity_hurdle_adaptive metaedge_degree_hurdle_adaptive; do
  python scripts/experiments/tier0_b_convergence.py \
    --substrate-dir \"$SUBSTRATE\" --strategy \"\$s\" --output-dir $OUT/b_convergence
done
for s in capacity_hurdle_adaptive metaedge_degree_hurdle_adaptive; do
  python scripts/experiments/tier0_capacity_controls.py \
    --substrate-dir \"$SUBSTRATE\" --strategy \"\$s\" --output-dir $OUT
done
python scripts/experiments/tier0_runtime_benchmark.py \
  --substrate-dir \"$SUBSTRATE\" --strategy capacity_hurdle_adaptive --output-dir $OUT
python scripts/experiments/tier0_b_comparison.py \
  --convergence-dir $OUT/b_convergence --output-dir $OUT"

VALIDATE_ID=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
  --job-name=cap-validate --time=02:00:00 --mem=32G --cpus-per-task=2 \
  --dependency=afterok:"$PREWARM_ID" \
  --output=hpc/logs/cap-validate-%j.out \
  --wrap="bash -c 'cd \"$REPO_ROOT\" && $VALIDATE_CMD'")
echo "validation job: $VALIDATE_ID (afterok:$PREWARM_ID)"
```

Check `hpc/submit_end_to_end.sh` for the QOS value it actually uses and match it (memory says `cpu-normal` may be the QOS name — copy whatever the working script uses, do not guess). Run `bash -n hpc/submit_capacity_validation.sh` locally.

- [ ] **Step 3: Commit and push the branch (standing grant: origin, this task's branch)**

```bash
conda activate multi_dwpc && python -m pytest tests/tier0 -q   # gate before push
git add scripts/experiments/tier0_list_subsample_metapaths.py hpc/submit_capacity_validation.sh
git commit -m "Add two-stage Alpine campaign script for capacity-null validation"
git push -u origin fix/random-null-stratified-srswor
```

- [ ] **Step 4: Verify Alpine inputs BEFORE submitting (read-only)**

```bash
ssh alpine 'cd /scratch/alpine/$USER/multi-dwpc && git fetch origin && git status && \
  ls "output/end_to_end_2026_4_23/lv_experiment (1)/feature_manifest.csv" 2>/dev/null || echo SUBSTRATE-MISSING; \
  ls data/edges/GpBP.sparse.npz 2>/dev/null || echo EDGES-MISSING'
```

Decision point: if `SUBSTRATE-MISSING` or `EDGES-MISSING`, STOP and put the staging question to the user (regenerate on Alpine vs approve a one-time transfer) — the design doc makes this its own approved step, and scp/rsync will prompt by design.

- [ ] **Step 5: Deploy and submit (git-only)**

```bash
ssh alpine 'cd /scratch/alpine/$USER/multi-dwpc && git checkout fix/random-null-stratified-srswor && git pull --ff-only && bash hpc/submit_capacity_validation.sh'
```

Then verify the dependency chain took:

```bash
ssh alpine 'squeue -u $USER -o "%.10i %.14j %.8T %.20E"'
```

Expected: cap-validate shows `afterok:<prewarm-id>` in its dependency column. A chain that silently dropped its dependency runs immediately and out of order — if the column is empty, scancel the validation job id (single id — within the standing grant) and resubmit.

- [ ] **Step 6: Monitor to completion, pull logs and outputs back**

Poll read-only (`ssh alpine 'squeue -u $USER'`; then `sacct -j <ids> --format=JobID,State,Elapsed,MaxRSS`). When both jobs COMPLETE, pull the campaign outputs and logs into the local worktree (rsync will prompt — expected, approve it):

```bash
rsync -av 'alpine:/scratch/alpine/$USER/multi-dwpc/output/tier0_capacity_hurdle/' \
  'output/tier0_capacity_hurdle/'
rsync -av --include='cap-*' --exclude='*' \
  'alpine:/scratch/alpine/$USER/multi-dwpc/hpc/logs/' 'output/tier0_capacity_hurdle/logs/'
```

- [ ] **Step 7: Commit the retrieved evidence**

```bash
git add output/tier0_capacity_hurdle
git commit -m "Add capacity-null validation campaign outputs and job logs"
```

(If `output/` is gitignored on this branch, add a negation for `output/tier0_capacity_hurdle/` in `.gitignore` in the same commit — evidence must land on the branch, per the design doc's out-of-scope note about invisible artifacts.)

---

### Task 11: Verification against the spec's success criteria

**Files:**
- Create: `docs/tasks/capacity-hurdle-adaptive-null/verification.md`
- Modify: `docs/tasks/capacity-hurdle-adaptive-null/design.md` (dated decision appended)

- [ ] **Step 1: Score every success criterion from the design doc**

Write `verification.md` with one section per criterion, each showing the exact command run and its output (not a paraphrase):

1. **Formula concordance** — from `rows_at_max_b.csv` per strategy: Spearman rho (analytical vs MC at B=10,000) per length bucket >= 0.99; max relative sd error <= 5%; Jaccard = 1.0 except rows within 2 MC standard errors of 1.65 (list any such flips explicitly).
2. **Calibration** — `negative_summary_*.csv`: is 0.05 inside the 95% CI? Also generate the spec's QQ plot from the per-draw file and save it beside the evidence:

```bash
conda activate multi_dwpc && python -c "
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
for s in ['capacity_hurdle_adaptive', 'metaedge_degree_hurdle_adaptive']:
    z = pd.read_csv(f'output/tier0_capacity_hurdle/negative_control_{s}.csv')['z_draw'].dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(z, dist='norm', plot=ax)
    ax.set_title(f'Negative-control z vs N(0,1): {s} (n={len(z)})')
    fig.savefig(f'output/tier0_capacity_hurdle/qq_negative_{s}.png', dpi=150, bbox_inches='tight')
    print(s, 'upper-tail 0.05-quantile z:', np.quantile(z, 0.95))
"
```
3. **Power** — `positive_control_*.csv`: does every S1 row recover at f = 0.5? Distribution of minimum recovered f, S1 vs S2. If S1 fails rows S2 passes: the fallback decision triggers — record it, S2 becomes the shipping strategy.
4. **Informativeness** — `pass_rates.csv`: `n_near_threshold` > 0 for the shipping strategy; the full pass-rate table across all four strategies (this regenerates and supersedes the unpersisted 17/36 figure).
5. **Runtime** — `runtime_benchmark_*.csv` medians and speedups.
6. **Fallback merges** — counts per strategy from `merge_log.csv`.

- [ ] **Step 2: Append the outcome decision to design.md**

Under `## Decisions`, add a dated entry naming the shipping strategy (S1 or the S2 fallback), the pass-rate consequence, and pointers to `verification.md` and the output files.

- [ ] **Step 3: Commit; the branch is now ready for the integration task**

```bash
git add docs/tasks/capacity-hurdle-adaptive-null/verification.md docs/tasks/capacity-hurdle-adaptive-null/design.md
git commit -m "Record capacity-null validation verdict against spec criteria"
```

Then stop: the follow-on work (query-path integration PR off `upstream/main`, docs PR) is out of this plan's scope by design.
