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
