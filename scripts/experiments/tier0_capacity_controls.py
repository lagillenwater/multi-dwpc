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
