"""Tier 0 B-convergence sweep: how do Spearman rho (analytical vs Monte
Carlo) and Jaccard selection concordance behave as B grows, at L=2 and
L>=3? Complements the single-B snapshot in tier0_offline_recompute.py's
summary.md.

Usage:
    conda activate multi_dwpc
    python scripts/experiments/tier0_b_convergence.py \\
        --substrate-dir "output/end_to_end_2026_4_23/lv_experiment (1)" \\
        --per-cell-max 15 --dwpc-z-threshold 1.65 --strategy promiscuity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT / "src"))
# Also needed (unlike tier0_offline_recompute.py) because this module is the
# first Tier 0 script to import another scripts/ module package-style (below):
# `python scripts/experiments/tier0_b_convergence.py ...` sets sys.path[0] to
# the script's own directory, not the repo root, so `scripts.experiments.*`
# doesn't resolve without this.
sys.path.insert(0, str(REPO_ROOT))

from src.tier0.analytical_null import analytical_null  # noqa: E402
from src.tier0.montecarlo_reference import montecarlo_reference  # noqa: E402
from src.tier0.pool_construction import build_pools_and_counts  # noqa: E402
from src.tier0.subsample import select_stratified_subsample  # noqa: E402
from scripts.experiments.tier0_offline_recompute import (  # noqa: E402
    compute_concordance_row,
    join_and_score,
    _mc_seed_for_row,
)

OUTPUT_DIR = Path("output/tier0_b_convergence")
DEFAULT_B_VALUES = [10, 30, 100, 300, 1000, 3000, 10000]

PoolFn = Callable[[str, int, Path], tuple]


def sweep_b(
    substrate_dir: Path,
    per_cell_max: int,
    b_values: list[int],
    dwpc_z_threshold: float,
    random_state: int,
    pool_fn: PoolFn,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    substrate_dir = Path(substrate_dir)
    subsample = select_stratified_subsample(substrate_dir, per_cell_max, random_state)

    # Per-row context is B-independent (pools, analytical result, original
    # arm's z) -- compute once, reuse across every B in the sweep, rather
    # than recomputing pools/analytical_null redundantly per B.
    row_context = []
    for _, r in subsample.iterrows():
        lv_id = r["lv_id"]
        feature_idx = int(r["feature_idx"])
        scores, pools, counts, observed = pool_fn(lv_id, feature_idx, substrate_dir)
        exact = analytical_null(scores, pools, counts, observed)
        z_original = (
            (observed - r["null_mean"]) / r["null_std"] if r["null_std"] > 0 else np.nan
        )
        row_context.append(
            dict(
                lv_id=lv_id, feature_idx=feature_idx, length=int(r["length"]),
                scores=scores, pools=pools, counts=counts, observed=observed,
                exact=exact, z_original=z_original,
                # B-independent, same formula run_tier0 uses -- carried
                # through so the sweep records evidence of how many strata
                # each row's pool_fn actually populated (the whole point of
                # the metaedge-degree strategy: more rows with >1 active
                # stratum than promiscuity's single-stratum collapse).
                n_active_strata=sum(1 for c in counts if c > 0),
            )
        )

    records = []
    for b in b_values:
        rows = []
        for ctx in row_context:
            row_seed = _mc_seed_for_row(random_state, ctx["lv_id"], ctx["feature_idx"])
            mc = montecarlo_reference(
                ctx["scores"], ctx["pools"], ctx["counts"], ctx["observed"],
                b=b, random_state=row_seed,
            )
            row = compute_concordance_row(
                z_original=ctx["z_original"], z_mc_highb=mc.z, z_analytical=ctx["exact"].z,
                dwpc_z_threshold=dwpc_z_threshold,
            )
            row.update(
                lv_id=ctx["lv_id"], feature_idx=ctx["feature_idx"],
                length=ctx["length"], n_active_strata=ctx["n_active_strata"],
                mean_analytical=ctx["exact"].mean, std_analytical=ctx["exact"].std,
                mean_mc_highb=mc.mean, std_mc_highb=mc.std, b_mc_highb=mc.b,
            )
            rows.append(row)

        df_b = pd.DataFrame(rows)
        if b == max(b_values):
            rows_at_max_b = df_b
        for label, cell in (("L=2", df_b[df_b["length"] == 2]), ("L>=3", df_b[df_b["length"] >= 3])):
            if cell.empty:
                continue
            metrics = join_and_score(cell)
            metrics.update(
                b=b, length_bucket=label,
                # Evidence that a stratification strategy actually spreads
                # rows across multiple strata (join_and_score's own return
                # contract is shared with the single-B report and
                # deliberately left untouched -- computed here instead).
                median_active_strata=float(cell["n_active_strata"].median()),
                min_active_strata=int(cell["n_active_strata"].min()),
            )
            records.append(metrics)

    return pd.DataFrame(records), rows_at_max_b


def _strategy_pool_fn(strategy: str) -> PoolFn:
    if strategy == "promiscuity":
        return build_pools_and_counts
    if strategy == "metaedge_degree":
        # Deferred import: this module doesn't exist until Task 2 lands, and
        # Task 1 must be runnable (with --strategy promiscuity, the default)
        # before Task 2 exists.
        from src.tier0.metaedge_degree_pool_construction import MetaedgeDegreePoolStrategy
        return MetaedgeDegreePoolStrategy(data_dir=Path("data"))
    if strategy == "capacity_hurdle_adaptive":
        # Deferred import: same rationale as metaedge_degree above.
        from src.tier0.hurdle_adaptive_pool_construction import CapacityHurdleAdaptiveStrategy
        return CapacityHurdleAdaptiveStrategy(data_dir=Path("data"))
    if strategy == "metaedge_degree_hurdle_adaptive":
        from src.tier0.hurdle_adaptive_pool_construction import MetaedgeDegreeHurdleAdaptiveStrategy
        return MetaedgeDegreeHurdleAdaptiveStrategy(data_dir=Path("data"))
    raise ValueError(f"unknown strategy: {strategy!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate-dir", required=True, type=Path)
    parser.add_argument("--per-cell-max", type=int, default=15)
    parser.add_argument("--b-values", type=str, default=",".join(str(b) for b in DEFAULT_B_VALUES))
    parser.add_argument("--dwpc-z-threshold", type=float, default=1.65)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--strategy",
        choices=[
            "promiscuity", "metaedge_degree",
            "capacity_hurdle_adaptive", "metaedge_degree_hurdle_adaptive",
        ],
        default="promiscuity",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    b_values = [int(x) for x in args.b_values.split(",")]
    pool_fn = _strategy_pool_fn(args.strategy)

    metrics, rows = sweep_b(
        args.substrate_dir, args.per_cell_max, b_values, args.dwpc_z_threshold,
        args.random_state, pool_fn,
    )

    out_dir = args.output_dir / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "curve_data.csv", index=False)
    rows.to_csv(out_dir / "rows_at_max_b.csv", index=False)
    merge_log = getattr(pool_fn, "merge_log", [])
    pd.DataFrame(merge_log, columns=["lv_id", "feature_idx", "from_stratum", "into_stratum"]).to_csv(
        out_dir / "merge_log.csv", index=False
    )
    print(f"Wrote {out_dir / 'curve_data.csv'} ({len(metrics)} rows)")


if __name__ == "__main__":
    main()
