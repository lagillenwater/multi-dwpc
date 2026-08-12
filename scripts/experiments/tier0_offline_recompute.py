"""Tier 0 offline recompute: does HetNetEX-MD's exact_resampling_moments
agree with a high-B Monte Carlo reference, on real LV score data?
See docs/superpowers/specs/2026-08-07-hetnetex-md-validation-design.md.

Usage:
    conda activate multi_dwpc
    python scripts/experiments/tier0_offline_recompute.py \\
        --substrate-dir "output/end_to_end_2026_4_23/lv_experiment (1)" \\
        --per-cell-max 15 --b 10000 --dwpc-z-threshold 1.65
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# src/dwpc_validation.py (pulled in transitively by src.tier0.subsample) uses a
# flat `from dwpc_direct import ...`, which only resolves if <repo_root>/src is
# itself on sys.path -- matches the sys.path setup used by every other script
# in this repo that depends on dwpc_validation/dwpc_direct (see e.g.
# scripts/api/validate_dwpc_concordance.py).
if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT / "src"))

from src.tier0.analytical_null import analytical_null  # noqa: E402
from src.tier0.montecarlo_reference import montecarlo_reference  # noqa: E402
from src.tier0.pool_construction import build_pools_and_counts  # noqa: E402
from src.tier0.subsample import select_stratified_subsample  # noqa: E402

OUTPUT_DIR = Path("output/tier0_offline_recompute")


def compute_concordance_row(
    z_original: float, z_mc_highb: float, z_analytical: float, dwpc_z_threshold: float
) -> dict:
    return {
        "z_original": z_original,
        "z_mc_highb": z_mc_highb,
        "z_analytical": z_analytical,
        "selected_original": bool(z_original >= dwpc_z_threshold),
        "selected_mc_highb": bool(z_mc_highb >= dwpc_z_threshold),
        "selected_analytical": bool(z_analytical >= dwpc_z_threshold),
    }


def join_and_score(df: pd.DataFrame) -> dict:
    """Score one arm-comparison table.

    Rows where any of z_original/z_mc_highb/z_analytical is NaN (degenerate
    null: zero-variance pool) are excluded up front from *every* metric below
    -- Spearman and Jaccard alike -- rather than letting NaN silently coerce
    to False in the `selected_*` booleans (which previously miscounted
    degenerate rows as "not selected" in Jaccard while Spearman already
    excluded them). `n`/`n_excluded_nan` report how many rows fed each metric.

    Spearman rho and Jaccard selection concordance are rank/threshold-based
    and can stay at 1.0 even when one arm's null variance is off by a large
    multiplicative factor (all 48 real subsampled rows here share a single
    promiscuity stratum, so z_analytical and z_mc_highb differ only by a
    near-constant scale plus MC noise -- ranks and threshold crossings barely
    move even under a big magnitude error). max_abs_relative_std_error and
    max_abs_mean_diff_in_mc_se are real magnitude-agreement checks that can
    actually fail where the rank-based metrics can't.
    """
    nan_mask = df[["z_original", "z_mc_highb", "z_analytical"]].isna().any(axis=1)
    valid = df[~nan_mask]
    n_excluded_nan = int(nan_mask.sum())

    rho_mc, _ = spearmanr(valid["z_analytical"], valid["z_mc_highb"], nan_policy="omit")
    rho_orig, _ = spearmanr(valid["z_analytical"], valid["z_original"], nan_policy="omit")

    def jaccard(a: pd.Series, b: pd.Series) -> float:
        a_set = set(valid.index[a])
        b_set = set(valid.index[b])
        union = a_set | b_set
        if not union:
            return 1.0
        return len(a_set & b_set) / len(union)

    if valid.empty:
        max_abs_relative_std_error = float("nan")
        max_abs_mean_diff_in_mc_se = float("nan")
    else:
        max_abs_relative_std_error = float(
            (valid["std_analytical"] / valid["std_mc_highb"] - 1.0).abs().max()
        )
        mc_se = valid["std_mc_highb"] / np.sqrt(valid["b_mc_highb"])
        max_abs_mean_diff_in_mc_se = float(
            ((valid["mean_analytical"] - valid["mean_mc_highb"]).abs() / mc_se).max()
        )

    return {
        "n": int(len(valid)),
        "n_excluded_nan": n_excluded_nan,
        "spearman_rho_analytical_vs_mc_highb": float(rho_mc),
        "spearman_rho_analytical_vs_original": float(rho_orig),
        "jaccard_selected_analytical_vs_mc_highb": jaccard(
            valid["selected_analytical"], valid["selected_mc_highb"]
        ),
        "jaccard_selected_analytical_vs_original": jaccard(
            valid["selected_analytical"], valid["selected_original"]
        ),
        "max_abs_relative_std_error": max_abs_relative_std_error,
        "max_abs_mean_diff_in_mc_se": max_abs_mean_diff_in_mc_se,
    }


def _mc_seed_for_row(random_state: int, lv_id: str, feature_idx: int) -> int:
    """Derive a deterministic, row-specific MC seed from (lv_id, feature_idx).

    Previously every row shared the same `random_state`, so `rng.choice`
    drew the same positional index pattern for all 48 rows -- MC error was
    perfectly correlated across the table instead of being 48 independent
    draws. Hashing (lv_id, feature_idx) keeps the seed reproducible and
    independent of row order/count (unlike e.g. `random_state + row_position`,
    which would shift every seed if the subsample changed size).
    """
    digest = hashlib.sha256(f"{lv_id}:{feature_idx}".encode()).hexdigest()
    return (random_state + int(digest[:8], 16)) % (2**32 - 1)


def run_tier0(
    substrate_dir: Path,
    per_cell_max: int,
    b: int,
    dwpc_z_threshold: float,
    random_state: int,
) -> pd.DataFrame:
    substrate_dir = Path(substrate_dir)
    subsample = select_stratified_subsample(substrate_dir, per_cell_max, random_state)

    rows = []
    for _, r in subsample.iterrows():
        lv_id = r["lv_id"]
        feature_idx = int(r["feature_idx"])
        scores, pools, counts, observed = build_pools_and_counts(
            lv_id, feature_idx, substrate_dir
        )
        exact = analytical_null(scores, pools, counts, observed)
        row_seed = _mc_seed_for_row(random_state, lv_id, feature_idx)
        mc = montecarlo_reference(scores, pools, counts, observed, b=b, random_state=row_seed)

        row = compute_concordance_row(
            z_original=(observed - r["null_mean"]) / r["null_std"] if r["null_std"] > 0 else np.nan,
            z_mc_highb=mc.z,
            z_analytical=exact.z,
            dwpc_z_threshold=dwpc_z_threshold,
        )
        row.update(
            lv_id=lv_id, feature_idx=feature_idx, metapath=r["metapath"],
            length=int(r["length"]), is_floor_pinned=bool(r["is_floor_pinned"]),
            mean_analytical=exact.mean, std_analytical=exact.std, var_analytical=exact.var,
            mean_mc_highb=mc.mean, std_mc_highb=mc.std, b_mc_highb=mc.b,
            n_active_strata=int(sum(1 for c in counts if c > 0)),
        )
        rows.append(row)

    return pd.DataFrame(rows)


def check_degenerate_inputs(
    substrate_dir: Path, concordance_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Flags the three degenerate-input categories the spec calls out as
    HetNetEX-MD's known weak spots, restricted to the `random`-null rows
    this plan is scoped to (see Global Constraints).
    """
    substrate_dir = Path(substrate_dir)
    feature_manifest = pd.read_csv(substrate_dir / "feature_manifest.csv")
    real_scores = pd.read_csv(substrate_dir / "real_feature_scores.csv")
    gene_scores = np.load(substrate_dir / "gene_feature_scores.npy")

    issues = []
    for _, feat in feature_manifest.iterrows():
        col = gene_scores[:, int(feat["feature_idx"])]
        if np.nanstd(col) == 0.0:
            issues.append(
                {"lv_id": feat["lv_id"], "feature_idx": int(feat["feature_idx"]),
                 "metapath": feat["metapath"], "issue": "zero_variance_scores"}
            )

        real_row = real_scores[
            (real_scores["lv_id"] == feat["lv_id"])
            & (real_scores["feature_idx"] == feat["feature_idx"])
        ]
        if not real_row.empty:
            n_real_genes = int(real_row.iloc[0]["n_real_genes"])
            if n_real_genes <= 1:
                issues.append(
                    {"lv_id": feat["lv_id"], "feature_idx": int(feat["feature_idx"]),
                     "metapath": feat["metapath"], "issue": "single_gene_stratum"}
                )

        n_with_paths = int((col > 0).sum())
        if n_with_paths < 5:
            issues.append(
                {"lv_id": feat["lv_id"], "feature_idx": int(feat["feature_idx"]),
                 "metapath": feat["metapath"], "issue": "few_genes_with_paths"}
            )

    # Fourth category: pool-level zero-variance null. This is not a
    # column-level property of feature_manifest.csv (a row's full score
    # column can have nonzero variance overall while still landing a
    # zero-variance analytical null, if every nonzero-scoring gene happens
    # to sit outside the sampled pool for that row's stratum) -- it can only
    # be known from the actual (scores, pools, counts) context run_tier0
    # already builds, so it's only checked for rows passed in via
    # `concordance_df` (run_tier0's output), not the full feature_manifest.
    if concordance_df is not None and "var_analytical" in concordance_df.columns:
        zero_var_null = concordance_df[concordance_df["var_analytical"] == 0.0]
        for _, r in zero_var_null.iterrows():
            issues.append(
                {"lv_id": r["lv_id"], "feature_idx": int(r["feature_idx"]),
                 "metapath": r["metapath"], "issue": "zero_variance_null"}
            )

    return pd.DataFrame(issues, columns=["lv_id", "feature_idx", "metapath", "issue"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate-dir", required=True, type=Path)
    parser.add_argument("--per-cell-max", type=int, default=15)
    parser.add_argument("--b", type=int, default=10_000)
    parser.add_argument("--dwpc-z-threshold", type=float, default=1.65)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    df = run_tier0(
        args.substrate_dir, args.per_cell_max, args.b, args.dwpc_z_threshold, args.random_state
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "concordance.csv", index=False)

    metrics = join_and_score(df)
    length_buckets = {"L=2": df[df["length"] == 2], "L>=3": df[df["length"] >= 3]}
    n_single_stratum = int((df["n_active_strata"] <= 1).sum()) if len(df) else 0

    with open(OUTPUT_DIR / "summary.md", "w") as f:
        f.write("# Tier 0 offline recompute -- summary\n\n")

        f.write("## Run parameters\n\n")
        f.write(f"- **substrate_dir**: {args.substrate_dir}\n")
        f.write(f"- **per_cell_max**: {args.per_cell_max}\n")
        f.write(f"- **b**: {args.b}\n")
        f.write(f"- **dwpc_z_threshold**: {args.dwpc_z_threshold}\n")
        f.write(f"- **random_state**: {args.random_state}\n")

        f.write("\n## Overall concordance\n\n")
        for k, v in metrics.items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## Concordance by metapath length\n\n")
        for label, cell in length_buckets.items():
            f.write(f"### {label}\n\n")
            if cell.empty:
                f.write("No rows in this bucket.\n\n")
                continue
            for k, v in join_and_score(cell).items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")

        f.write("## Caveats\n\n")
        f.write(
            f"- **Single-stratum real data:** of {len(df)} subsampled rows, "
            f"{n_single_stratum} use only a single active promiscuity stratum "
            "(`n_active_strata <= 1`) -- in this substrate, every real gene for "
            "all three LVs lands in the same promiscuity bin (bin 9 of 10), so "
            "`counts` is effectively `[0,...,0,34]`-shaped for every row here. "
            "The multi-stratum summation path in `exact_resampling_moments` is "
            "therefore exercised only by "
            "`tests/tier0/test_hetnetex_md_import.py`'s synthetic 2-stratum "
            "cases, not by this real-data comparison. A human making the "
            "Tier-1 go/no-go call should treat multi-stratum correctness as "
            "unverified against real data.\n"
        )

    degenerate = check_degenerate_inputs(args.substrate_dir, concordance_df=df)
    degenerate.to_csv(OUTPUT_DIR / "degenerate_inputs.csv", index=False)
    with open(OUTPUT_DIR / "summary.md", "a") as f:
        f.write("\n## Degenerate inputs\n\n")
        if degenerate.empty:
            f.write("None found.\n")
        else:
            f.write(degenerate["issue"].value_counts().to_markdown() + "\n")

    print(f"Wrote {OUTPUT_DIR / 'concordance.csv'} ({len(df)} rows)")
    print(f"Wrote {OUTPUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
