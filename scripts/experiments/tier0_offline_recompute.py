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
    rho_mc, _ = spearmanr(df["z_analytical"], df["z_mc_highb"])
    rho_orig, _ = spearmanr(df["z_analytical"], df["z_original"])

    def jaccard(a: pd.Series, b: pd.Series) -> float:
        a_set = set(df.index[a])
        b_set = set(df.index[b])
        union = a_set | b_set
        if not union:
            return 1.0
        return len(a_set & b_set) / len(union)

    return {
        "spearman_rho_analytical_vs_mc_highb": float(rho_mc),
        "spearman_rho_analytical_vs_original": float(rho_orig),
        "jaccard_selected_analytical_vs_mc_highb": jaccard(
            df["selected_analytical"], df["selected_mc_highb"]
        ),
        "jaccard_selected_analytical_vs_original": jaccard(
            df["selected_analytical"], df["selected_original"]
        ),
    }


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
        scores, pools, counts, observed = build_pools_and_counts(
            r["lv_id"], int(r["feature_idx"]), substrate_dir
        )
        exact = analytical_null(scores, pools, counts, observed)
        mc = montecarlo_reference(scores, pools, counts, observed, b=b, random_state=random_state)

        row = compute_concordance_row(
            z_original=(observed - r["null_mean"]) / r["null_std"] if r["null_std"] > 0 else np.nan,
            z_mc_highb=mc.z,
            z_analytical=exact.z,
            dwpc_z_threshold=dwpc_z_threshold,
        )
        row.update(
            lv_id=r["lv_id"], feature_idx=int(r["feature_idx"]), metapath=r["metapath"],
            length=int(r["length"]), is_floor_pinned=bool(r["is_floor_pinned"]),
        )
        rows.append(row)

    return pd.DataFrame(rows)


def check_degenerate_inputs(substrate_dir: Path) -> pd.DataFrame:
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
    for l2 in (True, False):
        cell = df[df["length"] == 2] if l2 else df[df["length"] >= 3]
        if not cell.empty:
            metrics[f"n_rows_L{'eq2' if l2 else 'ge3'}"] = len(cell)

    with open(OUTPUT_DIR / "summary.md", "w") as f:
        f.write("# Tier 0 offline recompute -- summary\n\n")
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

    degenerate = check_degenerate_inputs(args.substrate_dir)
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
