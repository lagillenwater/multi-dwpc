"""Tier 0 B-convergence comparison: combines the promiscuity and
metaedge-degree `curve_data.csv` outputs of tier0_b_convergence.py into a
single side-by-side table and markdown report.

Regenerates what a prior ad-hoc `python3 -c` heredoc produced (see
docs/superpowers/plans/2026-08-13-tier0-metaedge-degree-and-b-curves.md's
Task 3 Step 2) as committed, tested code, so `comparison.csv`/`comparison.md`
are reproducible from version control rather than a one-off shell command
that only ever existed in a plan document.

Usage:
    conda activate multi_dwpc
    python scripts/experiments/tier0_b_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("output/tier0_b_convergence")

# Finding 4 (final-review fix wave): n/n_excluded_nan are not the same row
# population across strategies -- a row's analytical null can be
# zero-variance (excluded) under one stratification scheme and not the
# other, since the two schemes assign different candidate pools to the same
# (lv_id, feature_idx) row. Surfaced explicitly so the side-by-side `n`
# columns in the tables below aren't misread as directly comparable counts
# over an identical row population.
ROW_COMPOSITION_CAVEAT = (
    "**Note on `n` / `n_excluded_nan`:** these columns are not directly "
    "comparable counts across strategies -- they can differ in *which* "
    "rows are excluded, not just how many. A given (lv_id, feature_idx) "
    "row's analytical null can be zero-variance (NaN z-score, excluded "
    "from `n`) under one stratification scheme and non-degenerate under "
    "the other, because promiscuity and metaedge-degree assign different "
    "candidate pools to the same row. Read `n` per strategy as \"how many "
    "rows fed that strategy's metrics\", not as evidence both strategies "
    "are scored over an identical row population.\n\n"
)

COMPARISON_COLUMNS = [
    "strategy", "length_bucket", "n", "n_excluded_nan",
    "median_active_strata", "min_active_strata",
    "spearman_rho_analytical_vs_mc_highb",
    "jaccard_selected_analytical_vs_mc_highb",
    "max_abs_relative_std_error",
]


def build_comparison(promiscuity_df: pd.DataFrame, metaedge_df: pd.DataFrame) -> pd.DataFrame:
    """Tag each curve_data.csv frame with its strategy and stack them into
    one long-format comparison table. Pure/testable: takes already-loaded
    DataFrames rather than reading files itself, so it can be exercised on
    small synthetic frames without touching disk.
    """
    promiscuity_df = promiscuity_df.copy()
    metaedge_df = metaedge_df.copy()
    promiscuity_df["strategy"] = "promiscuity"
    metaedge_df["strategy"] = "metaedge_degree"
    return pd.concat([promiscuity_df, metaedge_df], ignore_index=True)


def render_markdown(combined: pd.DataFrame) -> str:
    lines = ["# B-convergence: promiscuity vs metaedge-degree\n\n", ROW_COMPOSITION_CAVEAT]
    cols = [c for c in COMPARISON_COLUMNS if c in combined.columns]
    for b in sorted(combined["b"].unique()):
        lines.append(f"## B={b}\n\n")
        cell = combined[combined["b"] == b][cols]
        lines.append(cell.to_markdown(index=False) + "\n\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promiscuity-csv", type=Path, default=OUTPUT_DIR / "promiscuity" / "curve_data.csv"
    )
    parser.add_argument(
        "--metaedge-degree-csv", type=Path, default=OUTPUT_DIR / "metaedge_degree" / "curve_data.csv"
    )
    args = parser.parse_args()

    promiscuity_df = pd.read_csv(args.promiscuity_csv)
    metaedge_df = pd.read_csv(args.metaedge_degree_csv)
    combined = build_comparison(promiscuity_df, metaedge_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    (OUTPUT_DIR / "comparison.md").write_text(render_markdown(combined))

    print(f"Wrote {OUTPUT_DIR / 'comparison.csv'} ({len(combined)} rows)")
    print(f"Wrote {OUTPUT_DIR / 'comparison.md'}")


if __name__ == "__main__":
    main()
