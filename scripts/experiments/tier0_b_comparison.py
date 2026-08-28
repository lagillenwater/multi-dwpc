"""Tier 0 B-convergence comparison: combines the per-strategy `curve_data.csv`
outputs of tier0_b_convergence.py into a single side-by-side table and
markdown report, across any number of stratification strategies.

Regenerates what a prior ad-hoc `python3 -c` heredoc produced (see
docs/superpowers/plans/2026-08-13-tier0-metaedge-degree-and-b-curves.md's
Task 3 Step 2) as committed, tested code, so `comparison.csv`/`comparison.md`
are reproducible from version control rather than a one-off shell command
that only ever existed in a plan document.

Usage:
    conda activate multi_dwpc
    python scripts/experiments/tier0_b_comparison.py \\
        --convergence-dir output/tier0_b_convergence \\
        --strategies promiscuity,metaedge_degree,capacity_hurdle_adaptive,metaedge_degree_hurdle_adaptive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("output/tier0_b_convergence")

DEFAULT_STRATEGIES = [
    "promiscuity",
    "metaedge_degree",
    "capacity_hurdle_adaptive",
    "metaedge_degree_hurdle_adaptive",
]

# Finding 4 (final-review fix wave): n/n_excluded_nan are not the same row
# population across strategies -- a row's analytical null can be
# zero-variance (excluded) under one stratification scheme and not the
# other, since different schemes assign different candidate pools to the
# same (lv_id, feature_idx) row. Surfaced explicitly so the side-by-side `n`
# columns in the tables below aren't misread as directly comparable counts
# over an identical row population.
ROW_COMPOSITION_CAVEAT = (
    "**Note on `n` / `n_excluded_nan`:** these columns are not directly "
    "comparable counts across strategies -- they can differ in *which* "
    "rows are excluded, not just how many. A given (lv_id, feature_idx) "
    "row's analytical null can be zero-variance (NaN z-score, excluded "
    "from `n`) under one stratification scheme and non-degenerate under "
    "another, because each strategy assigns different candidate pools to "
    "the same row. Read `n` per strategy as \"how many rows fed that "
    "strategy's metrics\", not as evidence that strategies are scored "
    "over an identical row population.\n\n"
)

COMPARISON_COLUMNS = [
    "strategy", "length_bucket", "n", "n_excluded_nan",
    "median_active_strata", "min_active_strata",
    "spearman_rho_analytical_vs_mc_highb",
    "jaccard_selected_analytical_vs_mc_highb",
    "max_abs_relative_std_error",
]


def stack_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tag each strategy's curve_data.csv frame with its strategy (taken
    from the dict key) and stack them into one long-format comparison
    table. Pure/testable: takes already-loaded DataFrames rather than
    reading files itself, so it can be exercised on small synthetic frames
    without touching disk. Works for any number of strategies.
    """
    tagged = []
    for strategy, df in frames.items():
        df = df.copy()
        df["strategy"] = strategy
        tagged.append(df)
    return pd.concat(tagged, ignore_index=True)


def pass_rate_table(rows_by_strategy: dict, threshold: float) -> pd.DataFrame:
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


def render_markdown(
    combined: pd.DataFrame,
    pass_rates: pd.DataFrame | None = None,
    merge_counts: dict[str, int] | None = None,
) -> str:
    strategies = list(pd.unique(combined["strategy"]))
    lines = [f"# B-convergence: {' vs '.join(strategies)}\n\n", ROW_COMPOSITION_CAVEAT]
    cols = [c for c in COMPARISON_COLUMNS if c in combined.columns]
    for b in sorted(combined["b"].unique()):
        lines.append(f"## B={b}\n\n")
        cell = combined[combined["b"] == b][cols]
        lines.append(cell.to_markdown(index=False) + "\n\n")

    if pass_rates is not None:
        lines.append("## Pass rates and near-threshold density\n\n")
        lines.append(pass_rates.to_markdown(index=False) + "\n\n")

    if merge_counts is not None:
        lines.append("## Fallback merges\n\n")
        for strategy, count in merge_counts.items():
            lines.append(f"- **{strategy}**: {count} fallback merge(s)\n")
        lines.append("\n")

    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategies", type=str, default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated list of strategy names to compare.",
    )
    parser.add_argument("--convergence-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dwpc-z-threshold", type=float, default=1.65)
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    curve_frames: dict[str, pd.DataFrame] = {}
    rows_frames: dict[str, pd.DataFrame] = {}
    merge_counts: dict[str, int] = {}
    for strategy in strategies:
        strat_dir = args.convergence_dir / strategy
        if not strat_dir.exists():
            print(f"Warning: skipping {strategy!r} -- directory not found: {strat_dir}")
            continue
        curve_frames[strategy] = pd.read_csv(strat_dir / "curve_data.csv")
        rows_frames[strategy] = pd.read_csv(strat_dir / "rows_at_max_b.csv")
        merge_counts[strategy] = len(pd.read_csv(strat_dir / "merge_log.csv"))

    if not curve_frames:
        raise SystemExit(f"No strategy directories found under {args.convergence_dir}")

    combined = stack_frames(curve_frames)
    pass_rates = pass_rate_table(rows_frames, args.dwpc_z_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "comparison.csv", index=False)
    pass_rates.to_csv(args.output_dir / "pass_rates.csv", index=False)
    (args.output_dir / "comparison.md").write_text(
        render_markdown(combined, pass_rates=pass_rates, merge_counts=merge_counts)
    )

    print(f"Wrote {args.output_dir / 'comparison.csv'} ({len(combined)} rows)")
    print(f"Wrote {args.output_dir / 'pass_rates.csv'}")
    print(f"Wrote {args.output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
