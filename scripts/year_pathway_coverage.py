#!/usr/bin/env python3
"""Pathway-level coverage analysis for year (GO-term) experiments.

Compares multi-DWPC (all supported metapaths) against a single-best-metapath
baseline at the (year, go_id) level.

Inputs
------
- dwpc_*_{year}_real.csv        : raw pair-level DWPCs (all metapaths, all genes)
- year_direct_go_term_support.csv : per-(year, go_id, metapath) support table

The support table determines which metapaths are "selected" per GO term.
Only supported metapaths are included in the analysis.

Outputs
-------
- year_pathway_coverage.csv          : metrics under top-1 and all conditions
- year_pathway_rescue.csv            : gene rescue rates
- year_metapath_complementarity.csv  : per-metapath uniqueness fractions
- year_cumulative_coverage.csv       : cumulative coverage by metapath rank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))

from src.pathway_coverage import (  # noqa: E402
    CoverageConfig,
    classify_added_pair_signal,
    classify_gene_signal,
    compare_conditions,
    compute_added_pair_coverage,
    compute_added_pair_coverage_per_group,
    compute_cumulative_coverage,
    compute_metapath_complementarity,
    compute_rescue_table,
    rank_metapaths_from_results,
    summarize_added_pair_signal,
    summarize_signal_categories,
)

YEAR_CONFIG = CoverageConfig(
    group_cols=["year", "go_id"],
    gene_col="entrez_gene_id",
    target_col=None,  # single target per group (the GO term itself)
    dwpc_col="dwpc",
    metapath_col="metapath",
    weight_col=None,  # no gene weights in year analysis
    label_cols=[],
)


def _load_year_dwpc(
    results_dir: Path,
    years: list[int],
    *,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Load raw real DWPC files for the requested years."""
    frames = []
    for year in years:
        pattern = f"dwpc_*_{year}_real.csv"
        matches = sorted(results_dir.glob(pattern))
        if not matches:
            print(f"Warning: no DWPC file matching {pattern} in {results_dir}", file=sys.stderr)
            continue
        for path in matches:
            chunks = []
            for chunk in pd.read_csv(path, chunksize=chunksize):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            df["year"] = year
            frames.append(df)
    if not frames:
        print(f"Error: no DWPC files found in {results_dir}", file=sys.stderr)
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def _filter_to_selected(
    pair_dwpc: pd.DataFrame,
    support_df: pd.DataFrame,
    *,
    selection_col: str = "selected_by_effective_n_all",
) -> pd.DataFrame:
    """Keep only rows whose (year, go_id, metapath) passes the selection filter.

    Parameters
    ----------
    selection_col
        Boolean column in *support_df* to filter on.  Defaults to
        ``selected_by_effective_n_all``; alternatives include ``supported``
        or ``selected_by_effective_n_supported_only``.
    """
    work = support_df.copy()
    work[selection_col] = (
        work[selection_col]
        .astype(str).str.strip().str.lower()
        .isin({"1", "true", "t", "yes"})
    )
    selected = work[work[selection_col]].copy()
    selected["year"] = selected["year"].astype(int)
    keep_keys = selected[["year", "go_id", "metapath"]].drop_duplicates()
    return pair_dwpc.merge(keep_keys, on=["year", "go_id", "metapath"], how="inner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="output/dwpc_direct/all_GO_positive_growth/results",
        help="Directory containing dwpc_*_{year}_real.csv files.",
    )
    parser.add_argument(
        "--support-path",
        default="output/year_direct_go_term_support_b5.csv",
        help="Path to year GO-term support table (e.g. year_direct_go_term_support_b5.csv).",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2016, 2024],
    )
    parser.add_argument(
        "--rank-col",
        default="consensus_score",
        help="Column in support table to rank metapaths by (within each year+go_id group).",
    )
    parser.add_argument(
        "--coverage-output-dir",
        default="output/year_pathway_coverage",
        help="Output directory for coverage results.",
    )
    parser.add_argument(
        "--added-pairs-path",
        default=None,
        help="Path to added-pair validation set (e.g. upd_go_bp_2024_added.csv). "
             "If provided, computes added-pair coverage metrics.",
    )
    parser.add_argument(
        "--selection-col",
        default="selected_by_effective_n_all",
        help="Boolean column in support table used to filter metapaths.",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    support_path = Path(args.support_path)
    cov_dir = Path(args.coverage_output_dir)

    if not support_path.exists():
        print(f"Error: support table not found: {support_path}", file=sys.stderr)
        sys.exit(1)

    cov_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw DWPC files for years {args.years} ...")
    pair_dwpc = _load_year_dwpc(results_dir, args.years, chunksize=args.chunksize)
    print(f"Loaded {len(pair_dwpc):,} raw pair rows")

    support_df = pd.read_csv(support_path)
    pair_dwpc = _filter_to_selected(
        pair_dwpc, support_df, selection_col=args.selection_col,
    )
    print(f"After filtering to selected metapaths ({args.selection_col}): {len(pair_dwpc):,} rows")

    if pair_dwpc.empty:
        print("No supported pair rows found. Exiting.", file=sys.stderr)
        sys.exit(1)

    cfg = YEAR_CONFIG

    # Rank metapaths within each (year, go_id) using the support table
    ascending = args.rank_col in {"consensus_rank", "fdr_sum"}
    pair_dwpc = rank_metapaths_from_results(
        pair_dwpc, support_df, cfg,
        rank_col=args.rank_col, ascending=ascending,
    )
    n_max = int(pair_dwpc["metapath_rank"].max())
    print(f"Max metapath rank: {n_max}")

    # Gene universe: all genes present in the pair file per (year, go_id)
    gene_universe = pair_dwpc[["year", "go_id", "entrez_gene_id"]].drop_duplicates()

    # 1. Condition comparison
    ranks_to_compare = sorted({1, 2, n_max} & set(range(1, n_max + 1)))
    conditions = compare_conditions(
        pair_dwpc, cfg,
        gene_universe=gene_universe,
        max_ranks=ranks_to_compare,
    )
    conditions.to_csv(cov_dir / "year_pathway_coverage.csv", index=False)
    print(f"Saved: {cov_dir / 'year_pathway_coverage.csv'}")

    # 2. Rescue table
    rescue = compute_rescue_table(pair_dwpc, cfg, gene_universe=gene_universe)
    rescue.to_csv(cov_dir / "year_pathway_rescue.csv", index=False)
    print(f"Saved: {cov_dir / 'year_pathway_rescue.csv'}")

    # 3. Metapath complementarity
    complementarity = compute_metapath_complementarity(pair_dwpc, cfg)
    complementarity.to_csv(cov_dir / "year_metapath_complementarity.csv", index=False)
    print(f"Saved: {cov_dir / 'year_metapath_complementarity.csv'}")

    # 4. Cumulative coverage curve
    cumulative = compute_cumulative_coverage(pair_dwpc, cfg, gene_universe=gene_universe)
    cumulative.to_csv(cov_dir / "year_cumulative_coverage.csv", index=False)
    print(f"Saved: {cov_dir / 'year_cumulative_coverage.csv'}")

    # 5. Added-pair coverage (if validation set provided)
    if args.added_pairs_path is not None:
        added_path = Path(args.added_pairs_path)
        if not added_path.exists():
            print(f"Warning: added-pairs file not found: {added_path}", file=sys.stderr)
        else:
            added_pairs_all = pd.read_csv(added_path)
            print(f"\nLoaded {len(added_pairs_all):,} added pairs from {added_path}")

            # Added pairs are year-independent (they represent 2024-only annotations).
            # Evaluate against the 2024 DWPC data only.
            dwpc_2024 = pair_dwpc[pair_dwpc["year"] == 2024].copy()
            if dwpc_2024.empty:
                print("Warning: no 2024 rows in pair DWPC; skipping added-pair analysis.", file=sys.stderr)
            else:
                # GO terms that have selected metapaths in the 2024 DWPC
                go_with_metapaths = set(dwpc_2024["go_id"].unique())

                # Conditional added pairs: restrict to GO terms with selected metapaths
                added_pairs_cond = added_pairs_all[
                    added_pairs_all["go_id"].isin(go_with_metapaths)
                ].copy()

                n_all = len(added_pairs_all[["go_id", "entrez_gene_id"]].drop_duplicates())
                n_cond = len(added_pairs_cond[["go_id", "entrez_gene_id"]].drop_duplicates())
                n_go_all = added_pairs_all["go_id"].nunique()
                n_go_cond = added_pairs_cond["go_id"].nunique()
                print(f"GO terms with selected metapaths: {len(go_with_metapaths):,}")
                print(f"Added pairs in those GO terms: {n_cond:,} / {n_all:,} "
                      f"({n_go_cond:,} / {n_go_all:,} GO terms)")

                # --- Global cumulative (all added pairs) ---
                added_cumulative_all = compute_added_pair_coverage(
                    dwpc_2024, added_pairs_all, cfg, added_group_col="go_id",
                )
                added_cumulative_all.to_csv(cov_dir / "year_added_pair_cumulative_global.csv", index=False)

                # --- Conditional cumulative (only GO terms with selected metapaths) ---
                added_cumulative_cond = compute_added_pair_coverage(
                    dwpc_2024, added_pairs_cond, cfg, added_group_col="go_id",
                )
                added_cumulative_cond.to_csv(cov_dir / "year_added_pair_cumulative_conditional.csv", index=False)
                print(f"Saved: {cov_dir / 'year_added_pair_cumulative_global.csv'}")
                print(f"Saved: {cov_dir / 'year_added_pair_cumulative_conditional.csv'}")

                # --- Per-GO-term rescue (conditional) ---
                added_per_go = compute_added_pair_coverage_per_group(
                    dwpc_2024, added_pairs_cond, cfg, added_group_col="go_id",
                )
                added_per_go.to_csv(cov_dir / "year_added_pair_per_go.csv", index=False)
                print(f"Saved: {cov_dir / 'year_added_pair_per_go.csv'}")

                # --- Print summaries ---
                print("\n--- Added-pair coverage: global (all GO terms) ---")
                for _, row in added_cumulative_all.iterrows():
                    print(f"  k={int(row['k']):>2}: {int(row['n_added_covered']):>6,} / "
                          f"{int(row['n_added_total']):,} "
                          f"({row['added_pair_coverage']:.4f})")

                print(f"\n--- Added-pair coverage: conditional "
                      f"({n_go_cond:,} GO terms with selected metapaths) ---")
                for _, row in added_cumulative_cond.iterrows():
                    print(f"  k={int(row['k']):>2}: {int(row['n_added_covered']):>6,} / "
                          f"{int(row['n_added_total']):,} "
                          f"({row['added_pair_coverage']:.4f})")

                n_go_with_rescue = int((added_per_go["n_rescued"] > 0).sum())
                has_added = added_per_go[added_per_go["n_added"] > 0]
                n_go_total = len(has_added)
                print(f"\n  GO terms with rescued added pairs: {n_go_with_rescue:,} / {n_go_total:,}")
                print(f"  Total added pairs rescued (top1->all): {int(added_per_go['n_rescued'].sum()):,}")

                # Coverage distribution across GO terms with added pairs
                print(f"\n  Added-pair coverage distribution (across {n_go_total:,} GO terms):")
                print(f"    top1  -- median: {has_added['added_coverage_top1'].median():.3f}, "
                      f"mean: {has_added['added_coverage_top1'].mean():.3f}, "
                      f"25th: {has_added['added_coverage_top1'].quantile(0.25):.3f}, "
                      f"75th: {has_added['added_coverage_top1'].quantile(0.75):.3f}")
                print(f"    multi -- median: {has_added['added_coverage_all'].median():.3f}, "
                      f"mean: {has_added['added_coverage_all'].mean():.3f}, "
                      f"25th: {has_added['added_coverage_all'].quantile(0.25):.3f}, "
                      f"75th: {has_added['added_coverage_all'].quantile(0.75):.3f}")
                pct_full = (has_added["added_coverage_all"] >= 1.0).sum()
                print(f"    GO terms reaching 100% coverage (multi): {int(pct_full):,} / {n_go_total:,} "
                      f"({pct_full / n_go_total:.1%})")

                top_rescued = added_per_go[added_per_go["n_rescued"] > 0].head(5)
                if not top_rescued.empty:
                    print("\n  Top GO terms by rescue count:")
                    for _, r in top_rescued.iterrows():
                        print(f"    {r['go_id']}: +{int(r['n_rescued'])} rescued "
                              f"({r['added_coverage_top1']:.2f} -> {r['added_coverage_all']:.2f})")

                # --- Group-level signal classification of added pairs ---
                # Classify each added pair's DWPC relative to the group null
                support_2024 = support_df[support_df["year"] == 2024].copy()

                classified_added = classify_added_pair_signal(
                    dwpc_2024, added_pairs_cond, support_2024, cfg,
                    null_mean_col="perm_null_mean",
                    null_std_col="perm_null_std",
                    added_group_col="go_id",
                )
                classified_added.to_csv(cov_dir / "year_added_pair_classified.csv", index=False)
                print(f"\nSaved: {cov_dir / 'year_added_pair_classified.csv'}")

                signal_summary = summarize_added_pair_signal(classified_added, cfg)

                print("\n--- Group-level signal classification (2024 added pairs) ---")
                print(f"  Added pairs with DWPC rows: {signal_summary['n_added_pairs_with_dwpc_rows']:,}")
                print(f"  Nonzero DWPC:    {signal_summary['n_nonzero']:,}")
                print(f"    strong  (DWPC > null_mean + std): {signal_summary['n_strong']:,} "
                      f"({signal_summary['frac_strong']:.1%})")
                print(f"    moderate (null_mean < DWPC <= null_mean + std): {signal_summary['n_moderate']:,} "
                      f"({signal_summary['frac_moderate']:.1%})")
                print(f"    weak    (0 < DWPC <= null_mean): {signal_summary['n_weak']:,} "
                      f"({signal_summary['frac_weak']:.1%})")
                print(f"  Zero DWPC:       {signal_summary['n_zero']:,}")
                print(f"\n  Fraction of nonzero signal attributable to group inference "
                      f"(weak + moderate): {signal_summary['frac_group_only']:.1%}")

                # Per-GO-term signal summary
                signal_per_go = summarize_signal_categories(classified_added, cfg)
                signal_per_go.to_csv(cov_dir / "year_added_pair_signal_per_go.csv", index=False)
                print(f"Saved: {cov_dir / 'year_added_pair_signal_per_go.csv'}")

                has_nonzero = signal_per_go[signal_per_go["n_nonzero"] > 0]
                if not has_nonzero.empty:
                    print(f"\n  Per-GO distribution of group-only fraction "
                          f"(weak+moderate / nonzero, {len(has_nonzero):,} GO terms):")
                    print(f"    median: {has_nonzero['frac_group_only'].median():.3f}, "
                          f"mean: {has_nonzero['frac_group_only'].mean():.3f}, "
                          f"25th: {has_nonzero['frac_group_only'].quantile(0.25):.3f}, "
                          f"75th: {has_nonzero['frac_group_only'].quantile(0.75):.3f}")

    # Summary per year
    print("\n--- Gene-level summary ---")
    for year in sorted(rescue["year"].unique()):
        yr = rescue[rescue["year"] == year]
        print(f"\nYear {year}:")
        print(f"  N pathway groups:          {len(yr):,}")
        print(f"  Median gene_coverage top1: {yr['gene_coverage_top1'].median():.3f}")
        print(f"  Median gene_coverage multi:{yr['gene_coverage_multi'].median():.3f}")
        print(f"  Median gene_rescue_rate:   {yr['gene_rescue_rate'].median():.3f}")
        print(f"  Median coverage_gain:      {yr['gene_coverage_gain'].median():.3f}")


if __name__ == "__main__":
    main()
