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
    compare_conditions,
    compute_cumulative_coverage,
    compute_metapath_complementarity,
    compute_rescue_table,
    rank_metapaths_from_results,
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

    # Summary per year
    print("\n--- Summary ---")
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
