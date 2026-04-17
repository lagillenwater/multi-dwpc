#!/usr/bin/env python3
"""Pathway-level coverage analysis for LV experiments.

Compares multi-DWPC (all consensus-selected metapaths) against a
single-best-metapath baseline at the (LV, target_set) level.

Inputs
------
- lv_pair_dwpc_top_selected.csv : pair-level DWPCs for selected metapaths
- lv_top_genes.csv              : full top-gene set per LV with loadings
- lv_metapath_results.csv       : metapath-level stats (for ranking)

Outputs
-------
- lv_pathway_coverage.csv          : metrics under top-1 and all conditions
- lv_pathway_rescue.csv            : gene/target rescue rates
- lv_metapath_complementarity.csv  : per-metapath uniqueness fractions
- lv_cumulative_coverage.csv       : cumulative coverage by metapath rank
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
    classify_gene_signal,
    compare_conditions,
    compute_cumulative_coverage,
    compute_metapath_complementarity,
    compute_rescue_table,
    rank_metapaths_by_dwpc_mass,
    rank_metapaths_from_results,
    summarize_added_pair_signal,
    summarize_signal_categories,
)

LV_CONFIG = CoverageConfig(
    group_cols=["lv_id", "target_set_id"],
    gene_col="gene_identifier",
    target_col="target_id",
    dwpc_col="dwpc",
    metapath_col="metapath",
    weight_col="loading",
    label_cols=["target_set_label", "node_type"],
)


def _attach_labels(df: pd.DataFrame, pair_dwpc: pd.DataFrame) -> pd.DataFrame:
    """Carry target_set_label and node_type through to output."""
    labels = (
        pair_dwpc[["lv_id", "target_set_id", "target_set_label", "node_type"]]
        .drop_duplicates()
    )
    return df.merge(labels, on=["lv_id", "target_set_id"], how="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="output/lv_multidwpc",
        help="LV workspace directory containing input CSVs.",
    )
    parser.add_argument(
        "--pair-dwpc-path",
        default=None,
        help="Override path to lv_pair_dwpc_top_selected.csv.",
    )
    parser.add_argument(
        "--top-genes-path",
        default=None,
        help="Override path to lv_top_genes.csv.",
    )
    parser.add_argument(
        "--metapath-results-path",
        default=None,
        help="Override path to lv_metapath_results.csv (for metapath ranking).",
    )
    parser.add_argument(
        "--rank-source",
        default="dwpc_mass",
        choices=["dwpc_mass", "consensus_score", "min_d", "min_diff"],
        help="How to rank metapaths within each pathway group.",
    )
    parser.add_argument(
        "--coverage-output-dir",
        default=None,
        help="Output directory. Defaults to <output-dir>/pathway_coverage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    pair_path = Path(args.pair_dwpc_path) if args.pair_dwpc_path else output_dir / "lv_pair_dwpc_top_selected.csv"
    genes_path = Path(args.top_genes_path) if args.top_genes_path else output_dir / "lv_top_genes.csv"
    results_path = Path(args.metapath_results_path) if args.metapath_results_path else output_dir / "lv_metapath_results.csv"
    cov_dir = Path(args.coverage_output_dir) if args.coverage_output_dir else output_dir / "pathway_coverage"

    for p in [pair_path, genes_path]:
        if not p.exists():
            print(f"Error: required input not found: {p}", file=sys.stderr)
            sys.exit(1)

    cov_dir.mkdir(parents=True, exist_ok=True)

    pair_dwpc = pd.read_csv(pair_path)
    top_genes = pd.read_csv(genes_path)
    print(f"Loaded {len(pair_dwpc):,} pair rows, {len(top_genes):,} top genes")

    # Build gene universe: need group_cols + gene_col + weight_col
    # top_genes has lv_id but not target_set_id; broadcast across target_sets
    lv_targets = pair_dwpc[["lv_id", "target_set_id"]].drop_duplicates()
    gene_universe = lv_targets.merge(
        top_genes[["lv_id", "gene_identifier", "loading"]],
        on="lv_id",
        how="inner",
    )

    # Rank metapaths
    cfg = LV_CONFIG
    if args.rank_source == "dwpc_mass":
        pair_dwpc = rank_metapaths_by_dwpc_mass(pair_dwpc, cfg)
    else:
        if not results_path.exists():
            print(f"Error: --rank-source={args.rank_source} requires {results_path}", file=sys.stderr)
            sys.exit(1)
        results_df = pd.read_csv(results_path)
        ascending = args.rank_source in {"consensus_rank"}
        pair_dwpc = rank_metapaths_from_results(
            pair_dwpc, results_df, cfg,
            rank_col=args.rank_source, ascending=ascending,
        )

    n_max = int(pair_dwpc["metapath_rank"].max())
    print(f"Max metapath rank: {n_max}")

    # 1. Condition comparison (top-1, top-2, ..., all)
    ranks_to_compare = sorted({1, 2, n_max} & set(range(1, n_max + 1)))
    conditions = compare_conditions(
        pair_dwpc, cfg,
        gene_universe=gene_universe,
        max_ranks=ranks_to_compare,
    )
    conditions = _attach_labels(conditions, pair_dwpc)
    conditions.to_csv(cov_dir / "lv_pathway_coverage.csv", index=False)
    print(f"Saved pathway coverage: {cov_dir / 'lv_pathway_coverage.csv'}")

    # 2. Rescue table
    rescue = compute_rescue_table(pair_dwpc, cfg, gene_universe=gene_universe)
    rescue = _attach_labels(rescue, pair_dwpc)
    rescue.to_csv(cov_dir / "lv_pathway_rescue.csv", index=False)
    print(f"Saved rescue table: {cov_dir / 'lv_pathway_rescue.csv'}")

    # 3. Metapath complementarity
    complementarity = compute_metapath_complementarity(pair_dwpc, cfg)
    complementarity = _attach_labels(complementarity, pair_dwpc)
    complementarity.to_csv(cov_dir / "lv_metapath_complementarity.csv", index=False)
    print(f"Saved complementarity: {cov_dir / 'lv_metapath_complementarity.csv'}")

    # 4. Cumulative coverage curve
    cumulative = compute_cumulative_coverage(pair_dwpc, cfg, gene_universe=gene_universe)
    cumulative = _attach_labels(cumulative, pair_dwpc)
    cumulative.to_csv(cov_dir / "lv_cumulative_coverage.csv", index=False)
    print(f"Saved cumulative coverage: {cov_dir / 'lv_cumulative_coverage.csv'}")

    # 5. Group-level signal classification
    if results_path.exists():
        results_df = pd.read_csv(results_path) if "results_df" not in dir() else results_df
        classified = classify_gene_signal(
            pair_dwpc, results_df, cfg,
            null_mean_col="perm_null_mean",
            null_std_col="perm_null_std",
            group_target_col="target_set_id",
        )
        classified.to_csv(cov_dir / "lv_gene_signal_classified.csv", index=False)
        print(f"Saved signal classification: {cov_dir / 'lv_gene_signal_classified.csv'}")

        signal_per_group = summarize_signal_categories(classified, cfg)
        signal_per_group = _attach_labels(signal_per_group, pair_dwpc)
        signal_per_group.to_csv(cov_dir / "lv_signal_per_group.csv", index=False)
        print(f"Saved signal summary: {cov_dir / 'lv_signal_per_group.csv'}")

        signal_summary = summarize_added_pair_signal(classified, cfg)
        print("\n--- Group-level signal classification ---")
        print(f"  Pairs with DWPC rows: {signal_summary['n_added_pairs_with_dwpc_rows']:,}")
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

        has_nonzero = signal_per_group[signal_per_group["n_nonzero"] > 0]
        if not has_nonzero.empty:
            print(f"\n  Per-group distribution of group-only fraction "
                  f"({len(has_nonzero):,} groups with nonzero signal):")
            print(f"    median: {has_nonzero['frac_group_only'].median():.3f}, "
                  f"mean: {has_nonzero['frac_group_only'].mean():.3f}, "
                  f"25th: {has_nonzero['frac_group_only'].quantile(0.25):.3f}, "
                  f"75th: {has_nonzero['frac_group_only'].quantile(0.75):.3f}")
    else:
        print(f"\nSkipping signal classification (no metapath results at {results_path})")

    # Summary
    print("\n--- Coverage summary ---")
    rescue_summary = rescue[["gene_coverage_top1", "gene_coverage_multi", "gene_rescue_rate", "gene_coverage_gain"]]
    print(f"Median gene_coverage (top-1): {rescue_summary['gene_coverage_top1'].median():.3f}")
    print(f"Median gene_coverage (multi): {rescue_summary['gene_coverage_multi'].median():.3f}")
    print(f"Median gene_rescue_rate:      {rescue_summary['gene_rescue_rate'].median():.3f}")
    print(f"Median gene_coverage_gain:    {rescue_summary['gene_coverage_gain'].median():.3f}")
    if "target_rescue_rate" in rescue.columns:
        print(f"Median target_rescue_rate:    {rescue['target_rescue_rate'].median():.3f}")


if __name__ == "__main__":
    main()
