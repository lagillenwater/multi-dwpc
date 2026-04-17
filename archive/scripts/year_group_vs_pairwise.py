#!/usr/bin/env python3
"""Compare group-level (Multi-DWPC) vs pairwise (API) significance for added pairs.

Uses two independent data sources:
- Multi-DWPC side: direct-DWPC pair values from the b=5 HPC run, filtered to
  group-selected metapaths.  This is what a Multi-DWPC user actually has.
- Pairwise side: Hetio API results with per-pair adjusted_p_value from the
  API's own degree-preserving null (millions of null samples per pair).

For each 2024-added gene-GO pair, classifies:
- both: individually significant AND group-reachable
- group_only: nonzero direct-DWPC through a group-validated metapath, but
  NOT individually significant by the API test
- pairwise_only: individually significant but zero in direct-DWPC
- neither: no signal from either approach

Runs cumulatively over k = 1, 2, ..., max selected metapaths per GO term.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.dwpc_direct import reverse_metapath_abbrev  # noqa: E402


def _load_neo4j_mappings(data_dir: Path) -> tuple[dict, dict]:
    gene_map = pd.read_csv(data_dir / "neo4j_gene_mapping.csv")
    bp_map = pd.read_csv(data_dir / "neo4j_bp_mapping.csv")
    neo4j_to_entrez = dict(zip(gene_map["neo4j_id"], gene_map["identifier"]))
    neo4j_to_go = dict(zip(bp_map["neo4j_id"], bp_map["identifier"]))
    return neo4j_to_entrez, neo4j_to_go


def _normalize_api_results(
    api_df: pd.DataFrame,
    neo4j_to_entrez: dict,
    neo4j_to_go: dict,
) -> pd.DataFrame:
    out = api_df.copy()
    out["go_id"] = out["neo4j_source_id"].map(neo4j_to_go)
    target_col = "neo4j_pseudo_target_id" if "neo4j_pseudo_target_id" in out.columns else "neo4j_target_id"
    out["entrez_gene_id"] = out[target_col].map(neo4j_to_entrez)
    out = out.dropna(subset=["go_id", "entrez_gene_id"]).copy()
    out["entrez_gene_id"] = out["entrez_gene_id"].astype(int)

    mp_col = "metapath_abbreviation" if "metapath_abbreviation" in out.columns else "metapath"
    out["metapath_api"] = out[mp_col].astype(str)
    mp_cache = {}
    def _reverse(mp):
        if mp not in mp_cache:
            mp_cache[mp] = reverse_metapath_abbrev(mp)
        return mp_cache[mp]
    out["metapath"] = out["metapath_api"].apply(_reverse)
    return out


def _load_selected_metapaths(
    support_path: Path,
    selection_col: str,
    year: int,
    rank_col: str = "consensus_score",
) -> pd.DataFrame:
    support = pd.read_csv(support_path)
    support[selection_col] = (
        support[selection_col]
        .astype(str).str.strip().str.lower()
        .isin({"1", "true", "t", "yes"})
    )
    selected = support[
        (support["year"] == year) & support[selection_col]
    ].copy()
    ascending = rank_col in {"consensus_rank", "fdr_sum"}
    selected["metapath_rank"] = (
        selected.groupby("go_id")[rank_col]
        .rank(method="first", ascending=ascending)
        .astype(int)
    )
    return selected[["go_id", "metapath", "metapath_rank"]].drop_duplicates()


def _load_direct_dwpc(
    results_dir: Path,
    year: int,
    *,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Load raw direct-DWPC pair results for a single year."""
    pattern = f"dwpc_*_{year}_real.csv"
    matches = sorted(results_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No DWPC file matching {pattern} in {results_dir}")
    frames = []
    for path in matches:
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunks.append(chunk)
        frames.append(pd.concat(chunks, ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def _classify_at_k(
    added_cond: pd.DataFrame,
    direct_selected: pd.DataFrame,
    api_selected: pd.DataFrame,
    selected_mp: pd.DataFrame,
    threshold: float,
    k: int,
) -> pd.DataFrame:
    """Classify added pairs using top-k metapaths per GO term."""
    pair_key = ["go_id", "entrez_gene_id"]
    mp_at_k = selected_mp[selected_mp["metapath_rank"] <= k][["go_id", "metapath"]]

    # Multi-DWPC side: direct-DWPC, best score across top-k metapaths
    direct_k = direct_selected.merge(mp_at_k, on=["go_id", "metapath"], how="inner")
    direct_best = (
        direct_k.groupby(pair_key, as_index=False)
        .agg(direct_best_dwpc=("dwpc", "max"))
    )

    # Pairwise side: API, best adj_p across top-k metapaths
    api_k = api_selected.merge(mp_at_k, on=["go_id", "metapath"], how="inner")
    api_best = (
        api_k.groupby(pair_key, as_index=False)
        .agg(api_best_adj_p=("adjusted_p_value", "min"))
    )

    result = added_cond[pair_key].copy()
    result = result.merge(direct_best, on=pair_key, how="left")
    result = result.merge(api_best, on=pair_key, how="left")
    result["direct_best_dwpc"] = result["direct_best_dwpc"].fillna(0.0)
    result["api_best_adj_p"] = result["api_best_adj_p"].fillna(1.0)

    result["group_reachable"] = result["direct_best_dwpc"] > 0
    result["pairwise_significant"] = result["api_best_adj_p"] < threshold
    result["group_only"] = result["group_reachable"] & ~result["pairwise_significant"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-results-path",
        default="output/dwpc_com/all_GO_positive_growth/results/res_all_GO_positive_growth_2024_real.csv",
        help="Hetio API results with per-pair adjusted_p_value.",
    )
    parser.add_argument(
        "--direct-results-dir",
        default="output/dwpc_direct/all_GO_positive_growth/results",
        help="Directory with direct-DWPC pair results from HPC b=5 run.",
    )
    parser.add_argument(
        "--added-pairs-path",
        default="output/intermediate/upd_go_bp_2024_added.csv",
    )
    parser.add_argument(
        "--support-path",
        default="output/year_direct_go_term_support_b5.csv",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--selection-col", default="selected_by_effective_n_all")
    parser.add_argument(
        "--p-threshold", type=float, default=0.05,
        help="Adjusted p-value threshold for pairwise significance.",
    )
    parser.add_argument("--max-metapath-rank", type=int, default=5,
                        help="Only use top-k metapaths per GO term.")
    parser.add_argument("--output-dir", default="output/year_group_vs_pairwise")
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_path = Path(args.api_results_path)
    direct_dir = Path(args.direct_results_dir)
    added_path = Path(args.added_pairs_path)
    support_path = Path(args.support_path)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    for p in [api_path, added_path, support_path]:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_key = ["go_id", "entrez_gene_id"]
    threshold = float(args.p_threshold)

    # --- Load data ---
    neo4j_to_entrez, neo4j_to_go = _load_neo4j_mappings(data_dir)

    print("Loading API results (pairwise significance) ...")
    chunks = []
    for chunk in pd.read_csv(api_path, chunksize=args.chunksize):
        chunks.append(chunk)
    api_raw = pd.concat(chunks, ignore_index=True)
    api = _normalize_api_results(api_raw, neo4j_to_entrez, neo4j_to_go)
    print(f"  API rows after normalization: {len(api):,}")

    print("Loading direct-DWPC results (group-level, b=5) ...")
    direct = _load_direct_dwpc(direct_dir, year=2024, chunksize=args.chunksize)
    print(f"  Direct-DWPC rows: {len(direct):,}")

    # Use metapaths selected from 2016 gene sets -- the 2024-added genes
    # were NOT part of the groups that drove this selection
    selected_mp = _load_selected_metapaths(support_path, args.selection_col, year=2016)
    selected_mp = selected_mp[selected_mp["metapath_rank"] <= args.max_metapath_rank].copy()
    max_k = int(selected_mp["metapath_rank"].max())
    print(f"  Group-selected metapaths (rank <= {args.max_metapath_rank}): {len(selected_mp):,} (max rank = {max_k})")

    # Pre-filter to selected metapaths
    direct_selected = direct.merge(
        selected_mp[["go_id", "metapath", "metapath_rank"]], on=["go_id", "metapath"], how="inner",
    )
    api_selected = api.merge(
        selected_mp[["go_id", "metapath"]], on=["go_id", "metapath"], how="inner",
    )
    print(f"  Direct rows in selected metapaths: {len(direct_selected):,}")
    print(f"  API rows in selected metapaths:    {len(api_selected):,}")

    # Added pairs in GO terms with selected metapaths
    added = pd.read_csv(added_path)[pair_key].drop_duplicates()
    go_with_mp = set(selected_mp["go_id"].unique())
    added_cond = added[added["go_id"].isin(go_with_mp)].copy()
    print(f"  Added pairs in GO terms with selected metapaths: {len(added_cond):,}")

    # Pre-filter direct and API to added pairs only (for speed)
    direct_added = direct_selected.merge(added_cond, on=pair_key, how="inner")
    api_added = api_selected.merge(added_cond, on=pair_key, how="inner")
    print(f"  Direct rows for added pairs: {len(direct_added):,}")
    print(f"  API rows for added pairs:    {len(api_added):,}")

    # --- Cumulative classification over k ---
    print(f"\n--- Group (direct-DWPC b=5) vs Pairwise (API adj_p < {threshold}) ---")
    print(f"{'k':>3}  {'n_group':>8}  {'n_pairwise':>10}  {'both':>8}  "
          f"{'group_only':>10}  {'pw_only':>8}  {'neither':>8}  "
          f"{'frac_group_only':>15}")

    cumulative_rows = []
    for k in range(1, max_k + 1):
        result = _classify_at_k(
            added_cond, direct_added, api_added, selected_mp, threshold, k,
        )
        n_total = len(result)
        n_group = int(result["group_reachable"].sum())
        n_pw = int(result["pairwise_significant"].sum())
        n_both = int((result["group_reachable"] & result["pairwise_significant"]).sum())
        n_group_only = int(result["group_only"].sum())
        n_pw_only = int((result["pairwise_significant"] & ~result["group_reachable"]).sum())
        n_neither = int((~result["group_reachable"] & ~result["pairwise_significant"]).sum())
        frac_go = n_group_only / n_group if n_group > 0 else 0.0

        cumulative_rows.append({
            "k": k,
            "n_total": n_total,
            "n_group_reachable": n_group,
            "n_pairwise_significant": n_pw,
            "n_both": n_both,
            "n_group_only": n_group_only,
            "n_pairwise_only": n_pw_only,
            "n_neither": n_neither,
            "frac_group_reachable": n_group / n_total if n_total > 0 else 0.0,
            "frac_pairwise_significant": n_pw / n_total if n_total > 0 else 0.0,
            "frac_group_only_of_reachable": frac_go,
        })

        print(f"{k:>3}  {n_group:>8,}  {n_pw:>10,}  {n_both:>8,}  "
              f"{n_group_only:>10,}  {n_pw_only:>8,}  {n_neither:>8,}  "
              f"{frac_go:>15.1%}")

    cumulative_df = pd.DataFrame(cumulative_rows)
    cumulative_df.to_csv(out_dir / "cumulative_group_vs_pairwise.csv", index=False)
    print(f"\nSaved: {out_dir / 'cumulative_group_vs_pairwise.csv'}")

    # --- Full classification at max k for per-GO breakdown ---
    result_all = _classify_at_k(
        added_cond, direct_added, api_added, selected_mp, threshold, max_k,
    )
    result_all.to_csv(out_dir / "added_pair_classification.csv", index=False)

    per_go = result_all.groupby("go_id", as_index=False).agg(
        n_added=("entrez_gene_id", "size"),
        n_pairwise_sig=("pairwise_significant", "sum"),
        n_group_reachable=("group_reachable", "sum"),
        n_group_only=("group_only", "sum"),
    )
    per_go["frac_pairwise"] = per_go["n_pairwise_sig"] / per_go["n_added"]
    per_go["frac_group_reachable"] = per_go["n_group_reachable"] / per_go["n_added"]
    per_go["frac_group_only"] = per_go["n_group_only"] / per_go["n_added"]
    per_go = per_go.sort_values("n_group_only", ascending=False).reset_index(drop=True)
    per_go.to_csv(out_dir / "per_go_classification.csv", index=False)
    print(f"Saved: {out_dir / 'per_go_classification.csv'}")

    # --- Summary at max k ---
    n_total = len(result_all)
    n_group = int(result_all["group_reachable"].sum())
    n_group_only = int(result_all["group_only"].sum())

    print(f"\n--- Summary (all selected metapaths, k={max_k}) ---")
    print(f"  Added pairs evaluated: {n_total:,}")
    print(f"  Group-reachable (direct-DWPC > 0): {n_group:,} ({n_group/n_total:.1%})")
    if n_group > 0:
        print(f"  Of those, only via group inference: {n_group_only:,} ({n_group_only/n_group:.1%})")

    has_go = per_go[per_go["n_group_only"] > 0]
    print(f"\n  GO terms where group inference adds coverage: "
          f"{len(has_go):,} / {len(per_go):,}")

    if not has_go.empty:
        print(f"\n  Top GO terms by group-only count:")
        for _, r in has_go.head(5).iterrows():
            print(f"    {r['go_id']}: {int(r['n_group_only'])} group-only / "
                  f"{int(r['n_added'])} added "
                  f"(pairwise: {r['frac_pairwise']:.0%}, "
                  f"group: {r['frac_group_reachable']:.0%})")


if __name__ == "__main__":
    main()
