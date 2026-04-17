#!/usr/bin/env python3
"""Count how many added gene-GO pairs fall within the effective-number-selected
top pairs per (GO term, metapath) using consensus-selected metapaths.

Uses existing direct-DWPC pair values -- no path enumeration needed.

Inputs
------
- dwpc_*_2024_real.csv          : direct-DWPC pair values
- year_direct_go_term_support_b5.csv : consensus metapath selection
- upd_go_bp_2024_added.csv      : added (validation) pairs

Outputs
-------
- added_pair_selected_summary.csv : per-k cumulative summary
- added_pair_selected_per_go.csv  : per-GO breakdown at full k
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


def _effective_number(scores: np.ndarray) -> float:
    vals = scores[np.isfinite(scores)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    weights = vals / vals.sum()
    entropy = float(-(weights * np.log(weights)).sum())
    return float(np.exp(entropy))


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


def _select_pairs_by_effective_n(
    dwpc_df: pd.DataFrame,
    *,
    min_n: int = 1,
    max_n: int | None = None,
) -> pd.DataFrame:
    """For each (go_id, metapath), select top ceil(effective_n) pairs by DWPC."""
    group_cols = ["go_id", "metapath"]
    rows = []
    for (go_id, metapath), group in dwpc_df.groupby(group_cols, sort=False):
        scores = group["dwpc"].to_numpy(dtype=float)
        eff_n = _effective_number(scores)
        k = max(min_n, int(np.ceil(eff_n)))
        if max_n is not None:
            k = min(k, max_n)
        top = group.nlargest(k, "dwpc")
        top = top.copy()
        top["effective_n"] = float(eff_n)
        top["pair_rank"] = range(1, len(top) + 1)
        rows.append(top)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct-results-dir",
        default="output/dwpc_direct/all_GO_positive_growth/results",
    )
    parser.add_argument(
        "--added-pairs-path",
        default="output/intermediate/upd_go_bp_2024_added.csv",
    )
    parser.add_argument(
        "--support-path",
        default="output/year_direct_go_term_support_b5.csv",
    )
    parser.add_argument("--selection-col", default="selected_by_effective_n_all")
    parser.add_argument("--effective-min-n", type=int, default=1)
    parser.add_argument("--effective-max-n", type=int, default=None)
    parser.add_argument("--max-metapath-rank", type=int, default=5,
                        help="Only use top-k metapaths per GO term.")
    parser.add_argument("--output-dir", default="output/year_added_pair_selection")
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct_dir = Path(args.direct_results_dir)
    added_path = Path(args.added_pairs_path)
    support_path = Path(args.support_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load direct-DWPC
    pattern = "dwpc_*_2024_real.csv"
    matches = sorted(direct_dir.glob(pattern))
    if not matches:
        print(f"Error: no files matching {pattern} in {direct_dir}", file=sys.stderr)
        sys.exit(1)
    chunks = []
    for path in matches:
        for chunk in pd.read_csv(path, chunksize=args.chunksize):
            chunks.append(chunk)
    direct = pd.concat(chunks, ignore_index=True)
    print(f"Direct-DWPC rows: {len(direct):,}")

    # Load metapath selection from 2016 -- metapaths validated by 2016 gene sets,
    # tested against 2024-added genes that were NOT part of that selection
    selected_mp = _load_selected_metapaths(support_path, args.selection_col, year=2016)
    # Cap at --max-metapath-rank
    selected_mp = selected_mp[selected_mp["metapath_rank"] <= args.max_metapath_rank].copy()
    max_k = int(selected_mp["metapath_rank"].max())
    print(f"Selected metapaths (rank <= {args.max_metapath_rank}): {len(selected_mp):,} (max rank = {max_k})")

    # Load added pairs
    added = pd.read_csv(added_path)[["go_id", "entrez_gene_id"]].drop_duplicates()
    go_with_mp = set(selected_mp["go_id"].unique())
    added_cond = added[added["go_id"].isin(go_with_mp)].copy()
    added_cond["is_added"] = True
    print(f"Added pairs in GO terms with selected metapaths: {len(added_cond):,}")

    # Cumulative over k
    pair_key = ["go_id", "entrez_gene_id"]
    print(f"\n{'k':>3}  {'n_selected':>10}  {'n_unique_pairs':>14}  "
          f"{'n_added_in_sel':>14}  {'frac_added':>10}")

    cumulative_rows = []
    for k in range(1, max_k + 1):
        mp_at_k = selected_mp[selected_mp["metapath_rank"] <= k][["go_id", "metapath"]]
        direct_k = direct.merge(mp_at_k, on=["go_id", "metapath"], how="inner")

        # Select top effective-n pairs per (go_id, metapath)
        selected_pairs = _select_pairs_by_effective_n(
            direct_k,
            min_n=args.effective_min_n,
            max_n=args.effective_max_n,
        )
        if selected_pairs.empty:
            continue

        # Unique (go_id, gene) pairs across all metapaths at this k
        unique_pairs = selected_pairs[pair_key].drop_duplicates()
        n_unique = len(unique_pairs)

        # How many are added pairs?
        merged = unique_pairs.merge(added_cond[pair_key + ["is_added"]], on=pair_key, how="left")
        merged["is_added"] = merged["is_added"].fillna(False)
        n_added_in = int(merged["is_added"].sum())
        n_added_total = len(added_cond)
        frac = n_added_in / n_added_total if n_added_total > 0 else 0.0

        cumulative_rows.append({
            "k": k,
            "n_selected_rows": len(selected_pairs),
            "n_unique_pairs": n_unique,
            "n_added_in_selected": n_added_in,
            "n_added_total": n_added_total,
            "frac_added_covered": frac,
        })

        print(f"{k:>3}  {len(selected_pairs):>10,}  {n_unique:>14,}  "
              f"{n_added_in:>14,}  {frac:>10.1%}")

    cumulative_df = pd.DataFrame(cumulative_rows)
    cumulative_df.to_csv(out_dir / "added_pair_selected_summary.csv", index=False)
    print(f"\nSaved: {out_dir / 'added_pair_selected_summary.csv'}")

    # Per-GO breakdown at full k
    mp_all = selected_mp[["go_id", "metapath"]]
    direct_all = direct.merge(mp_all, on=["go_id", "metapath"], how="inner")
    selected_all = _select_pairs_by_effective_n(
        direct_all,
        min_n=args.effective_min_n,
        max_n=args.effective_max_n,
    )
    if not selected_all.empty:
        unique_all = selected_all[pair_key].drop_duplicates()
        merged_all = unique_all.merge(
            added_cond[pair_key + ["is_added"]], on=pair_key, how="left",
        )
        merged_all["is_added"] = merged_all["is_added"].fillna(False)

        per_go = merged_all.groupby("go_id", as_index=False).agg(
            n_selected=("entrez_gene_id", "size"),
            n_added_in_selected=("is_added", "sum"),
        )
        # Total added per GO
        added_per_go = added_cond.groupby("go_id", as_index=False).agg(
            n_added_total=("entrez_gene_id", "size"),
        )
        per_go = per_go.merge(added_per_go, on="go_id", how="left")
        per_go["n_added_total"] = per_go["n_added_total"].fillna(0).astype(int)
        per_go["frac_added_covered"] = np.where(
            per_go["n_added_total"] > 0,
            per_go["n_added_in_selected"] / per_go["n_added_total"],
            0.0,
        )
        per_go = per_go.sort_values("n_added_in_selected", ascending=False).reset_index(drop=True)
        per_go.to_csv(out_dir / "added_pair_selected_per_go.csv", index=False)
        print(f"Saved: {out_dir / 'added_pair_selected_per_go.csv'}")

        has_added = per_go[per_go["n_added_total"] > 0]
        print(f"\nPer-GO coverage of added pairs (effective-n selected, all metapaths):")
        print(f"  median: {has_added['frac_added_covered'].median():.3f}, "
              f"mean: {has_added['frac_added_covered'].mean():.3f}, "
              f"25th: {has_added['frac_added_covered'].quantile(0.25):.3f}, "
              f"75th: {has_added['frac_added_covered'].quantile(0.75):.3f}")


if __name__ == "__main__":
    main()
