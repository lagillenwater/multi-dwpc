# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
"""
Select supported GO-term/metapath combinations and extract top gene-BP pairs.

Uses direct DWPC outputs:
  output/dwpc_direct/all_GO_positive_growth/results/dwpc_*_real.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_name_map(path: Path) -> dict:
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["identifier"], df["name"]))


def load_support_table(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    support_df = pd.read_csv(path).copy()
    required = {"year", "metapath"}
    missing = required - set(support_df.columns)
    if missing:
        raise ValueError(f"Support table missing required columns: {sorted(missing)}")
    support_df["year"] = support_df["year"].astype(int)
    support_df["metapath"] = support_df["metapath"].astype(str)
    if "supported" in support_df.columns:
        support_df["supported"] = (
            support_df["supported"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "t", "yes"})
        )
    for col in [
        "selected_by_effective_n_all",
        "selected_by_effective_n_supported_only",
    ]:
        if col in support_df.columns:
            support_df[col] = (
                support_df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"1", "true", "t", "yes"})
            )
    return support_df


def effective_number(scores: pd.Series) -> float:
    vals = scores.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    weights = vals / vals.sum()
    entropy = float(-(weights * np.log(weights)).sum())
    return float(np.exp(entropy))


def effective_k(scores: pd.Series, *, min_n: int = 1, max_n: int | None = None) -> int:
    k = int(np.ceil(effective_number(scores)))
    k = max(int(min_n), k)
    if max_n is not None:
        k = min(k, int(max_n))
    return k


def compute_mean_dwpc(path: Path, chunksize: int = 200_000) -> pd.DataFrame:
    agg: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    usecols = ["metapath", "go_id", "dwpc"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        grouped = chunk.groupby(["metapath", "go_id"])["dwpc"].agg(["sum", "count"])
        for (metapath, go_id), row in grouped.iterrows():
            acc = agg[(metapath, go_id)]
            acc[0] += float(row["sum"])
            acc[1] += int(row["count"])

    records = []
    for (metapath, go_id), (s, c) in agg.items():
        records.append(
            {"metapath": metapath, "go_id": go_id, "mean_dwpc": s / c if c else np.nan}
        )
    return pd.DataFrame(records)


def select_supported_go_metapaths(
    support_df: pd.DataFrame | None,
    *,
    year: int,
    support_only: bool,
    sort_metric: str,
    top_n: int,
    top_quantile: float | None,
    top_z: float | None,
    use_effective_number: bool,
    effective_min_n: int,
    effective_max_n: int | None,
) -> pd.DataFrame | None:
    if support_df is None:
        return None
    subset = support_df[support_df["year"].astype(int) == int(year)].copy()
    if subset.empty:
        return subset
    if support_only and "supported" in subset.columns:
        subset = subset[subset["supported"] == True].copy()  # noqa: E712
    if subset.empty:
        return subset
    if sort_metric not in subset.columns:
        raise ValueError(f"Requested support sort metric '{sort_metric}' not found in support table")
    if "go_id" not in subset.columns:
        raise ValueError("GO-term support table is required for GO-centric subgraph extraction")

    ascending_metrics = {"consensus_rank", "rank_perm", "rank_rand", "fdr_sum", "p_perm_fdr", "p_rand_fdr"}
    sort_ascending = sort_metric in ascending_metrics
    rows = []
    for go_id, group in subset.groupby("go_id", sort=True):
        secondary_metric = "consensus_score" if "consensus_score" in group.columns else "min_diff"
        group = group.sort_values(
            [sort_metric, secondary_metric, "metapath"],
            ascending=[sort_ascending, False, True],
        ).reset_index(drop=True)
        threshold = np.nan
        selection = None
        if use_effective_number:
            if sort_metric == "consensus_rank":
                effective_scores = 1.0 / group[sort_metric].replace(0, np.nan)
            else:
                effective_scores = group[sort_metric].clip(lower=0)
            k = effective_k(
                effective_scores,
                min_n=effective_min_n,
                max_n=effective_max_n,
            )
            selected = group.head(k).copy()
            selection = "effective_number"
        elif top_quantile is not None:
            threshold = group[sort_metric].quantile(top_quantile)
            selected = group[group[sort_metric] >= threshold].copy()
            selection = f"quantile_{top_quantile:.2f}"
        elif top_z is not None:
            mu = group[sort_metric].mean()
            sigma = group[sort_metric].std(ddof=0)
            threshold = mu + top_z * sigma
            selected = group[group[sort_metric] >= threshold].copy()
            selection = f"z_{top_z:.2f}"
        else:
            selected = group.head(int(top_n)).copy()
            selection = f"top_{top_n}"

        if selected.empty:
            selected = group.head(max(1, int(top_n))).copy()
            selection = f"top_{top_n}_fallback"
        selected["rank_within_go"] = range(1, len(selected) + 1)
        selected["selection"] = selection
        selected["threshold"] = threshold
        rows.append(selected)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_top_pairs(
    path: Path,
    top_bp_pairs: set[tuple[str, str]],
    pair_top_n: int,
    pair_selection_method: str = "top_n",
    pair_cumulative_frac: float | None = None,
    pair_min_n: int = 1,
    pair_max_n: int | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    if pair_selection_method == "effective_number":
        return collect_pairs_by_effective_number(
            path,
            top_bp_pairs,
            pair_min_n=pair_min_n,
            pair_max_n=pair_max_n,
            chunksize=chunksize,
        )
    if pair_cumulative_frac is not None:
        return collect_pairs_by_cumulative(
            path,
            top_bp_pairs,
            pair_cumulative_frac=pair_cumulative_frac,
            pair_min_n=pair_min_n,
            pair_max_n=pair_max_n,
            chunksize=chunksize,
        )
    return collect_pairs_by_top_n(path, top_bp_pairs, pair_top_n=pair_top_n, chunksize=chunksize)


def _collect_group_rows(
    path: Path,
    top_bp_pairs: set[tuple[str, str]],
    chunksize: int = 200_000,
) -> dict[tuple[str, str], list[tuple[float, int]]]:
    rows_by_key: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    usecols = ["metapath", "go_id", "entrez_gene_id", "dwpc"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        keys = list(zip(chunk["metapath"], chunk["go_id"]))
        mask = [k in top_bp_pairs for k in keys]
        if not any(mask):
            continue
        sub = chunk.loc[mask]
        for row in sub.itertuples(index=False):
            rows_by_key[(row.metapath, row.go_id)].append((float(row.dwpc), int(row.entrez_gene_id)))
    return rows_by_key


def collect_pairs_by_top_n(
    path: Path,
    top_bp_pairs: set[tuple[str, str]],
    *,
    pair_top_n: int,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    rows_by_key = _collect_group_rows(path, top_bp_pairs, chunksize=chunksize)
    records = []
    for (metapath, go_id), rows in rows_by_key.items():
        ranked = sorted(rows, key=lambda x: (x[0], -x[1]), reverse=True)[: int(pair_top_n)]
        for rank, (dwpc, gene_id) in enumerate(ranked, start=1):
            records.append({"metapath": metapath, "go_id": go_id, "entrez_gene_id": gene_id, "dwpc": dwpc, "rank": rank})
    return pd.DataFrame(records)


def collect_pairs_by_cumulative(
    path: Path,
    top_bp_pairs: set[tuple[str, str]],
    *,
    pair_cumulative_frac: float,
    pair_min_n: int,
    pair_max_n: int | None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    rows_by_key = _collect_group_rows(path, top_bp_pairs, chunksize=chunksize)
    records = []
    for (metapath, go_id), rows in rows_by_key.items():
        ranked = sorted(rows, key=lambda x: (x[0], -x[1]), reverse=True)
        positive_total = sum(max(dwpc, 0.0) for dwpc, _ in ranked)
        cumulative = 0.0
        selected: list[tuple[float, int]] = []
        for dwpc, gene_id in ranked:
            selected.append((dwpc, gene_id))
            cumulative += max(dwpc, 0.0)
            enough_by_frac = positive_total <= 0 or cumulative / positive_total >= float(pair_cumulative_frac)
            enough_by_min = len(selected) >= int(pair_min_n)
            enough_by_max = pair_max_n is not None and len(selected) >= int(pair_max_n)
            if enough_by_max or (enough_by_min and enough_by_frac):
                break

        for rank, (dwpc, gene_id) in enumerate(selected, start=1):
            records.append(
                {
                    "metapath": metapath,
                    "go_id": go_id,
                    "entrez_gene_id": gene_id,
                    "dwpc": dwpc,
                    "rank": rank,
                    "pair_selection": f"cumulative_{pair_cumulative_frac:.2f}",
                    "pair_cumulative_fraction": pair_cumulative_frac,
                }
            )
    return pd.DataFrame(records)


def collect_pairs_by_effective_number(
    path: Path,
    top_bp_pairs: set[tuple[str, str]],
    *,
    pair_min_n: int,
    pair_max_n: int | None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    rows_by_key = _collect_group_rows(path, top_bp_pairs, chunksize=chunksize)
    records = []
    for (metapath, go_id), rows in rows_by_key.items():
        ranked = sorted(rows, key=lambda x: (x[0], -x[1]), reverse=True)
        scores = pd.Series([max(dwpc, 0.0) for dwpc, _ in ranked], dtype=float)
        k = effective_k(scores, min_n=pair_min_n, max_n=pair_max_n)
        for rank, (dwpc, gene_id) in enumerate(ranked[:k], start=1):
            records.append(
                {
                    "metapath": metapath,
                    "go_id": go_id,
                    "entrez_gene_id": gene_id,
                    "dwpc": dwpc,
                    "rank": rank,
                    "pair_selection": "effective_number",
                    "pair_effective_n": float(effective_number(scores)),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select supported GO-term/metapath combinations and top gene-BP pairs.")
    parser.add_argument("--years", nargs="+", default=["2016", "2024"])
    parser.add_argument("--top-n", type=int, default=5, help="Fallback top metapaths per GO term.")
    parser.add_argument(
        "--top-metapaths",
        type=int,
        default=None,
        help="Optional cap on the number of metapaths retained per GO term after support filtering.",
    )
    parser.add_argument("--pair-top-n", type=int, default=10, help="Top gene-BP pairs per BP (default 10).")
    parser.add_argument("--go-selection-method", default="effective_number", choices=["effective_number", "top_n", "quantile", "z"])
    parser.add_argument("--go-effective-min-n", type=int, default=1)
    parser.add_argument("--go-effective-max-n", type=int, default=None)
    parser.add_argument("--pair-selection-method", default="effective_number", choices=["effective_number", "top_n", "cumulative"])
    parser.add_argument("--pair-cumulative-frac", type=float, default=None, help="Retain pairs until this cumulative DWPC fraction is reached per metapath/GO term.")
    parser.add_argument("--pair-min-n", type=int, default=1, help="Minimum pairs retained when using --pair-cumulative-frac.")
    parser.add_argument("--pair-max-n", type=int, default=None, help="Optional cap on retained pairs per metapath/GO term.")
    parser.add_argument("--top-quantile", type=float, default=None, help="Quantile threshold per GO term when using quantile selection.")
    parser.add_argument("--top-z", type=float, default=None, help="Z-score threshold per GO term when using z selection.")
    parser.add_argument("--results-dir", default=None, help="Direct year result directory.")
    parser.add_argument("--output-dir", default=None, help="Destination for top-path selection CSVs.")
    parser.add_argument("--support-path", default=None, help="GO-term support table CSV.")
    parser.add_argument("--support-only", action="store_true", help="When support table is provided, keep only supported metapaths.")
    parser.add_argument(
        "--support-sort-metric",
        default="consensus_score",
        choices=[
            "consensus_score",
            "consensus_rank",
            "mean_std_score",
            "min_d",
            "min_diff",
            "diff_perm",
            "diff_rand",
            "real_mean",
            "rank_perm",
            "rank_rand",
        ],
        help="Sort metric for metapaths from support table within each GO term.",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    if args.top_quantile is not None and args.top_z is not None:
        raise ValueError("Use either --top-quantile or --top-z, not both.")

    repo_root = Path(__file__).resolve().parent.parent
    base_name = "all_GO_positive_growth"
    results_dir = Path(args.results_dir) if args.results_dir else repo_root / "output" / "dwpc_direct" / base_name / "results"
    out_dir = Path(args.output_dir) if args.output_dir else repo_root / "output" / "metapath_analysis" / "top_paths"
    out_dir.mkdir(parents=True, exist_ok=True)
    support_df = load_support_table(Path(args.support_path)) if args.support_path else None

    if args.go_selection_method == "quantile" and args.top_quantile is None:
        raise ValueError("--top-quantile is required when --go-selection-method=quantile")
    if args.go_selection_method == "z" and args.top_z is None:
        raise ValueError("--top-z is required when --go-selection-method=z")

    bp_map = load_name_map(repo_root / "data" / "nodes" / "Biological Process.tsv")
    gene_map = load_name_map(repo_root / "data" / "nodes" / "Gene.tsv")

    for year in args.years:
        path = results_dir / f"dwpc_{base_name}_{year}_real.csv"
        if not path.exists():
            print(f"Missing {path}, skipping.")
            continue

        print(f"Processing {year} from {path}")
        support_subset = select_supported_go_metapaths(
            support_df,
            year=int(year),
            support_only=bool(args.support_only),
            sort_metric=str(args.support_sort_metric),
            top_n=int(args.top_n),
            top_quantile=args.top_quantile,
            top_z=args.top_z,
            use_effective_number=(args.go_selection_method == "effective_number"),
            effective_min_n=int(args.go_effective_min_n),
            effective_max_n=args.go_effective_max_n if args.go_selection_method == "effective_number" else args.top_metapaths,
        )
        if support_subset is None:
            raise ValueError("GO-term support table is required for the updated year top-subgraph workflow")
        top_bps = support_subset.copy()
        top_bps["go_name"] = top_bps["go_id"].map(bp_map)
        top_bps["year"] = int(year)
        print(f"Retained {len(top_bps):,} GO-term/metapath rows for {year} from support table")

        top_bp_path = out_dir / f"top_bps_by_metapath_{year}.csv"
        top_bps.to_csv(top_bp_path, index=False)
        print(f"Saved: {top_bp_path}")

        top_pairs_key = set(zip(top_bps["metapath"], top_bps["go_id"]))
        pairs_df = collect_top_pairs(
            path,
            top_pairs_key,
            args.pair_top_n,
            pair_selection_method=str(args.pair_selection_method),
            pair_cumulative_frac=args.pair_cumulative_frac,
            pair_min_n=args.pair_min_n,
            pair_max_n=args.pair_max_n,
            chunksize=args.chunksize,
        )
        pairs_df["go_name"] = pairs_df["go_id"].map(bp_map)
        pairs_df["gene_name"] = pairs_df["entrez_gene_id"].map(gene_map)
        pairs_df["year"] = int(year)

        pair_path = out_dir / f"top_gene_bp_pairs_{year}.csv"
        pairs_df.to_csv(pair_path, index=False)
        print(f"Saved: {pair_path}")


if __name__ == "__main__":
    main()
