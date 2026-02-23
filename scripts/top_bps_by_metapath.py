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
Rank top Biological Processes (GO terms) per metapath and extract top gene-BP pairs.

Uses direct DWPC outputs:
  output/dwpc_direct/all_GO_positive_growth/results/dwpc_*_real.csv
"""

from __future__ import annotations

import argparse
import heapq
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_name_map(path: Path) -> dict:
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["identifier"], df["name"]))


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


def select_top_bps(
    df: pd.DataFrame,
    top_n: int,
    top_quantile: float | None,
    top_z: float | None,
) -> pd.DataFrame:
    results = []
    for metapath, group in df.groupby("metapath"):
        group = group.sort_values("mean_dwpc", ascending=False).reset_index(drop=True)
        selection = None
        threshold = None

        if top_quantile is not None:
            threshold = group["mean_dwpc"].quantile(top_quantile)
            selected = group[group["mean_dwpc"] >= threshold]
            selection = f"quantile_{top_quantile:.2f}"
        elif top_z is not None:
            mu = group["mean_dwpc"].mean()
            sigma = group["mean_dwpc"].std(ddof=0)
            threshold = mu + top_z * sigma
            selected = group[group["mean_dwpc"] >= threshold]
            selection = f"z_{top_z:.2f}"
        else:
            selected = group.head(top_n)
            selection = f"top_{top_n}"

        if selected.empty:
            selected = group.head(top_n)
            selection = f"top_{top_n}_fallback"

        selected = selected.copy()
        selected["rank"] = range(1, len(selected) + 1)
        selected["selection"] = selection
        selected["threshold"] = threshold
        results.append(selected)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def collect_top_pairs(
    path: Path,
    top_bp_pairs: set[tuple[str, str]],
    pair_top_n: int,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    heaps: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    usecols = ["metapath", "go_id", "entrez_gene_id", "dwpc"]

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        keys = list(zip(chunk["metapath"], chunk["go_id"]))
        mask = [k in top_bp_pairs for k in keys]
        if not any(mask):
            continue
        sub = chunk.loc[mask]
        for row in sub.itertuples(index=False):
            key = (row.metapath, row.go_id)
            item = (float(row.dwpc), int(row.entrez_gene_id))
            heap = heaps[key]
            if len(heap) < pair_top_n:
                heapq.heappush(heap, item)
            else:
                if item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

    records = []
    for (metapath, go_id), heap in heaps.items():
        top = sorted(heap, key=lambda x: x[0], reverse=True)
        for rank, (dwpc, gene_id) in enumerate(top, start=1):
            records.append(
                {
                    "metapath": metapath,
                    "go_id": go_id,
                    "entrez_gene_id": gene_id,
                    "dwpc": dwpc,
                    "rank": rank,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Top BP terms and top gene-BP pairs per metapath.")
    parser.add_argument("--years", nargs="+", default=["2016", "2024"])
    parser.add_argument("--top-n", type=int, default=5, help="Top BPs per metapath (default 5).")
    parser.add_argument("--pair-top-n", type=int, default=10, help="Top gene-BP pairs per BP (default 10).")
    parser.add_argument("--top-quantile", type=float, default=None, help="Quantile threshold per metapath.")
    parser.add_argument("--top-z", type=float, default=None, help="Z-score threshold per metapath.")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    if args.top_quantile is not None and args.top_z is not None:
        raise ValueError("Use either --top-quantile or --top-z, not both.")

    repo_root = Path(__file__).resolve().parent.parent
    base_name = "all_GO_positive_growth"
    results_dir = repo_root / "output" / "dwpc_direct" / base_name / "results"
    out_dir = repo_root / "output" / "metapath_analysis" / "top_paths"
    out_dir.mkdir(parents=True, exist_ok=True)

    bp_map = load_name_map(repo_root / "data" / "nodes" / "Biological Process.tsv")
    gene_map = load_name_map(repo_root / "data" / "nodes" / "Gene.tsv")

    for year in args.years:
        path = results_dir / f"dwpc_{base_name}_{year}_real.csv"
        if not path.exists():
            print(f"Missing {path}, skipping.")
            continue

        print(f"Processing {year} from {path}")
        mean_df = compute_mean_dwpc(path, chunksize=args.chunksize)
        top_bps = select_top_bps(mean_df, args.top_n, args.top_quantile, args.top_z)
        top_bps["go_name"] = top_bps["go_id"].map(bp_map)
        top_bps["year"] = int(year)

        top_bp_path = out_dir / f"top_bps_by_metapath_{year}.csv"
        top_bps.to_csv(top_bp_path, index=False)
        print(f"Saved: {top_bp_path}")

        top_pairs_key = set(zip(top_bps["metapath"], top_bps["go_id"]))
        pairs_df = collect_top_pairs(path, top_pairs_key, args.pair_top_n, chunksize=args.chunksize)
        pairs_df["go_name"] = pairs_df["go_id"].map(bp_map)
        pairs_df["gene_name"] = pairs_df["entrez_gene_id"].map(gene_map)
        pairs_df["year"] = int(year)

        pair_path = out_dir / f"top_gene_bp_pairs_{year}.csv"
        pairs_df.to_csv(pair_path, index=False)
        print(f"Saved: {pair_path}")


if __name__ == "__main__":
    main()
