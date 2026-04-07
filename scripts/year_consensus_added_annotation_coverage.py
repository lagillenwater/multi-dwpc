#!/usr/bin/env python3
"""Coverage of newly added 2024 GO-gene annotations using top-K 2024 consensus metapaths in the 2016 graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SUPPORT_PATH = REPO_ROOT / "output" / "year_direct_go_term_support_b5.csv"
DEFAULT_ADDED_PAIRS_PATH = REPO_ROOT / "output" / "intermediate" / "upd_go_bp_2024_added.csv"
DEFAULT_RESULTS_2016_PATH = REPO_ROOT / "output" / "dwpc_direct" / "all_GO_positive_growth" / "results" / "dwpc_all_GO_positive_growth_2016_real.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "metapath_analysis" / "top_paths_consensus_exact_k_b5"


def _load_name_map(path: Path) -> dict:
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["identifier"], df["name"]))


def _load_support(path: Path, *, year: int) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    required = {"year", "go_id", "metapath", "consensus_score", "consensus_rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Support table missing required columns: {sorted(missing)}")
    df["year"] = df["year"].astype(int)
    df["go_id"] = df["go_id"].astype(str)
    df["metapath"] = df["metapath"].astype(str)
    df["consensus_score"] = pd.to_numeric(df["consensus_score"], errors="coerce")
    df["consensus_rank"] = pd.to_numeric(df["consensus_rank"], errors="coerce")
    return df[df["year"] == int(year)].copy()


def _select_top_k_metapaths_per_go(support_df: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    selected = (
        support_df.sort_values(
            ["go_id", "consensus_score", "consensus_rank", "metapath"],
            ascending=[True, False, True, True],
        )
        .groupby("go_id", as_index=False, sort=False)
        .head(int(top_k))
        .copy()
    )
    selected["rank_within_go"] = (
        selected.groupby("go_id", sort=False).cumcount().add(1)
    )
    return selected


def _load_added_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["go_id", "go_name", "entrez_gene_id", "gene_symbol"]).copy()
    df["go_id"] = df["go_id"].astype(str)
    df["entrez_gene_id"] = df["entrez_gene_id"].astype(int)
    df["gene_symbol"] = df["gene_symbol"].astype(str)
    return df.drop_duplicates().reset_index(drop=True)


def _pair_key_frame(df: pd.DataFrame) -> pd.Series:
    return df[["go_id", "entrez_gene_id"]].apply(lambda r: (str(r["go_id"]), int(r["entrez_gene_id"])), axis=1)


def _scan_2016_results(
    results_path: Path,
    *,
    selected_go_metapaths: set[tuple[str, str]],
    added_pairs: set[tuple[str, int]],
    chunksize: int,
) -> pd.DataFrame:
    usecols = ["metapath", "go_id", "entrez_gene_id", "dwpc"]
    kept = []
    for chunk in pd.read_csv(results_path, usecols=usecols, chunksize=chunksize):
        chunk["go_id"] = chunk["go_id"].astype(str)
        chunk["metapath"] = chunk["metapath"].astype(str)
        chunk["entrez_gene_id"] = chunk["entrez_gene_id"].astype(int)
        mask_go_mp = [(go_id, mp) in selected_go_metapaths for go_id, mp in zip(chunk["go_id"], chunk["metapath"])]
        if not any(mask_go_mp):
            continue
        sub = chunk.loc[mask_go_mp].copy()
        mask_pairs = [(go_id, gene_id) in added_pairs for go_id, gene_id in zip(sub["go_id"], sub["entrez_gene_id"])]
        if not any(mask_pairs):
            continue
        kept.append(sub.loc[mask_pairs].copy())
    if not kept:
        return pd.DataFrame(columns=["metapath", "go_id", "entrez_gene_id", "dwpc"])
    return pd.concat(kept, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-path", default=str(DEFAULT_SUPPORT_PATH))
    parser.add_argument("--added-pairs-path", default=str(DEFAULT_ADDED_PAIRS_PATH))
    parser.add_argument("--results-2016-path", default=str(DEFAULT_RESULTS_2016_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--dwpc-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_2024 = _load_support(Path(args.support_path), year=2024)
    topk_2024 = _select_top_k_metapaths_per_go(support_2024, top_k=int(args.top_k))
    topk_2024.to_csv(output_dir / "top_metapaths_by_go_2024.csv", index=False)

    added_df = _load_added_pairs(Path(args.added_pairs_path))
    added_pairs = set(_pair_key_frame(added_df).tolist())
    selected_go_metapaths = set(zip(topk_2024["go_id"], topk_2024["metapath"]))

    hits_2016 = _scan_2016_results(
        Path(args.results_2016_path),
        selected_go_metapaths=selected_go_metapaths,
        added_pairs=added_pairs,
        chunksize=int(args.chunksize),
    )

    go_name_map = dict(zip(added_df["go_id"], added_df["go_name"]))
    gene_name_map = dict(zip(added_df["entrez_gene_id"], added_df["gene_symbol"]))
    topk_lookup = topk_2024[
        ["go_id", "metapath", "consensus_score", "consensus_rank", "rank_within_go"]
    ].drop_duplicates()
    hits_2016 = hits_2016.merge(topk_lookup, on=["go_id", "metapath"], how="left")
    hits_2016["go_name"] = hits_2016["go_id"].map(go_name_map)
    hits_2016["gene_name"] = hits_2016["entrez_gene_id"].map(gene_name_map)
    hits_2016["year"] = 2016

    if hits_2016.empty:
        best_pairs = pd.DataFrame(
            columns=[
                "metapath",
                "go_id",
                "go_name",
                "entrez_gene_id",
                "gene_name",
                "dwpc",
                "consensus_score",
                "consensus_rank",
                "rank_within_go",
                "rank",
                "year",
            ]
        )
        covered_pair_keys: set[tuple[str, int]] = set()
        positive_hits = hits_2016.copy()
    else:
        positive_hits = hits_2016[hits_2016["dwpc"].astype(float) > float(args.dwpc_threshold)].copy()
        positive_hits = positive_hits.sort_values(
            ["go_id", "entrez_gene_id", "dwpc", "consensus_score", "rank_within_go", "metapath"],
            ascending=[True, True, False, False, True, True],
        ).reset_index(drop=True)
        best_pairs = (
            positive_hits.groupby(["go_id", "entrez_gene_id"], as_index=False, sort=False)
            .head(1)
            .copy()
        )
        best_pairs["rank"] = 1
        covered_pair_keys = set(_pair_key_frame(best_pairs).tolist())

    best_pairs.to_csv(output_dir / "top_gene_bp_pairs_2016.csv", index=False)

    pair_summary = (
        hits_2016.groupby(["go_id", "entrez_gene_id"], as_index=False)
        .agg(
            n_topk_metapath_rows=("metapath", "size"),
            n_positive_2016_metapaths=("dwpc", lambda x: int((pd.to_numeric(x, errors="coerce") > float(args.dwpc_threshold)).sum())),
            best_2016_dwpc=("dwpc", "max"),
            best_2024_consensus_score=("consensus_score", "max"),
        )
        if not hits_2016.empty
        else pd.DataFrame(columns=["go_id", "entrez_gene_id", "n_topk_metapath_rows", "n_positive_2016_metapaths", "best_2016_dwpc", "best_2024_consensus_score"])
    )
    pair_summary["go_name"] = pair_summary["go_id"].map(go_name_map)
    pair_summary["gene_symbol"] = pair_summary["entrez_gene_id"].map(gene_name_map)
    pair_summary["covered_in_2016_by_topk"] = pair_summary["n_positive_2016_metapaths"].fillna(0).astype(int) > 0
    pair_summary.to_csv(output_dir / "covered_added_pairs.csv", index=False)

    metapath_summary = (
        positive_hits.groupby("metapath", as_index=False)
        .agg(
            n_covered_added_pairs=("go_id", "size"),
            n_unique_go_terms=("go_id", "nunique"),
            best_dwpc=("dwpc", "max"),
        )
        .sort_values(["n_covered_added_pairs", "n_unique_go_terms", "best_dwpc", "metapath"], ascending=[False, False, False, True])
        .reset_index(drop=True)
        if not positive_hits.empty
        else pd.DataFrame(columns=["metapath", "n_covered_added_pairs", "n_unique_go_terms", "best_dwpc"])
    )
    metapath_summary.to_csv(output_dir / "coverage_by_metapath.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "top_k": int(args.top_k),
                "dwpc_threshold": float(args.dwpc_threshold),
                "n_go_terms_2024": int(topk_2024["go_id"].nunique()),
                "n_go_metapath_rows_2024": int(len(topk_2024)),
                "n_added_pairs_total": int(len(added_df)),
                "n_added_pairs_with_any_topk_row_2016": int(pair_summary.shape[0]),
                "n_added_pairs_covered_in_2016": int(len(covered_pair_keys)),
                "coverage_of_added_pairs_in_2016": float(len(covered_pair_keys)) / float(len(added_df)) if len(added_df) else np.nan,
                "fraction_with_positive_dwpc_among_pairs_with_any_topk_row": float(len(covered_pair_keys)) / float(pair_summary.shape[0]) if pair_summary.shape[0] else np.nan,
                "median_positive_metapaths_per_covered_pair": float(pair_summary.loc[pair_summary["covered_in_2016_by_topk"], "n_positive_2016_metapaths"].median()) if not pair_summary.empty else np.nan,
                "median_best_2016_dwpc_among_covered_pairs": float(pair_summary.loc[pair_summary["covered_in_2016_by_topk"], "best_2016_dwpc"].median()) if not pair_summary.empty else np.nan,
            }
        ]
    )
    summary.to_csv(output_dir / "coverage_summary.csv", index=False)

    print(f"Saved top-K metapaths: {output_dir / 'top_metapaths_by_go_2024.csv'}")
    print(f"Saved 2016 best-pair table: {output_dir / 'top_gene_bp_pairs_2016.csv'}")
    print(f"Saved pair coverage table: {output_dir / 'covered_added_pairs.csv'}")
    print(f"Saved metapath coverage table: {output_dir / 'coverage_by_metapath.csv'}")
    print(f"Saved summary: {output_dir / 'coverage_summary.csv'}")


if __name__ == "__main__":
    main()
