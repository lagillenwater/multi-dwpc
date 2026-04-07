#!/usr/bin/env python3
"""Coverage of 2024-added GO-gene annotations under top-K 2024 consensus metapaths."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SUPPORT_PATH = REPO_ROOT / "output" / "year_direct_go_term_support_b5.csv"
DEFAULT_ADDED_PAIRS_PATH = REPO_ROOT / "output" / "intermediate" / "upd_go_bp_2024_added.csv"
DEFAULT_RESULTS_PATH = REPO_ROOT / "output" / "dwpc_direct" / "all_GO_positive_growth" / "results" / "dwpc_all_GO_positive_growth_2024_real.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "metapath_analysis" / "consensus_added_coverage_b5"
NODE_ABBREVS = ["BP", "CC", "MF", "PW", "SE", "PC", "G", "A", "D", "C", "S"]
NODE_ABBREVS = sorted(NODE_ABBREVS, key=len, reverse=True)


def _parse_int_list(arg: str) -> list[int]:
    vals = sorted({int(tok.strip()) for tok in str(arg).split(",") if tok.strip()})
    if not vals:
        raise ValueError("Expected at least one integer top-k value")
    return vals


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
    selected["rank_within_go"] = selected.groupby("go_id", sort=False).cumcount().add(1)
    return selected


def _load_added_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    required = {"go_id", "entrez_gene_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Added-pairs file missing required columns: {sorted(missing)}")
    if "go_name" not in df.columns:
        df["go_name"] = pd.NA
    if "gene_symbol" not in df.columns:
        if "gene_name" in df.columns:
            df = df.rename(columns={"gene_name": "gene_symbol"})
        else:
            df["gene_symbol"] = pd.NA
    out = df[["go_id", "go_name", "entrez_gene_id", "gene_symbol"]].copy()
    out["go_id"] = out["go_id"].astype(str)
    out["entrez_gene_id"] = out["entrez_gene_id"].astype(int)
    out["go_name"] = out["go_name"].astype("string")
    out["gene_symbol"] = out["gene_symbol"].astype("string")
    return out.drop_duplicates().reset_index(drop=True)


def _parse_nodes(metapath: str) -> list[str]:
    nodes: list[str] = []
    i = 0
    text = str(metapath)
    while i < len(text):
        matched = False
        for ab in NODE_ABBREVS:
            if text.startswith(ab, i):
                nodes.append(ab)
                i += len(ab)
                matched = True
                break
        if not matched:
            i += 1
    return nodes


def _has_excluded_intermediate_node(metapath: str, excluded_nodes: set[str]) -> bool:
    nodes = _parse_nodes(str(metapath))
    internal = nodes[1:-1]
    return any(node in excluded_nodes for node in internal)


def _scan_results(
    results_path: Path,
    *,
    selected_go_metapaths_df: pd.DataFrame,
    added_pairs_df: pd.DataFrame,
    chunksize: int,
) -> pd.DataFrame:
    selected_lookup = selected_go_metapaths_df[
        ["go_id", "metapath", "consensus_score", "consensus_rank", "rank_within_go"]
    ].drop_duplicates()
    added_lookup = added_pairs_df[["go_id", "entrez_gene_id", "go_name", "gene_symbol"]].drop_duplicates()

    hits: list[pd.DataFrame] = []
    usecols = ["entrez_gene_id", "go_id", "metapath", "dwpc"]
    for chunk in pd.read_csv(results_path, usecols=usecols, chunksize=chunksize):
        chunk["go_id"] = chunk["go_id"].astype(str)
        chunk["metapath"] = chunk["metapath"].astype(str)
        chunk["entrez_gene_id"] = chunk["entrez_gene_id"].astype(int)
        sub = chunk.merge(selected_lookup, on=["go_id", "metapath"], how="inner")
        if sub.empty:
            continue
        sub = sub.merge(added_lookup, on=["go_id", "entrez_gene_id"], how="inner")
        if sub.empty:
            continue
        hits.append(sub)

    if not hits:
        return pd.DataFrame(
            columns=[
                "go_id",
                "go_name",
                "entrez_gene_id",
                "gene_symbol",
                "metapath",
                "dwpc",
                "consensus_score",
                "consensus_rank",
                "rank_within_go",
            ]
        )
    return pd.concat(hits, ignore_index=True)


def _summarize_for_k(
    *,
    top_k: int,
    support_2024: pd.DataFrame,
    added_df: pd.DataFrame,
    results_path: Path,
    dwpc_threshold: float,
    chunksize: int,
    output_dir: Path,
) -> dict:
    topk_2024 = _select_top_k_metapaths_per_go(support_2024, top_k=int(top_k))
    hits_df = _scan_results(
        results_path,
        selected_go_metapaths_df=topk_2024,
        added_pairs_df=added_df,
        chunksize=chunksize,
    )
    hits_df["dwpc"] = pd.to_numeric(hits_df["dwpc"], errors="coerce")

    positive_hits = hits_df[hits_df["dwpc"] > float(dwpc_threshold)].copy()
    positive_hits = positive_hits.sort_values(
        ["go_id", "entrez_gene_id", "dwpc", "consensus_score", "rank_within_go", "metapath"],
        ascending=[True, True, False, False, True, True],
    ).reset_index(drop=True)

    if positive_hits.empty:
        best_pairs = pd.DataFrame(
            columns=[
                "metapath",
                "go_id",
                "go_name",
                "entrez_gene_id",
                "gene_symbol",
                "dwpc",
                "consensus_score",
                "consensus_rank",
                "rank_within_go",
                "rank",
                "year",
            ]
        )
    else:
        best_pairs = positive_hits.groupby(["go_id", "entrez_gene_id"], as_index=False, sort=False).head(1).copy()
        best_pairs["rank"] = 1
        best_pairs["year"] = 2024

    pair_summary = (
        hits_df.groupby(["go_id", "entrez_gene_id"], as_index=False)
        .agg(
            go_name=("go_name", "first"),
            gene_symbol=("gene_symbol", "first"),
            n_topk_metapath_rows=("metapath", "size"),
            n_positive_topk_metapaths=("dwpc", lambda x: int((pd.to_numeric(x, errors="coerce") > float(dwpc_threshold)).sum())),
            best_2024_dwpc=("dwpc", "max"),
            best_consensus_score=("consensus_score", "max"),
        )
        if not hits_df.empty
        else pd.DataFrame(columns=["go_id", "entrez_gene_id", "go_name", "gene_symbol", "n_topk_metapath_rows", "n_positive_topk_metapaths", "best_2024_dwpc", "best_consensus_score"])
    )
    if not pair_summary.empty:
        pair_summary["covered_by_topk_metapaths"] = pair_summary["n_positive_topk_metapaths"].astype(int) > 0
    else:
        pair_summary["covered_by_topk_metapaths"] = pd.Series(dtype=bool)

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

    topk_2024.to_csv(output_dir / f"top_metapaths_by_go_2024_top{int(top_k)}.csv", index=False)
    best_pairs.to_csv(output_dir / f"top_gene_bp_pairs_2024_top{int(top_k)}.csv", index=False)
    pair_summary.to_csv(output_dir / f"covered_added_pairs_top{int(top_k)}.csv", index=False)
    metapath_summary.to_csv(output_dir / f"coverage_by_metapath_top{int(top_k)}.csv", index=False)

    n_added_total = int(len(added_df))
    n_pairs_with_any_rows = int(pair_summary.shape[0])
    n_pairs_covered = int(pair_summary["covered_by_topk_metapaths"].sum()) if not pair_summary.empty else 0
    return {
        "top_k": int(top_k),
        "dwpc_threshold": float(dwpc_threshold),
        "n_go_terms_2024": int(topk_2024["go_id"].nunique()),
        "n_go_metapath_rows_2024": int(len(topk_2024)),
        "n_added_pairs_total": n_added_total,
        "n_added_pairs_with_any_topk_row_2024": n_pairs_with_any_rows,
        "n_added_pairs_covered_in_2024": n_pairs_covered,
        "coverage_of_added_pairs_in_2024": (float(n_pairs_covered) / float(n_added_total)) if n_added_total else np.nan,
        "fraction_with_positive_dwpc_among_pairs_with_any_topk_row": (float(n_pairs_covered) / float(n_pairs_with_any_rows)) if n_pairs_with_any_rows else np.nan,
        "median_positive_metapaths_per_covered_pair": (
            float(pair_summary.loc[pair_summary["covered_by_topk_metapaths"], "n_positive_topk_metapaths"].median())
            if not pair_summary.empty and n_pairs_covered > 0 else np.nan
        ),
        "median_best_2024_dwpc_among_covered_pairs": (
            float(pair_summary.loc[pair_summary["covered_by_topk_metapaths"], "best_2024_dwpc"].median())
            if not pair_summary.empty and n_pairs_covered > 0 else np.nan
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-path", default=str(DEFAULT_SUPPORT_PATH))
    parser.add_argument("--added-pairs-path", default=str(DEFAULT_ADDED_PAIRS_PATH))
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k-values", default="1,2,3")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--dwpc-threshold", type=float, default=0.0)
    parser.add_argument(
        "--exclude-intermediate-nodes",
        default="",
        help="Comma-separated internal node abbreviations to exclude from candidate metapaths, e.g. BP,CC,MF,PW",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_2024 = _load_support(Path(args.support_path), year=2024)
    if str(args.exclude_intermediate_nodes).strip():
        excluded_nodes = {
            tok.strip()
            for tok in str(args.exclude_intermediate_nodes).split(",")
            if tok.strip()
        }
        before = int(support_2024["metapath"].nunique())
        support_2024 = support_2024[
            ~support_2024["metapath"].map(lambda mp: _has_excluded_intermediate_node(mp, excluded_nodes))
        ].copy()
        after = int(support_2024["metapath"].nunique())
        print(
            "Excluded internal-node metapaths: "
            f"{sorted(excluded_nodes)}; removed {before - after} unique metapaths, kept {after}"
        )
    added_df = _load_added_pairs(Path(args.added_pairs_path))
    top_k_values = _parse_int_list(args.top_k_values)

    summary_rows = []
    for top_k in top_k_values:
        summary_rows.append(
            _summarize_for_k(
                top_k=int(top_k),
                support_2024=support_2024,
                added_df=added_df,
                results_path=Path(args.results_path),
                dwpc_threshold=float(args.dwpc_threshold),
                chunksize=int(args.chunksize),
                output_dir=output_dir,
            )
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("top_k").reset_index(drop=True)
    summary_df.to_csv(output_dir / "coverage_summary.csv", index=False)

    print(f"Saved summary: {output_dir / 'coverage_summary.csv'}")
    for top_k in top_k_values:
        print(f"Saved top-{int(top_k)} outputs under: {output_dir}")


if __name__ == "__main__":
    main()
