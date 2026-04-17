#!/usr/bin/env python3
"""Summarize added-edge coverage in the year top-subgraph outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()


DEFAULT_ADDED_PAIRS_PATH = REPO_ROOT / "output" / "intermediate" / "upd_go_bp_2024_added.csv"
DEFAULT_TOP_PAIRS_PATH = REPO_ROOT / "output" / "metapath_analysis" / "top_paths" / "top_gene_bp_pairs_2024.csv"
DEFAULT_PATH_INSTANCES_PATH = REPO_ROOT / "output" / "metapath_analysis" / "top_paths" / "path_instances_2024.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "metapath_analysis" / "top_paths" / "coverage_2024_added"


PAIR_COLS = ["go_id", "entrez_gene_id"]


def _pair_key_frame(df: pd.DataFrame) -> pd.Series:
    return df[PAIR_COLS].astype({"go_id": str, "entrez_gene_id": int}).apply(tuple, axis=1)


def _load_pair_file(path: Path, extra_cols: list[str] | None = None) -> pd.DataFrame:
    cols = [*PAIR_COLS, *(extra_cols or [])]
    df = pd.read_csv(path, usecols=cols).copy()
    df["go_id"] = df["go_id"].astype(str)
    df["entrez_gene_id"] = df["entrez_gene_id"].astype(int)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--added-pairs-path", default=str(DEFAULT_ADDED_PAIRS_PATH))
    parser.add_argument("--top-pairs-path", default=str(DEFAULT_TOP_PAIRS_PATH))
    parser.add_argument("--path-instances-path", default=str(DEFAULT_PATH_INSTANCES_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--year-a", type=int, default=2016)
    parser.add_argument("--year-b", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    added_pairs_path = Path(args.added_pairs_path)
    top_pairs_path = Path(args.top_pairs_path)
    path_instances_path = Path(args.path_instances_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    added_df = _load_pair_file(added_pairs_path).drop_duplicates().reset_index(drop=True)
    top_pairs_df = _load_pair_file(top_pairs_path, extra_cols=["metapath", "dwpc", "rank"])
    path_df = _load_pair_file(path_instances_path, extra_cols=["metapath", "path_rank", "path_score"])

    added_pairs = set(_pair_key_frame(added_df).tolist())
    top_pairs_df["pair_key"] = _pair_key_frame(top_pairs_df)
    path_df["pair_key"] = _pair_key_frame(path_df)

    selected_top_pairs_df = top_pairs_df[top_pairs_df["pair_key"].isin(added_pairs)].copy()
    selected_path_df = path_df[path_df["pair_key"].isin(added_pairs)].copy()

    selected_top_pair_keys = set(selected_top_pairs_df["pair_key"].tolist())
    selected_path_pair_keys = set(selected_path_df["pair_key"].tolist())

    pair_support_df = (
        selected_path_df.groupby(PAIR_COLS, as_index=False)
        .agg(
            n_path_rows=("metapath", "size"),
            n_distinct_path_metapaths=("metapath", "nunique"),
            best_path_score=("path_score", "max"),
        )
        .sort_values(["n_path_rows", "n_distinct_path_metapaths", "best_path_score"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    if not pair_support_df.empty:
        pair_support_df["covered_by_path_instances"] = True

    top_pair_summary_df = (
        selected_top_pairs_df.groupby(PAIR_COLS, as_index=False)
        .agg(
            n_top_pair_rows=("metapath", "size"),
            n_distinct_top_metapaths=("metapath", "nunique"),
            best_dwpc=("dwpc", "max"),
        )
        .sort_values(["n_top_pair_rows", "n_distinct_top_metapaths", "best_dwpc"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    pair_level_df = top_pair_summary_df.merge(pair_support_df, on=PAIR_COLS, how="left")
    if not pair_level_df.empty:
        pair_level_df["covered_by_path_instances"] = (
            pair_level_df["covered_by_path_instances"] == True
        )
        pair_level_df["n_path_rows"] = pair_level_df["n_path_rows"].fillna(0).astype(int)
        pair_level_df["n_distinct_path_metapaths"] = (
            pair_level_df["n_distinct_path_metapaths"].fillna(0).astype(int)
        )

    metapath_summary_df = (
        selected_top_pairs_df.groupby("metapath", as_index=False)
        .agg(
            n_top_pair_rows=("pair_key", "size"),
            n_unique_added_pairs=("pair_key", "nunique"),
        )
        .sort_values(["n_unique_added_pairs", "n_top_pair_rows", "metapath"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    summary_row = {
        "year_a": int(args.year_a),
        "year_b": int(args.year_b),
        "n_added_pairs_total": int(len(added_pairs)),
        "n_top_pair_rows": int(len(top_pairs_df)),
        "n_top_pair_rows_in_added": int(len(selected_top_pairs_df)),
        "n_unique_top_pairs": int(top_pairs_df[PAIR_COLS].drop_duplicates().shape[0]),
        "n_unique_top_pairs_in_added": int(len(selected_top_pair_keys)),
        "fraction_unique_top_pairs_in_added": (
            float(len(selected_top_pair_keys)) / float(top_pairs_df[PAIR_COLS].drop_duplicates().shape[0])
            if len(top_pairs_df) > 0
            else float("nan")
        ),
        "n_unique_path_pairs_in_added": int(len(selected_path_pair_keys)),
        "coverage_of_added_by_top_pairs": (
            float(len(selected_top_pair_keys)) / float(len(added_pairs))
            if added_pairs
            else float("nan")
        ),
        "coverage_of_added_by_path_instances": (
            float(len(selected_path_pair_keys)) / float(len(added_pairs))
            if added_pairs
            else float("nan")
        ),
        "fraction_selected_top_pairs_with_path_instances": (
            float(len(selected_path_pair_keys)) / float(len(selected_top_pair_keys))
            if selected_top_pair_keys
            else float("nan")
        ),
        "median_path_rows_per_selected_pair": (
            float(pair_level_df["n_path_rows"].median()) if not pair_level_df.empty else float("nan")
        ),
        "median_distinct_path_metapaths_per_selected_pair": (
            float(pair_level_df["n_distinct_path_metapaths"].median()) if not pair_level_df.empty else float("nan")
        ),
        "max_path_rows_per_selected_pair": (
            int(pair_level_df["n_path_rows"].max()) if not pair_level_df.empty else 0
        ),
    }
    summary_df = pd.DataFrame([summary_row])

    summary_df.to_csv(output_dir / "coverage_summary.csv", index=False)
    pair_level_df.to_csv(output_dir / "covered_added_pairs.csv", index=False)
    metapath_summary_df.to_csv(output_dir / "coverage_by_metapath.csv", index=False)

    print(f"Added GO-gene pairs ({args.year_b} only): {summary_row['n_added_pairs_total']:,}")
    print(
        "Top-pair coverage of added pairs: "
        f"{summary_row['n_unique_top_pairs_in_added']:,} "
        f"({summary_row['coverage_of_added_by_top_pairs']:.3%})"
    )
    print(
        "Path-instance coverage of added pairs: "
        f"{summary_row['n_unique_path_pairs_in_added']:,} "
        f"({summary_row['coverage_of_added_by_path_instances']:.3%})"
    )
    print(
        "Selected top pairs with path-instance support: "
        f"{summary_row['fraction_selected_top_pairs_with_path_instances']:.3%}"
    )
    print(f"Saved summary: {output_dir / 'coverage_summary.csv'}")
    print(f"Saved pair table: {output_dir / 'covered_added_pairs.csv'}")
    print(f"Saved metapath summary: {output_dir / 'coverage_by_metapath.csv'}")


if __name__ == "__main__":
    main()
