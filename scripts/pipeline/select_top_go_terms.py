#!/usr/bin/env python3
"""Select top GO terms from year intermediate-sharing results.

Reads `intermediate_sharing_by_metapath.csv` from a chosen-B directory,
aggregates per GO term (median effect size across its surviving metapaths),
takes the top N, and writes two artifacts:

    <output-dir>/top_go_terms.csv   -- full summary per selected GO term
    <output-dir>/top_go_ids.json    -- just the ordered list of go_id strings

The JSON list is consumed by `plot_metapath_subgraphs.py` and related
downstream scripts via `scripts/read_json_value.py`.

Usage:
    python3 scripts/select_top_go_terms.py \\
        --input-dir output/year_full_analysis/intermediate_sharing/b10 \\
        --top-n 10 \\
        --output-dir output/year_full_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory holding intermediate_sharing_by_metapath.csv "
        "(usually a b{value}/ subdir).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many top GO terms to select (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write top_go_terms.csv and top_go_ids.json.",
    )
    parser.add_argument(
        "--rank-by",
        default="median_permutation_z",
        choices=["median_permutation_z", "max_permutation_z"],
        help="Column used to rank GO terms (default: median_permutation_z).",
    )
    parser.add_argument(
        "--min-added-genes",
        type=int,
        default=5,
        help=(
            "Require at least this many 2024-added genes per GO term. "
            "Guards against small-cohort artifacts in within-cohort Jaccards "
            "and against z-score inflation from low null variance on tiny "
            "cohorts. Default: 5."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = input_dir / "intermediate_sharing_by_metapath.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_path} to exist")

    df = pd.read_csv(csv_path)
    if "go_id" not in df.columns or "permutation_z" not in df.columns:
        raise ValueError(
            f"{csv_path} must have go_id and permutation_z columns; "
            f"found: {sorted(df.columns.tolist())}"
        )

    agg = {
        "n_metapaths": ("metapath", "count"),
        "median_permutation_z": ("permutation_z", "median"),
        "max_permutation_z": ("permutation_z", "max"),
    }
    for optional in ("n_genes_2016", "n_genes_2024_added"):
        if optional in df.columns:
            agg[optional] = (optional, "first")

    go_summary = df.groupby("go_id").agg(**agg).reset_index()

    n_before_filter = len(go_summary)
    if args.min_added_genes > 0 and "n_genes_2024_added" in go_summary.columns:
        go_summary = go_summary[
            go_summary["n_genes_2024_added"] >= int(args.min_added_genes)
        ].reset_index(drop=True)
        print(
            f"Filter: n_genes_2024_added >= {args.min_added_genes} "
            f"kept {len(go_summary)} of {n_before_filter} GO terms"
        )

    top_go = go_summary.nlargest(int(args.top_n), args.rank_by).reset_index(drop=True)
    top_go.to_csv(output_dir / "top_go_terms.csv", index=False)
    print(f"Selected {len(top_go)} GO terms ranked by {args.rank_by}")
    print(f"Saved: {output_dir / 'top_go_terms.csv'}")

    top_go_ids = top_go["go_id"].astype(str).tolist()
    with open(output_dir / "top_go_ids.json", "w") as f:
        json.dump(top_go_ids, f, indent=2)
    print(f"Saved: {output_dir / 'top_go_ids.json'}")
    print(f"Top GO terms: {top_go_ids}")


if __name__ == "__main__":
    main()
