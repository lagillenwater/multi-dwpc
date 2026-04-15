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
        default="median_effect_size_d",
        choices=["median_effect_size_d", "max_effect_size_d"],
        help="Column used to rank GO terms (default: median_effect_size_d).",
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
    if "go_id" not in df.columns or "effect_size_d" not in df.columns:
        raise ValueError(
            f"{csv_path} must have go_id and effect_size_d columns; "
            f"found: {sorted(df.columns.tolist())}"
        )

    agg = {
        "n_metapaths": ("metapath", "count"),
        "median_effect_size_d": ("effect_size_d", "median"),
        "max_effect_size_d": ("effect_size_d", "max"),
    }
    for optional in ("n_genes_2016", "n_genes_2024_added"):
        if optional in df.columns:
            agg[optional] = (optional, "first")

    go_summary = df.groupby("go_id").agg(**agg).reset_index()

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
