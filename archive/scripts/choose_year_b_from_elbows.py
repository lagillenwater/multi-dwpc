#!/usr/bin/env python3
"""Choose a recommended year-analysis B from saved elbow summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing elbow summary: {path}")
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variance-elbows",
        default="output/year_null_variance_exp/year_null_variance_experiment/elbow_summary.csv",
    )
    parser.add_argument(
        "--rank-elbows",
        default="output/year_rank_stability_exp/year_rank_stability_experiment/elbow_summary.csv",
    )
    parser.add_argument(
        "--output-path",
        default="output/year_b_choice_summary.csv",
    )
    parser.add_argument(
        "--aggregation",
        default="max",
        choices=["max", "median"],
        help="How to combine elbow recommendations across metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variance_df = _load_csv(Path(args.variance_elbows)).copy()
    variance_df["source"] = "variance"
    rank_df = _load_csv(Path(args.rank_elbows)).copy()
    rank_df["source"] = "rank"
    elbows = pd.concat([variance_df, rank_df], ignore_index=True)
    if elbows.empty:
        raise ValueError("No elbow rows found.")

    if args.aggregation == "median":
        chosen_b = int(round(float(elbows["elbow_b"].median())))
    else:
        chosen_b = int(elbows["elbow_b"].max())

    summary = pd.DataFrame(
        [
            {
                "aggregation": str(args.aggregation),
                "n_rows": int(len(elbows)),
                "chosen_b": int(chosen_b),
                "min_elbow_b": int(elbows["elbow_b"].min()),
                "median_elbow_b": float(elbows["elbow_b"].median()),
                "max_elbow_b": int(elbows["elbow_b"].max()),
            }
        ]
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    elbows.to_csv(output_path.with_name(output_path.stem + "_details.csv"), index=False)
    summary.to_csv(output_path, index=False)
    print(f"Saved summary: {output_path}")
    print(f"Saved details: {output_path.with_name(output_path.stem + '_details.csv')}")
    print(f"Recommended B: {chosen_b}")


if __name__ == "__main__":
    main()
