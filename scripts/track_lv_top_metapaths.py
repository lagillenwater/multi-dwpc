#!/usr/bin/env python3
"""Track LV top-ranked metapaths across seeds and null replicate counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        default="output/lv_experiment/lv_rank_stability_experiment",
        help="Directory containing metapath_rank_table.csv",
    )
    parser.add_argument("--reference-seed", type=int, default=11)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--reference-b", type=int, default=None, help="Reference B; defaults to max available B")
    parser.add_argument("--control", default=None)
    parser.add_argument("--lv-id", default=None)
    parser.add_argument("--target-set-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    rank_path = analysis_dir / "metapath_rank_table.csv"
    if not rank_path.exists():
        raise FileNotFoundError(f"Required input file not found: {rank_path}")

    df = pd.read_csv(rank_path)
    required = {"b", "seed", "lv_id", "target_set_id", "metapath", "metapath_rank"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{rank_path} is missing required columns: {missing}")
    if "control" not in df.columns:
        df["control"] = "combined"

    df["b"] = df["b"].astype(int)
    df["seed"] = df["seed"].astype(int)
    ref_b = int(args.reference_b) if args.reference_b is not None else int(df["b"].max())

    if args.control is not None:
        df = df[df["control"].astype(str) == str(args.control)].copy()
    if args.lv_id is not None:
        df = df[df["lv_id"].astype(str) == str(args.lv_id)].copy()
    if args.target_set_id is not None:
        df = df[df["target_set_id"].astype(str) == str(args.target_set_id)].copy()
    if df.empty:
        raise ValueError("No rows remain after applying filters")

    ref = (
        df[(df["seed"] == int(args.reference_seed)) & (df["b"] == ref_b)]
        .sort_values(["control", "lv_id", "target_set_id", "metapath_rank"])
        .groupby(["control", "lv_id", "target_set_id"], group_keys=False)
        .head(int(args.top_n))
        [["control", "lv_id", "target_set_id", "metapath", "metapath_rank"]]
        .rename(columns={"metapath_rank": "ref_rank"})
    )
    if ref.empty:
        raise ValueError(
            f"No reference rows found for seed={args.reference_seed} and b={ref_b}"
        )

    track = (
        df.merge(ref, on=["control", "lv_id", "target_set_id", "metapath"], how="inner")
        .sort_values(["control", "lv_id", "target_set_id", "ref_rank", "b", "seed"])
        .reset_index(drop=True)
    )

    suffix_parts = [f"seed{int(args.reference_seed)}", f"b{ref_b}", f"top{int(args.top_n)}"]
    if args.control:
        suffix_parts.append(str(args.control))
    if args.lv_id:
        suffix_parts.append(str(args.lv_id))
    if args.target_set_id:
        suffix_parts.append(str(args.target_set_id))
    out_name = "top_metapath_rank_trajectories_" + "_".join(suffix_parts) + ".csv"
    out_path = analysis_dir / out_name
    track.to_csv(out_path, index=False)
    print(out_path)
    print(
        track[
            [
                "control",
                "lv_id",
                "target_set_id",
                "metapath",
                "ref_rank",
                "b",
                "seed",
                "metapath_rank",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
