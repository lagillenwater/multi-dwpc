#!/usr/bin/env python3
"""Build supported-only year variance and rank-stability summaries from existing outputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.replicate_analysis import (  # noqa: E402
    summarize_feature_variance,
    summarize_overall_variance,
    summarize_rank_stability,
)


FEATURE_KEYS = ["year", "control", "b", "go_id", "metapath"]


def _parse_top_k_values(arg: str) -> list[int | str]:
    values: list[int | str] = []
    for tok in str(arg).split(","):
        token = tok.strip()
        if not token:
            continue
        if token.lower() in {"all", "full"}:
            values.append("all")
        else:
            values.append(int(token))
    if not values:
        raise ValueError("Expected at least one top-k value")
    return values


def _load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path)


def _load_supported_key_frame(path: Path) -> pd.DataFrame:
    support_df = _load_required_csv(path)
    required = {"year", "go_id", "metapath", "supported"}
    missing = required - set(support_df.columns)
    if missing:
        raise ValueError(f"Support table missing required columns: {sorted(missing)}")
    out = support_df[support_df["supported"] == True].copy()  # noqa: E712
    out["year"] = out["year"].astype(int)
    out["go_id"] = out["go_id"].astype(str)
    out["metapath"] = out["metapath"].astype(str)
    return out[["year", "go_id", "metapath"]].drop_duplicates().reset_index(drop=True)


def build_supported_only_variance(
    analysis_dir: Path,
    supported_keys: pd.DataFrame,
    output_dir: Path,
) -> None:
    runs_df = _load_required_csv(analysis_dir / "all_runs_long.csv")
    runs_df["year"] = runs_df["year"].astype(int)
    runs_df["go_id"] = runs_df["go_id"].astype(str)
    runs_df["metapath"] = runs_df["metapath"].astype(str)
    filtered = runs_df.merge(supported_keys, on=["year", "go_id", "metapath"], how="inner")

    feature_df = summarize_feature_variance(filtered, FEATURE_KEYS)
    overall_df = summarize_overall_variance(
        feature_df,
        ["year", "control", "b"],
        runs_df=filtered,
        replicate_col="seed",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_dir / "all_runs_long_supported_only.csv", index=False)
    feature_df.to_csv(output_dir / "feature_variance_summary_supported_only.csv", index=False)
    overall_df.to_csv(output_dir / "overall_variance_summary_supported_only.csv", index=False)


def build_supported_only_rank_stability(
    analysis_dir: Path,
    supported_keys: pd.DataFrame,
    output_dir: Path,
    *,
    top_k: list[int | str],
    rbo_p: float,
) -> None:
    rank_df = _load_required_csv(analysis_dir / "metapath_rank_table.csv")
    rank_df["year"] = rank_df["year"].astype(int)
    rank_df["go_id"] = rank_df["go_id"].astype(str)
    rank_df["metapath"] = rank_df["metapath"].astype(str)
    filtered = rank_df.merge(supported_keys, on=["year", "go_id", "metapath"], how="inner")

    pairwise_df, go_summary_df, overall_df = summarize_rank_stability(
        filtered,
        outer_keys=["year", "control", "b", "go_id"],
        replicate_col="seed",
        feature_col="metapath",
        rank_col="metapath_rank",
        top_k=top_k,
        rbo_p=rbo_p,
    )
    if not overall_df.empty:
        overall_df = overall_df.rename(columns={"n_entities": "n_go_terms"})

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_dir / "metapath_rank_table_supported_only.csv", index=False)
    pairwise_df.to_csv(output_dir / "pairwise_metrics_supported_only.csv", index=False)
    go_summary_df.to_csv(output_dir / "go_term_stability_summary_supported_only.csv", index=False)
    overall_df.to_csv(output_dir / "overall_stability_summary_supported_only.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--support-path",
        default=str(REPO_ROOT / "output" / "year_direct_go_term_support.csv"),
    )
    parser.add_argument(
        "--variance-analysis-dir",
        default=str(REPO_ROOT / "output" / "year_null_variance_exp" / "year_null_variance_experiment"),
    )
    parser.add_argument(
        "--rank-analysis-dir",
        default=str(REPO_ROOT / "output" / "year_rank_stability_exp" / "year_rank_stability_experiment"),
    )
    parser.add_argument(
        "--variance-output-dir",
        default=str(REPO_ROOT / "output" / "year_null_variance_exp" / "year_null_variance_experiment" / "supported_only"),
    )
    parser.add_argument(
        "--rank-output-dir",
        default=str(REPO_ROOT / "output" / "year_rank_stability_exp" / "year_rank_stability_experiment" / "supported_only"),
    )
    parser.add_argument("--top-k-metapaths", default="5,10")
    parser.add_argument("--rbo-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supported_keys = _load_supported_key_frame(Path(args.support_path))
    build_supported_only_variance(
        Path(args.variance_analysis_dir),
        supported_keys,
        Path(args.variance_output_dir),
    )
    build_supported_only_rank_stability(
        Path(args.rank_analysis_dir),
        supported_keys,
        Path(args.rank_output_dir),
        top_k=_parse_top_k_values(args.top_k_metapaths),
        rbo_p=float(args.rbo_p),
    )
    print(f"Saved supported-only variance summaries under: {Path(args.variance_output_dir)}")
    print(f"Saved supported-only rank summaries under: {Path(args.rank_output_dir)}")


if __name__ == "__main__":
    main()
