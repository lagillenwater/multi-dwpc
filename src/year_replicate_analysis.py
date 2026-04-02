"""Year-domain helpers for explicit replicate summary analyses with B/seed resampling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.replicate_analysis import build_b_seed_runs as build_b_seed_runs_generic
from src.replicate_workspace import load_summary_bank as load_workspace_summary_bank

FEATURE_KEYS = ["go_id", "metapath"]


def load_summary_bank(summaries_dir: Path) -> pd.DataFrame:
    return load_workspace_summary_bank(
        summaries_dir,
        required_summary_cols=["domain", "name", "control", "replicate", "year", *FEATURE_KEYS, "mean_score"],
    )


def build_b_seed_runs(
    summary_df: pd.DataFrame,
    b_values: list[int],
    seeds: list[int],
) -> pd.DataFrame:
    return build_b_seed_runs_generic(
        summary_df,
        b_values=b_values,
        seeds=seeds,
        join_keys=["year", *FEATURE_KEYS],
        replicate_pool_keys=["year", "control"],
    )
