"""Year-domain helpers for explicit replicate summary analyses with B/seed resampling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.replicate_analysis import build_b_seed_runs as build_b_seed_runs_generic
from src.replicate_workspace import (
    load_manifest as load_workspace_manifest,
    load_summary_bank as load_workspace_summary_bank,
    load_summary_bank_from_manifest,
    resolve_manifest_path,
)

FEATURE_KEYS = ["go_id", "metapath"]
YEAR_MANIFEST_COLUMNS = [
    "domain",
    "name",
    "control",
    "replicate",
    "year",
    "source_path",
    "result_path",
    "summary_path",
]
YEAR_SUMMARY_COLUMNS = ["domain", "name", "control", "replicate", "year", *FEATURE_KEYS, "mean_score"]


def load_summary_bank(workspace_path: Path | str) -> pd.DataFrame:
    workspace_path = Path(workspace_path)
    manifest_path = resolve_manifest_path(workspace_path)
    if manifest_path.exists():
        manifest_df = load_workspace_manifest(
            workspace_path,
            required_cols=YEAR_MANIFEST_COLUMNS,
        )
        return load_summary_bank_from_manifest(
            manifest_df,
            required_summary_cols=YEAR_SUMMARY_COLUMNS,
        )
    return load_workspace_summary_bank(
        workspace_path,
        required_summary_cols=YEAR_SUMMARY_COLUMNS,
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
