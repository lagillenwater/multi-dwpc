"""LV analysis helpers built on explicit replicate summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.lv_explicit_replicates import load_manifest
from src.replicate_analysis import build_b_seed_runs as build_b_seed_runs_generic
from src.replicate_workspace import (
    load_manifest as load_workspace_manifest,
    load_summary_bank_from_manifest,
)


FEATURE_KEYS = ["lv_id", "target_id", "target_name", "node_type", "metapath"]
LV_MANIFEST_COLUMNS = [
    "domain",
    "name",
    "control",
    "replicate",
    "source_path",
    "result_path",
    "summary_path",
]
LV_SUMMARY_COLUMNS = ["domain", "name", "control", "replicate", *FEATURE_KEYS, "mean_score"]


def load_summary_bank(output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "replicate_manifest.csv"
    if manifest_path.exists():
        manifest = load_workspace_manifest(
            manifest_path,
            required_cols=LV_MANIFEST_COLUMNS,
        )
    else:
        manifest = load_manifest(output_dir)
    if manifest.empty:
        raise ValueError(f"No LV replicate artifacts discovered under {output_dir}")
    return load_summary_bank_from_manifest(
        manifest,
        required_summary_cols=LV_SUMMARY_COLUMNS,
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
        join_keys=FEATURE_KEYS,
        replicate_pool_keys=["control"],
    )
