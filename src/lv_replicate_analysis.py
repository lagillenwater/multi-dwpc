"""LV analysis helpers built on explicit replicate summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.lv_explicit_replicates import load_manifest


FEATURE_KEYS = ["lv_id", "target_set_id", "target_set_label", "node_type", "metapath"]


def load_summary_bank(output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    manifest = load_manifest(output_dir)
    if manifest.empty:
        raise ValueError(f"No LV replicate artifacts discovered under {output_dir}")

    frames = []
    for row in manifest.itertuples(index=False):
        summary_path = Path(row.summary_path)
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing LV replicate summary: {summary_path}. "
                "Run scripts/lv_compute_replicate_summaries.py first."
            )
        frames.append(pd.read_csv(summary_path))
    return pd.concat(frames, ignore_index=True)


def build_b_seed_runs(
    summary_df: pd.DataFrame,
    b_values: list[int],
    seeds: list[int],
) -> pd.DataFrame:
    real_df = summary_df[summary_df["control"].astype(str) == "real"].copy()
    if real_df.empty:
        raise ValueError("No real LV summary rows found")
    real_df = real_df[FEATURE_KEYS + ["mean_score"]].rename(columns={"mean_score": "real_mean_score"})

    null_df = summary_df[summary_df["control"].astype(str).isin(["permuted", "random"])].copy()
    if null_df.empty:
        raise ValueError("No explicit LV null summary rows found")

    available = (
        null_df.groupby("control")["replicate"]
        .nunique()
        .to_dict()
    )
    max_b = max(int(b) for b in b_values)
    for control, count in available.items():
        if int(count) < max_b:
            raise ValueError(
                f"Requested max B={max_b} but only {count} {control} replicate summaries are available."
            )

    rows = []
    for control, control_df in null_df.groupby("control", sort=True):
        rep_ids = sorted(control_df["replicate"].astype(int).unique().tolist())
        rep_ids_arr = np.asarray(rep_ids, dtype=int)
        for b in sorted(int(x) for x in b_values):
            for seed in sorted(int(x) for x in seeds):
                rng = np.random.RandomState(seed)
                selected = rng.choice(rep_ids_arr, size=b, replace=False)
                subset = control_df[control_df["replicate"].isin(selected)].copy()
                agg = (
                    subset.groupby(FEATURE_KEYS, as_index=False)["mean_score"]
                    .mean()
                    .rename(columns={"mean_score": "null_mean_score"})
                )
                merged = agg.merge(real_df, on=FEATURE_KEYS, how="inner")
                merged["control"] = str(control)
                merged["b"] = int(b)
                merged["seed"] = int(seed)
                merged["diff"] = merged["real_mean_score"] - merged["null_mean_score"]
                rows.append(merged)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["control", "b", "seed", *FEATURE_KEYS, "real_mean_score", "null_mean_score", "diff"]]
