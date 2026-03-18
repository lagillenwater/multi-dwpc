"""Year-domain helpers for explicit replicate summary analyses with B/seed resampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_KEYS = ["go_id", "metapath"]


def load_summary_bank(summaries_dir: Path) -> pd.DataFrame:
    files = sorted(Path(summaries_dir).glob("summary_*.csv"))
    if not files:
        raise FileNotFoundError(f"No year replicate summaries found under {summaries_dir}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def build_b_seed_runs(
    summary_df: pd.DataFrame,
    b_values: list[int],
    seeds: list[int],
) -> pd.DataFrame:
    real_df = summary_df[summary_df["control"].astype(str) == "real"].copy()
    if real_df.empty:
        raise ValueError("No real year summary rows found")
    real_df = real_df[["year", *FEATURE_KEYS, "mean_score"]].rename(
        columns={"mean_score": "real_mean_score"}
    )

    null_df = summary_df[summary_df["control"].astype(str).isin(["permuted", "random"])].copy()
    if null_df.empty:
        raise ValueError("No explicit year null summary rows found")

    available = (
        null_df.groupby(["year", "control"])["replicate"]
        .nunique()
        .to_dict()
    )
    max_b = max(int(b) for b in b_values)
    for (year, control), count in available.items():
        if int(count) < max_b:
            raise ValueError(
                f"Requested max B={max_b} but only {count} {control} replicate summaries "
                f"are available for year={year}."
            )

    rows = []
    for (year, control), group in null_df.groupby(["year", "control"], sort=True):
        rep_ids = sorted(group["replicate"].astype(int).unique().tolist())
        rep_ids_arr = np.asarray(rep_ids, dtype=int)
        for b in sorted(int(x) for x in b_values):
            for seed in sorted(int(x) for x in seeds):
                rng = np.random.RandomState(seed)
                selected = rng.choice(rep_ids_arr, size=b, replace=False)
                subset = group[group["replicate"].isin(selected)].copy()
                agg = (
                    subset.groupby(FEATURE_KEYS, as_index=False)["mean_score"]
                    .mean()
                    .rename(columns={"mean_score": "null_mean_score"})
                )
                agg.insert(0, "year", int(year))
                merged = agg.merge(real_df, on=["year", *FEATURE_KEYS], how="inner")
                merged["control"] = str(control)
                merged["b"] = int(b)
                merged["seed"] = int(seed)
                merged["diff"] = merged["real_mean_score"] - merged["null_mean_score"]
                rows.append(merged)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "year",
            "control",
            "b",
            "seed",
            *FEATURE_KEYS,
            "real_mean_score",
            "null_mean_score",
            "diff",
        ]
    ]
