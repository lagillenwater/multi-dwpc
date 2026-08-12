"""Selects a stratified subsample of null_streaming_summary.csv rows spanning
L=2 vs L>=3 and floor-pinned (exceed_count==0) vs not, restricted to the
`random` null type -- see this plan's Global Constraints for why `permuted`
is out of scope for LV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dwpc_validation import metapath_length


def select_stratified_subsample(
    substrate_dir: Path, per_cell_max: int, random_state: int
) -> pd.DataFrame:
    substrate_dir = Path(substrate_dir)
    df = pd.read_csv(substrate_dir / "null_streaming_summary.csv")
    df = df[df["null_type"] == "random"].copy()

    df["length"] = df["metapath"].apply(metapath_length)
    df["is_l2"] = df["length"] == 2
    df["is_floor_pinned"] = df["exceed_count"] == 0

    parts = []
    for is_l2 in (True, False):
        for pinned in (True, False):
            cell = df[(df["is_l2"] == is_l2) & (df["is_floor_pinned"] == pinned)]
            if cell.empty:
                continue
            n = min(per_cell_max, len(cell))
            parts.append(cell.sample(n=n, random_state=random_state))

    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)
