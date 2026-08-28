# scripts/experiments/tier0_list_subsample_metapaths.py
"""Write the distinct metapaths used by the Tier 0 stratified subsample --
the exact set the Alpine prewarm stage must have DWPC matrices for.
feature_idx is globally unique across the manifest, so filtering on it
alone selects the subsample's features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.tier0.subsample import select_stratified_subsample  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--substrate-dir", required=True, type=Path)
    p.add_argument("--per-cell-max", type=int, default=15)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    sub = select_stratified_subsample(a.substrate_dir, a.per_cell_max, a.random_state)
    manifest = pd.read_csv(a.substrate_dir / "feature_manifest.csv")
    mps = sorted(
        manifest[manifest["feature_idx"].isin(sub["feature_idx"])]["metapath"].unique()
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(mps) + "\n")
    print(f"{len(mps)} metapaths -> {a.out}")


if __name__ == "__main__":
    main()
