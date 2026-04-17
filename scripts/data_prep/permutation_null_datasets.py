#!/usr/bin/env python3
"""Generate degree-preserving year permutation datasets."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.bipartite_nulls import degree_preserving_permutations  # noqa: E402


def _replicate_ids() -> list[int]:
    raw = os.getenv("PERMUTATION_IDS", "").strip()
    if raw:
        ids = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    else:
        n = int(os.getenv("N_PERMUTATIONS", "1"))
        ids = list(range(1, n + 1))
    if not ids:
        raise ValueError("No permutation replicate ids configured")
    return ids


def _load_year_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"go_id", "entrez_gene_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    if len(df[["go_id", "entrez_gene_id"]].drop_duplicates()) != len(df):
        raise ValueError(f"Expected binary GO-gene edge list in {path}; found duplicate GO-gene rows")
    return df


def _validate(real_df: pd.DataFrame, perm_df: pd.DataFrame, year: int, rep_id: int) -> None:
    real_sizes = real_df.groupby("go_id").size().sort_index()
    perm_sizes = perm_df.groupby("go_id").size().sort_index()
    if not real_sizes.equals(perm_sizes):
        raise ValueError(f"GO term sizes changed for year={year}, replicate={rep_id}")

    real_deg = real_df.groupby("entrez_gene_id")["go_id"].nunique().sort_index()
    perm_deg = perm_df.groupby("entrez_gene_id")["go_id"].nunique().sort_index()
    if not real_deg.equals(perm_deg):
        raise ValueError(f"Gene annotation degrees changed for year={year}, replicate={rep_id}")

    if len(perm_df[["go_id", "entrez_gene_id"]].drop_duplicates()) != len(perm_df):
        raise ValueError(f"Duplicate GO-gene edges introduced for year={year}, replicate={rep_id}")


def _generate_for_year(real_df: pd.DataFrame, year: int, out_dir: Path, replicate_ids: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rep_id in replicate_ids:
        perm_df = degree_preserving_permutations(
            edge_df=real_df,
            source_col="go_id",
            target_col="entrez_gene_id",
            n_permutations=1,
            random_state=42 + int(rep_id) - 1,
            n_swap_attempts_per_edge=10,
        )[0]
        _validate(real_df, perm_df, year=year, rep_id=rep_id)
        path = out_dir / f"perm_{rep_id:03d}.csv"
        perm_df.to_csv(path, index=False)
        print(f"[saved] {path}")


def main() -> None:
    replicate_ids = _replicate_ids()
    real_2016 = _load_year_df(REPO_ROOT / "output/intermediate/hetio_bppg_all_GO_positive_growth_filtered.csv")
    real_2024 = _load_year_df(REPO_ROOT / "output/intermediate/hetio_bppg_all_GO_positive_growth_2024_filtered.csv")

    _generate_for_year(
        real_df=real_2016,
        year=2016,
        out_dir=REPO_ROOT / "output/permutations/all_GO_positive_growth_2016",
        replicate_ids=replicate_ids,
    )
    _generate_for_year(
        real_df=real_2024,
        year=2024,
        out_dir=REPO_ROOT / "output/permutations/all_GO_positive_growth_2024",
        replicate_ids=replicate_ids,
    )


if __name__ == "__main__":
    main()
