#!/usr/bin/env python3
"""Generate promiscuity-matched year random-control datasets."""

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
from src.bipartite_nulls import generate_promiscuity_matched_samples  # noqa: E402


PROMISCUITY_TOLERANCE = 2


def _replicate_ids() -> list[int]:
    raw = os.getenv("RANDOM_SAMPLE_IDS", "").strip()
    if raw:
        ids = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    else:
        n = int(os.getenv("N_RANDOM_SAMPLES", "1"))
        ids = list(range(1, n + 1))
    if not ids:
        raise ValueError("No random replicate ids configured")
    return ids


def _load_real_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"go_id", "entrez_gene_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    if len(df[["go_id", "entrez_gene_id"]].drop_duplicates()) != len(df):
        raise ValueError(f"Expected binary GO-gene edge list in {path}; found duplicate GO-gene rows")
    return df


def _load_all_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["go_id", "entrez_gene_id"]].drop_duplicates().reset_index(drop=True)


def _validate(real_df: pd.DataFrame, rand_df: pd.DataFrame, all_df: pd.DataFrame, year: int, rep_id: int) -> None:
    real_sizes = real_df.groupby("go_id").size().sort_index()
    rand_sizes = rand_df.groupby("go_id").size().sort_index()
    if not real_sizes.equals(rand_sizes):
        raise ValueError(f"GO term sizes changed for year={year}, replicate={rep_id}")

    real_sets = real_df.groupby("go_id")["entrez_gene_id"].apply(set).to_dict()
    rand_sets = rand_df.groupby("go_id")["entrez_gene_id"].apply(set).to_dict()
    overlap = sum(len(real_sets[go_id] & rand_sets.get(go_id, set())) for go_id in real_sets)
    if overlap != 0:
        raise ValueError(f"Random control overlaps with real genes for year={year}, replicate={rep_id}")

    if len(rand_df[["go_id", "entrez_gene_id"]].drop_duplicates()) != len(rand_df):
        raise ValueError(f"Duplicate GO-gene edges introduced for year={year}, replicate={rep_id}")

    annotated = set(all_df["entrez_gene_id"].astype(str))
    random_genes = set(rand_df["entrez_gene_id"].astype(str))
    if not random_genes.issubset(annotated):
        raise ValueError(f"Unannotated genes appeared in random control for year={year}, replicate={rep_id}")


def _generate_for_year(real_df: pd.DataFrame, all_df: pd.DataFrame, year: int, out_dir: Path, replicate_ids: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_universe = sorted(all_df["entrez_gene_id"].unique().tolist())
    for rep_id in replicate_ids:
        rand_df = generate_promiscuity_matched_samples(
            edge_df=real_df,
            all_annotations_df=all_df,
            source_col="go_id",
            target_col="entrez_gene_id",
            target_universe=target_universe,
            promiscuity_tolerance=PROMISCUITY_TOLERANCE,
            random_state=42 + int(rep_id),
            include_match_metadata=True,
        )
        rand_df["entrez_gene_id"] = rand_df["entrez_gene_id"].astype(real_df["entrez_gene_id"].dtype)
        _validate(real_df, rand_df, all_df, year=year, rep_id=rep_id)
        path = out_dir / f"random_{rep_id:03d}.csv"
        rand_df.to_csv(path, index=False)
        print(f"[saved] {path}")


def main() -> None:
    replicate_ids = _replicate_ids()
    real_2016 = _load_real_df(REPO_ROOT / "output/intermediate/hetio_bppg_all_GO_positive_growth_filtered.csv")
    real_2024 = _load_real_df(REPO_ROOT / "output/intermediate/hetio_bppg_all_GO_positive_growth_2024_filtered.csv")
    all_2016 = _load_all_annotations(REPO_ROOT / "output/intermediate/hetio_bppg_2016_stable.csv")
    all_2024 = _load_all_annotations(REPO_ROOT / "output/intermediate/upd_go_bp_2024_added.csv")

    _generate_for_year(
        real_df=real_2016,
        all_df=all_2016,
        year=2016,
        out_dir=REPO_ROOT / "output/random_samples/all_GO_positive_growth_2016",
        replicate_ids=replicate_ids,
    )
    _generate_for_year(
        real_df=real_2024,
        all_df=all_2024,
        year=2024,
        out_dir=REPO_ROOT / "output/random_samples/all_GO_positive_growth_2024",
        replicate_ids=replicate_ids,
    )


if __name__ == "__main__":
    main()
