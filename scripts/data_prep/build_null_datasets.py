#!/usr/bin/env python3
"""Generate year null datasets via degree-preserving permutation or promiscuity-matched random sampling.

Replaces the prior `permutation_null_datasets.py` and `random_null_datasets.py`.

Usage:
    python scripts/data_prep/build_null_datasets.py --method permutation
    python scripts/data_prep/build_null_datasets.py --method random

Replicate ids are controlled per-method via env vars (preserved from the original scripts):
    permutation: PERMUTATION_IDS (csv) or N_PERMUTATIONS (default 1)
    random:      RANDOM_SAMPLE_IDS (csv) or N_RANDOM_SAMPLES (default 1)

Outputs (paths preserved exactly so downstream consumers are unaffected):
    permutation: output/permutations/all_GO_positive_growth_{year}/perm_{id:03d}.csv
    random:      output/random_samples/all_GO_positive_growth_{year}/random_{id:03d}.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.bipartite_nulls import (  # noqa: E402
    degree_preserving_permutations,
    generate_promiscuity_matched_samples,
)

PROMISCUITY_TOLERANCE = 2
REAL_2016_PATH = REPO_ROOT / "output/intermediate/hetio_bppg_all_GO_positive_growth_filtered.csv"
REAL_2024_PATH = REPO_ROOT / "output/intermediate/hetio_bppg_all_GO_positive_growth_2024_filtered.csv"
ALL_2016_PATH = REPO_ROOT / "output/intermediate/hetio_bppg_2016_stable.csv"
ALL_2024_PATH = REPO_ROOT / "output/intermediate/upd_go_bp_2024_added.csv"


def _replicate_ids(env_ids: str, env_count: str) -> list[int]:
    raw = os.getenv(env_ids, "").strip()
    if raw:
        ids = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    else:
        n = int(os.getenv(env_count, "1"))
        ids = list(range(1, n + 1))
    if not ids:
        raise ValueError(f"No replicate ids configured (set {env_ids} or {env_count})")
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


def _load_all_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["go_id", "entrez_gene_id"]].drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------


def _validate_permutation(real_df: pd.DataFrame, perm_df: pd.DataFrame, year: int, rep_id: int) -> None:
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


def _generate_permutation_for_year(real_df: pd.DataFrame, year: int, out_dir: Path, replicate_ids: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rep_id in replicate_ids:
        perm_df = degree_preserving_permutations(
            edge_df=real_df,
            source_col="go_id",
            target_col="entrez_gene_id",
            n_permutations=1,
            random_state=42 + int(rep_id),
            n_swap_attempts_per_edge=10,
        )[0]
        _validate_permutation(real_df, perm_df, year=year, rep_id=rep_id)
        path = out_dir / f"perm_{rep_id:03d}.csv"
        perm_df.to_csv(path, index=False)
        print(f"[saved] {path}")


def _run_permutation(replicate_ids: list[int]) -> None:
    real_2016 = _load_year_df(REAL_2016_PATH)
    real_2024 = _load_year_df(REAL_2024_PATH)
    _generate_permutation_for_year(
        real_df=real_2016,
        year=2016,
        out_dir=REPO_ROOT / "output/permutations/all_GO_positive_growth_2016",
        replicate_ids=replicate_ids,
    )
    _generate_permutation_for_year(
        real_df=real_2024,
        year=2024,
        out_dir=REPO_ROOT / "output/permutations/all_GO_positive_growth_2024",
        replicate_ids=replicate_ids,
    )


# ---------------------------------------------------------------------------
# Random (promiscuity-matched)
# ---------------------------------------------------------------------------


def _validate_random(
    real_df: pd.DataFrame, rand_df: pd.DataFrame, all_df: pd.DataFrame, year: int, rep_id: int
) -> None:
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


def _generate_random_for_year(
    real_df: pd.DataFrame, all_df: pd.DataFrame, year: int, out_dir: Path, replicate_ids: list[int]
) -> None:
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
        _validate_random(real_df, rand_df, all_df, year=year, rep_id=rep_id)
        path = out_dir / f"random_{rep_id:03d}.csv"
        rand_df.to_csv(path, index=False)
        print(f"[saved] {path}")


def _run_random(replicate_ids: list[int]) -> None:
    real_2016 = _load_year_df(REAL_2016_PATH)
    real_2024 = _load_year_df(REAL_2024_PATH)
    all_2016 = _load_all_annotations(ALL_2016_PATH)
    all_2024 = _load_all_annotations(ALL_2024_PATH)
    _generate_random_for_year(
        real_df=real_2016,
        all_df=all_2016,
        year=2016,
        out_dir=REPO_ROOT / "output/random_samples/all_GO_positive_growth_2016",
        replicate_ids=replicate_ids,
    )
    _generate_random_for_year(
        real_df=real_2024,
        all_df=all_2024,
        year=2024,
        out_dir=REPO_ROOT / "output/random_samples/all_GO_positive_growth_2024",
        replicate_ids=replicate_ids,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=["permutation", "random"],
        help="Null generation method.",
    )
    args = parser.parse_args()

    if args.method == "permutation":
        _run_permutation(_replicate_ids(env_ids="PERMUTATION_IDS", env_count="N_PERMUTATIONS"))
    else:
        _run_random(_replicate_ids(env_ids="RANDOM_SAMPLE_IDS", env_count="N_RANDOM_SAMPLES"))


if __name__ == "__main__":
    main()
