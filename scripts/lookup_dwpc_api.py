"""
Compute DWPC values via the connectivity-search-backend API.

This script runs the API-based DWPC workflow used for publication figures.
It expects input CSVs with neo4j_source_id and neo4j_target_id columns.
"""

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

os.environ.setdefault("CONNECTIVITY_SEARCH_API", "http://localhost:8015")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.dwpc_api import run_metapaths_for_df, validate_parquet_files


BASE_NAME = "all_GO_positive_growth"
MAX_CONCURRENCY = 120
RETRIES = 10
BACKOFF_FIRST = 10.0
ENABLE_FINAL_VALIDATION = True
FINAL_MIN_COMPLETION_RATE = 0.2
CLEANUP_PARQUET = True


def load_neo4j_mappings(data_dir: Path) -> tuple[dict, dict]:
    """Load GO and gene identifier mappings to Neo4j IDs."""
    gene_map = pd.read_csv(data_dir / "neo4j_gene_mapping.csv")
    bp_map = pd.read_csv(data_dir / "neo4j_bp_mapping.csv")
    entrez_to_neo4j = dict(zip(gene_map["identifier"], gene_map["neo4j_id"]))
    go_to_neo4j = dict(zip(bp_map["identifier"], bp_map["neo4j_id"]))
    return entrez_to_neo4j, go_to_neo4j


def attach_neo4j_ids(
    df: pd.DataFrame,
    dataset: dict,
    entrez_to_neo4j: dict,
    go_to_neo4j: dict,
) -> pd.DataFrame:
    """Ensure Neo4j ID columns exist by mapping from GO/gene identifiers."""
    df = df.copy()
    required = {dataset["col_source"], dataset["col_target"]}
    missing = required - set(df.columns)

    if "neo4j_source_id" in missing and "go_id" in df.columns:
        df["neo4j_source_id"] = df["go_id"].map(go_to_neo4j)

    if "neo4j_target_id" in missing and "entrez_gene_id" in df.columns:
        df["neo4j_target_id"] = df["entrez_gene_id"].map(entrez_to_neo4j)

    if "neo4j_pseudo_target_id" in missing and "pseudo_gene_id" in df.columns:
        df["neo4j_pseudo_target_id"] = df["pseudo_gene_id"].map(entrez_to_neo4j)

    required = {dataset["col_source"], dataset["col_target"]}
    missing = required - set(df.columns)
    if missing:
        return df

    before = len(df)
    df = df.dropna(subset=[dataset["col_source"], dataset["col_target"]])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} rows missing Neo4j IDs for {dataset['name']}")

    df[dataset["col_source"]] = df[dataset["col_source"]].astype(int)
    df[dataset["col_target"]] = df[dataset["col_target"]].astype(int)
    return df


def collect_datasets(repo_root: Path) -> list[dict]:
    """Collect datasets available for API computation."""
    base_dir = repo_root / "output"
    datasets = []

    real_2016 = base_dir / "intermediate" / f"hetio_bppg_{BASE_NAME}_filtered.csv"
    real_2024 = base_dir / "intermediate" / f"hetio_bppg_{BASE_NAME}_2024_filtered.csv"

    if real_2016.exists():
        datasets.append({
            "name": f"{BASE_NAME}_2016_real",
            "path": real_2016,
            "col_source": "neo4j_source_id",
            "col_target": "neo4j_target_id",
            "year": 2016,
            "type": "real",
        })

    if real_2024.exists():
        datasets.append({
            "name": f"{BASE_NAME}_2024_real",
            "path": real_2024,
            "col_source": "neo4j_source_id",
            "col_target": "neo4j_target_id",
            "year": 2024,
            "type": "real",
        })

    for year in (2016, 2024):
        perm_dir = base_dir / "permutations" / f"{BASE_NAME}_{year}"
        perm_files = sorted(perm_dir.glob("perm_*.csv"))
        if perm_files:
            perm_file = perm_files[0]
            perm_num = perm_file.stem.split("_")[1]
            datasets.append({
                "name": f"{BASE_NAME}_{year}_perm_{perm_num}",
                "path": perm_file,
                "col_source": "neo4j_source_id",
                "col_target": "neo4j_target_id",
                "year": year,
                "type": "permuted",
            })

        random_dir = base_dir / "random_samples" / f"{BASE_NAME}_{year}"
        random_files = sorted(random_dir.glob("random_*.csv"))
        if random_files:
            random_file = random_files[0]
            random_num = random_file.stem.split("_")[1]
            datasets.append({
                "name": f"{BASE_NAME}_{year}_random_{random_num}",
                "path": random_file,
                "col_source": "neo4j_source_id",
                "col_target": "neo4j_pseudo_target_id",
                "year": year,
                "type": "random",
            })

    return datasets


def validate_columns(df: pd.DataFrame, dataset: dict) -> None:
    """Ensure dataset has required neo4j columns."""
    required = {dataset["col_source"], dataset["col_target"]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {dataset['path']}: {sorted(missing)}"
        )


def main() -> None:
    """Run the API-based DWPC computation pipeline."""
    output_root = repo_root / "output" / "dwpc_com" / BASE_NAME
    results_dir = output_root / "results"
    histograms_dir = output_root / "histograms"
    parquet_dir = output_root / "metapaths_parquet"

    results_dir.mkdir(parents=True, exist_ok=True)
    histograms_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    datasets = collect_datasets(repo_root)
    if not datasets:
        raise FileNotFoundError(
            f"No datasets found for {BASE_NAME}. "
            "Ensure parents_GO_postive_growth CSVs exist."
        )

    entrez_to_neo4j, go_to_neo4j = load_neo4j_mappings(repo_root / "data")

    print("=" * 80)
    print(f"STARTING API DWPC: {len(datasets)} datasets")
    print("=" * 80)

    batch_start = time.perf_counter()
    batch_summary = []

    for idx, dataset in enumerate(datasets, start=1):
        dataset_start = time.perf_counter()
        print(f"\n{'=' * 80}")
        print(f"Dataset {idx}/{len(datasets)}: {dataset['name']}")
        print(f"  Year: {dataset['year']}, Type: {dataset['type']}")
        print(f"  Path: {dataset['path']}")
        print(f"{'=' * 80}\n")

        csv_path = results_dir / f"res_{dataset['name']}.csv"
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                if len(existing_df) > 0:
                    print(f"CHECKPOINT: Dataset already processed ({len(existing_df):,} rows)")
                    batch_summary.append({
                        "dataset_number": idx,
                        "dataset_name": dataset["name"],
                        "year": dataset["year"],
                        "type": dataset["type"],
                        "n_input_pairs": None,
                        "n_output_rows": len(existing_df),
                        "time_seconds": 0,
                        "status": "skipped (already complete)",
                    })
                    continue
            except Exception as exc:
                print(f"Warning: Could not read existing CSV: {exc}")
                print("  Will reprocess dataset...\n")

        df = pd.read_csv(dataset["path"])
        df = attach_neo4j_ids(df, dataset, entrez_to_neo4j, go_to_neo4j)
        validate_columns(df, dataset)
        n_pairs = len(df)
        n_unique_pairs = len(
            df[[dataset["col_source"], dataset["col_target"]]]
            .dropna()
            .drop_duplicates()
        )
        print(f"Loaded {n_pairs:,} pairs ({n_unique_pairs:,} unique)\n")

        summary = asyncio.run(
            run_metapaths_for_df(
                df,
                col_source=dataset["col_source"],
                col_target=dataset["col_target"],
                base_out_dir=parquet_dir,
                group=dataset["name"],
                clear_group=False,
                max_concurrency=MAX_CONCURRENCY,
                retries=RETRIES,
                backoff_first=BACKOFF_FIRST,
            )
        )

        group_dir = parquet_dir / dataset["name"]
        parquet_files = sorted(group_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {group_dir}")

        if ENABLE_FINAL_VALIDATION:
            print("Validating parquet metadata...", flush=True)
            validate_parquet_files(
                group_dir,
                n_unique_pairs,
                min_completion_rate=FINAL_MIN_COMPLETION_RATE,
                show_progress=True,
            )

        print(f"Merging {len(parquet_files):,} parquet files...", flush=True)
        frames = []
        for fp in tqdm(parquet_files, desc="Loading parquet files", unit="file"):
            frames.append(pd.read_parquet(fp))
        res_df = pd.concat(frames, ignore_index=True)
        res_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        if "dwpc" in res_df.columns:
            dwpc_vals = res_df.loc[res_df["dwpc"] > 0, "dwpc"].dropna()
            if len(dwpc_vals) > 0:
                dwpc_mean = dwpc_vals.mean()
                print("Creating histogram...", flush=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(dwpc_vals, bins=50, edgecolor="black", linewidth=0.5)
                ax.axvline(dwpc_mean, color="red", linestyle="--", linewidth=1.2,
                           label=f"Mean = {dwpc_mean:.2f}")
                ax.set_title(f"DWPC Distribution: {dataset['name']}", fontsize=12)
                ax.set_xlabel("DWPC", fontsize=11)
                ax.set_ylabel("Count", fontsize=11)
                ax.legend(fontsize=10)
                ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
                plt.tight_layout()
                hist_path = histograms_dir / f"hist_{dataset['name']}.png"
                plt.savefig(hist_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved histogram: {hist_path}")
            else:
                print("Skipping histogram: no positive DWPC values")

        dataset_time = time.perf_counter() - dataset_start
        batch_summary.append({
            "dataset_number": idx,
            "dataset_name": dataset["name"],
            "year": dataset["year"],
            "type": dataset["type"],
            "n_input_pairs": n_pairs,
            "n_unique_pairs": n_unique_pairs,
            "n_output_rows": len(res_df),
            "time_seconds": dataset_time,
            "status": "completed",
        })

        print(f"\nCompleted {dataset['name']} in {dataset_time/60:.1f} minutes")

        if CLEANUP_PARQUET:
            shutil.rmtree(group_dir, ignore_errors=True)
            print(f"Cleaned up parquet files: {group_dir}")

    batch_time = time.perf_counter() - batch_start
    print(f"\n{'=' * 80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"  Total time: {batch_time/3600:.2f} hours")
    print(f"  Datasets processed: {len(batch_summary)}")
    print(f"  Results: {results_dir}")
    print(f"  Histograms: {histograms_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
