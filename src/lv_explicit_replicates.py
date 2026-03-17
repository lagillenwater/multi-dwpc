"""Explicit LV null-replicate generation and summary utilities."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.bipartite_nulls import (
    degree_preserving_permutations,
    generate_promiscuity_matched_samples,
)

ARTIFACT_DIRNAME = "replicate_artifacts"
SUMMARY_DIRNAME = "replicate_summaries"
MANIFEST_FILENAME = "replicate_manifest.csv"


LV_ARTIFACT_RE = re.compile(r"^lv_(real|permuted|random)(?:_(\d+))?\.csv$")


def artifact_dir(output_dir: Path) -> Path:
    return Path(output_dir) / ARTIFACT_DIRNAME


def summary_dir(output_dir: Path) -> Path:
    return Path(output_dir) / SUMMARY_DIRNAME


def manifest_path(output_dir: Path) -> Path:
    return Path(output_dir) / MANIFEST_FILENAME


def _required_prepare_files(output_dir: Path) -> list[Path]:
    return [
        Path(output_dir) / "lv_top_genes.csv",
        Path(output_dir) / "lv_target_map.csv",
        Path(output_dir) / "feature_manifest.csv",
        Path(output_dir) / "gene_feature_scores.npy",
        Path(output_dir) / "gene_ids.npy",
        Path(output_dir) / "real_feature_scores.csv",
    ]


def assert_prepared(output_dir: Path) -> None:
    missing = [str(path) for path in _required_prepare_files(output_dir) if not path.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "LV explicit replicate workflow requires prepared LV artifacts. Missing:\n"
            f"  - {joined}\n"
            "Run the prepare stage first."
        )


def load_base_edges(output_dir: Path) -> pd.DataFrame:
    assert_prepared(output_dir)
    top_genes = pd.read_csv(Path(output_dir) / "lv_top_genes.csv")
    lv_map = pd.read_csv(Path(output_dir) / "lv_target_map.csv")
    base = top_genes.merge(lv_map, on="lv_id", how="inner")
    base = base[["lv_id", "target_set_id", "gene_identifier"]].drop_duplicates()
    if base.empty:
        raise ValueError("No LV-gene edges available after merging lv_top_genes.csv with lv_target_map.csv")
    return base.sort_values(["lv_id", "gene_identifier"]).reset_index(drop=True)


def load_gene_universe(output_dir: Path) -> list[object]:
    gene_ids = np.load(Path(output_dir) / "gene_ids.npy", allow_pickle=False)
    return gene_ids.tolist()


def _artifact_name(control: str, replicate: int) -> str:
    if control == "real":
        return "lv_real"
    return f"lv_{control}_{int(replicate):03d}"


def _artifact_path(output_dir: Path, control: str, replicate: int) -> Path:
    return artifact_dir(output_dir) / f"{_artifact_name(control, replicate)}.csv"


def write_real_artifact(output_dir: Path, force: bool = False) -> Path:
    output_dir = Path(output_dir)
    path = _artifact_path(output_dir, "real", 0)
    if path.exists() and not force:
        return path
    base = load_base_edges(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(path, index=False)
    return path


def generate_control_artifact(
    output_dir: Path,
    control: str,
    replicate: int,
    random_state_base: int = 42,
    promiscuity_tolerance: int = 2,
    n_swap_attempts_per_edge: int = 10,
    force: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    if control not in {"permuted", "random"}:
        raise ValueError(f"Unsupported control: {control}")

    path = _artifact_path(output_dir, control, replicate)
    if path.exists() and not force:
        return path

    base = load_base_edges(output_dir)
    seed = int(random_state_base) + int(replicate) - 1
    if control == "permuted":
        artifact_df = degree_preserving_permutations(
            edge_df=base,
            source_col="lv_id",
            target_col="gene_identifier",
            n_permutations=1,
            random_state=seed,
            n_swap_attempts_per_edge=n_swap_attempts_per_edge,
        )[0]
    else:
        artifact_df = generate_promiscuity_matched_samples(
            edge_df=base,
            all_annotations_df=base,
            source_col="lv_id",
            target_col="gene_identifier",
            target_universe=load_gene_universe(output_dir),
            promiscuity_tolerance=promiscuity_tolerance,
            random_state=seed,
            include_match_metadata=True,
        )
    artifact_df.to_csv(path, index=False)
    return path


def _parse_artifact(path: Path) -> dict | None:
    match = LV_ARTIFACT_RE.match(path.name)
    if not match:
        return None
    control = match.group(1)
    rep = int(match.group(2)) if match.group(2) else 0
    return {
        "domain": "lv",
        "name": path.stem,
        "control": control,
        "replicate": rep,
        "source_path": str(path),
        "result_path": "",
        "summary_path": str(summary_dir(path.parent.parent) / f"summary_{path.stem}.csv"),
    }


def write_manifest(output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    art_dir = artifact_dir(output_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(art_dir.glob("lv_*.csv")):
        parsed = _parse_artifact(path)
        if parsed is not None:
            rows.append(parsed)
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest = manifest.sort_values(["control", "replicate", "name"]).reset_index(drop=True)
    manifest.to_csv(manifest_path(output_dir), index=False)
    return manifest


def load_manifest(output_dir: Path) -> pd.DataFrame:
    path = manifest_path(output_dir)
    if path.exists():
        return pd.read_csv(path)
    return write_manifest(output_dir)


def _gene_id_to_row_map(output_dir: Path) -> dict[str, int]:
    gene_ids = np.load(Path(output_dir) / "gene_ids.npy", allow_pickle=False)
    out: dict[str, int] = {}
    for idx, gene_id in enumerate(gene_ids.tolist()):
        try:
            out[str(int(gene_id))] = idx
        except Exception:
            out[str(gene_id)] = idx
        out[str(gene_id)] = idx
    return out


def compute_summary_for_artifact(output_dir: Path, artifact_name: str, force: bool = False) -> Path:
    output_dir = Path(output_dir)
    manifest = load_manifest(output_dir)
    matches = manifest[manifest["name"].astype(str) == str(artifact_name)]
    if matches.empty:
        available = ", ".join(manifest["name"].astype(str).tolist())
        raise ValueError(f"Unknown LV artifact name: {artifact_name}. Available: {available}")
    meta = matches.iloc[0]
    out_path = Path(meta["summary_path"])
    if out_path.exists() and not force:
        return out_path

    feature_manifest = pd.read_csv(output_dir / "feature_manifest.csv")
    lv_map = pd.read_csv(output_dir / "lv_target_map.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if str(meta["control"]) == "real":
        real_feature_scores = pd.read_csv(output_dir / "real_feature_scores.csv")
        summary_df = real_feature_scores.rename(columns={"real_mean": "mean_score"}).copy()
        summary_df.insert(0, "replicate", 0)
        summary_df.insert(0, "control", "real")
        summary_df.insert(0, "name", str(meta["name"]))
        summary_df.insert(0, "domain", "lv")
        summary_df.to_csv(out_path, index=False)
        return out_path

    score_matrix = np.load(output_dir / "gene_feature_scores.npy", allow_pickle=False)
    gene_id_to_row = _gene_id_to_row_map(output_dir)
    artifact_df = pd.read_csv(meta["source_path"])
    if "target_set_id" not in artifact_df.columns:
        artifact_df = artifact_df.merge(lv_map, on="lv_id", how="left")

    rows = []
    for lv_id, group in artifact_df.groupby("lv_id", sort=True):
        target_set_id = str(group["target_set_id"].iloc[0])
        feature_rows = feature_manifest[
            feature_manifest["target_set_id"].astype(str) == target_set_id
        ].copy()
        gene_row_idx = []
        for gene_id in group["gene_identifier"].tolist():
            token = str(gene_id)
            idx = gene_id_to_row.get(token)
            if idx is None:
                numeric = pd.to_numeric(pd.Series([gene_id]), errors="coerce").iloc[0]
                if not pd.isna(numeric):
                    idx = gene_id_to_row.get(str(int(numeric)))
            if idx is not None:
                gene_row_idx.append(int(idx))
        if not gene_row_idx:
            raise ValueError(f"No mapped genes found for artifact={artifact_name}, lv_id={lv_id}")
        lv_mean = score_matrix[np.asarray(gene_row_idx, dtype=np.int64)].mean(axis=0)
        for feature in feature_rows.itertuples(index=False):
            rows.append(
                {
                    "domain": "lv",
                    "name": str(meta["name"]),
                    "control": str(meta["control"]),
                    "replicate": int(meta["replicate"]),
                    "lv_id": lv_id,
                    "target_set_id": feature.target_set_id,
                    "target_set_label": feature.target_set_label,
                    "node_type": feature.node_type,
                    "feature_idx": int(feature.feature_idx),
                    "metapath": feature.metapath,
                    "mean_score": float(lv_mean[int(feature.feature_idx)]),
                    "n_genes": int(len(gene_row_idx)),
                }
            )
    summary_df = pd.DataFrame(rows).sort_values(["control", "replicate", "lv_id", "metapath"])
    summary_df.to_csv(out_path, index=False)
    return out_path


def list_artifact_names(output_dir: Path) -> list[str]:
    manifest = load_manifest(output_dir)
    if manifest.empty:
        return []
    return manifest["name"].astype(str).tolist()
