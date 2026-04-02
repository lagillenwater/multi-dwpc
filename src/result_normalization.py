"""Normalize direct and API DWPC outputs into a shared result schema."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


YEAR_DATASET_RE = re.compile(
    r"^(?P<base>.+)_(?P<year>\d{4})_(?P<kind>real|perm|random)(?:_(?P<rep>\d+))?$"
)


def parse_year_dataset_name(dataset_name: str) -> dict:
    """Parse a year dataset name into standardized metadata."""
    match = YEAR_DATASET_RE.match(str(dataset_name))
    if not match:
        raise ValueError(f"Unrecognized year dataset name: {dataset_name}")
    kind = match.group("kind")
    control_map = {"real": "real", "perm": "permuted", "random": "random"}
    rep = int(match.group("rep")) if match.group("rep") else 0
    return {
        "base_name": str(match.group("base")),
        "year": int(match.group("year")),
        "control": control_map[kind],
        "replicate": int(rep),
        "name": str(dataset_name),
    }


def load_neo4j_mappings(data_dir: Path) -> tuple[dict, dict]:
    """Load mappings from Neo4j node ids to external GO and Entrez identifiers."""
    gene_map = pd.read_csv(Path(data_dir) / "neo4j_gene_mapping.csv")
    bp_map = pd.read_csv(Path(data_dir) / "neo4j_bp_mapping.csv")
    neo4j_to_entrez = dict(zip(gene_map["neo4j_id"], gene_map["identifier"]))
    neo4j_to_go = dict(zip(bp_map["neo4j_id"], bp_map["identifier"]))
    return neo4j_to_entrez, neo4j_to_go


def normalize_direct_year_result(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Normalize a direct-DWPC year result frame into the shared schema."""
    meta = parse_year_dataset_name(dataset_name)
    required = {"go_id", "entrez_gene_id", "metapath", "dwpc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Direct year result {dataset_name} missing required columns: {sorted(missing)}"
        )

    out = df.copy()
    out["metapath"] = out["metapath"].astype(str)
    out.insert(0, "score_source", "direct")
    out.insert(0, "replicate", meta["replicate"])
    out.insert(0, "control", meta["control"])
    out.insert(0, "year", meta["year"])
    out.insert(0, "name", meta["name"])
    out.insert(0, "domain", "year")
    return out


def normalize_api_year_result(
    df: pd.DataFrame,
    dataset_name: str,
    neo4j_to_entrez: dict,
    neo4j_to_go: dict,
) -> pd.DataFrame:
    """Normalize an API year result frame into the shared schema."""
    meta = parse_year_dataset_name(dataset_name)
    required = {"neo4j_source_id", "neo4j_target_id", "dwpc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"API year result {dataset_name} missing required columns: {sorted(missing)}"
        )

    out = df.copy()
    if "metapath" not in out.columns:
        if "metapath_abbreviation" not in out.columns:
            raise ValueError(
                f"API year result {dataset_name} must include `metapath` or `metapath_abbreviation`."
            )
        out = out.rename(columns={"metapath_abbreviation": "metapath"})

    out["go_id"] = out["neo4j_source_id"].map(neo4j_to_go)
    target_col = "neo4j_pseudo_target_id" if "neo4j_pseudo_target_id" in out.columns else "neo4j_target_id"
    out["entrez_gene_id"] = out[target_col].map(neo4j_to_entrez)
    out = out.dropna(subset=["go_id", "entrez_gene_id", "metapath", "dwpc"]).copy()

    out.insert(0, "score_source", "api")
    out.insert(0, "replicate", meta["replicate"])
    out.insert(0, "control", meta["control"])
    out.insert(0, "year", meta["year"])
    out.insert(0, "name", meta["name"])
    out.insert(0, "domain", "year")
    return out
