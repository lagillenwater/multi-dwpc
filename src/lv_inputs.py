"""
Utilities for latent variable input processing.
- load LV-gene loading tables
- map genes to Hetionet gene nodes
- select top fraction of genes per LV
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


LV_COLUMN_CANDIDATES = ("lv_id", "lv", "latent_variable", "component")
GENE_COLUMN_CANDIDATES = (
    "gene",
    "gene_id",
    "gene_symbol",
    "symbol",
    "entrez_gene_id",
    "entrez",
)
LOADING_COLUMN_CANDIDATES = ("loading", "weight", "score", "value")


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        header = path.read_bytes()[:120]
    except OSError:
        return False
    return header.startswith(b"version https://git-lfs.github.com/spec/v1")


def _normalize_lv_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    upper_text = text.upper()
    if upper_text.startswith("LV"):
        suffix = upper_text[2:]
        if suffix.isdigit():
            return f"LV{int(suffix)}"
        return upper_text
    if text.isdigit():
        return f"LV{int(text)}"
    if text.endswith(".0"):
        stem = text[:-2]
        if stem.isdigit():
            return f"LV{int(stem)}"
    return upper_text


def _normalize_gene_token(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    if ":" in text:
        text = text.split(":")[-1].strip()
    if text.endswith(".0"):
        stem = text[:-2]
        if stem.isdigit():
            text = stem
    return text


def _read_lv_loadings(path: Path) -> pd.DataFrame:
    filename = path.name.lower()
    if not path.exists():
        raise FileNotFoundError(f"LV loadings file not found: {path}")

    if _is_git_lfs_pointer(path):
        raise ValueError(
            "LV loadings file appears to be a Git LFS pointer, not real data. "
            "Re-download the file (for example: `poe download-lv-loadings`) and "
            "ensure the downloaded file is the actual gzip content."
        )

    if filename.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path, sep="\t")
    if filename.endswith(".parquet"):
        return pd.read_parquet(path)
    if filename.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path)
    return pd.read_csv(path)


def _resolve_column(
    df: pd.DataFrame, candidates: Iterable[str], explicit: Optional[str]
) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Provided column '{explicit}' is missing. "
                f"Available: {sorted(df.columns.tolist())}"
            )
        return explicit

    lower_to_original = {col.strip().lower(): col for col in df.columns}
    for name in candidates:
        if name in lower_to_original:
            return lower_to_original[name]

    raise ValueError(
        f"Unable to resolve required column from candidates {tuple(candidates)}. "
        f"Available: {sorted(df.columns.tolist())}"
    )


def _coerce_wide_z_to_long(
    df: pd.DataFrame,
    requested_lvs: Optional[Iterable[str]],
    explicit_gene_column: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Convert wide MultiPLIER Z matrix format to long format.

    Expected wide shape:
    - one gene column (often "Unnamed: 0")
    - many LV columns (e.g., LV1, LV2, ..., LV987)
    """
    if df.empty:
        return None

    if explicit_gene_column:
        if explicit_gene_column not in df.columns:
            raise ValueError(
                f"Provided gene column '{explicit_gene_column}' is missing. "
                f"Available: {sorted(df.columns.tolist())}"
            )
        gene_col = explicit_gene_column
    else:
        lower_to_original = {col.strip().lower(): col for col in df.columns}
        gene_col = None
        for name in GENE_COLUMN_CANDIDATES:
            if name in lower_to_original:
                gene_col = lower_to_original[name]
                break
        if gene_col is None and "unnamed: 0" in lower_to_original:
            gene_col = lower_to_original["unnamed: 0"]
        if gene_col is None:
            gene_col = df.columns[0]

    normalized_requested = None
    if requested_lvs:
        normalized_requested = {_normalize_lv_id(item) for item in requested_lvs}

    lv_col_map: dict[str, str] = {}
    for col in df.columns:
        if col == gene_col:
            continue
        normalized = _normalize_lv_id(col)
        if not (normalized.startswith("LV") and normalized[2:].isdigit()):
            continue
        if normalized_requested and normalized not in normalized_requested:
            continue
        lv_col_map[col] = normalized

    if not lv_col_map:
        return None

    wide_subset = df[[gene_col, *lv_col_map.keys()]].copy()
    wide_subset = wide_subset.rename(columns={gene_col: "gene_raw"})
    long_df = wide_subset.melt(
        id_vars=["gene_raw"],
        value_vars=list(lv_col_map.keys()),
        var_name="lv_col",
        value_name="loading_raw",
    )
    long_df["lv_raw"] = long_df["lv_col"].map(lv_col_map)
    return long_df[["lv_raw", "gene_raw", "loading_raw"]]


def _load_gene_reference(gene_tsv_path: Path) -> pd.DataFrame:
    gene_df = pd.read_csv(gene_tsv_path, sep="\t")
    required = {"identifier", "name"}
    missing = required - set(gene_df.columns)
    if missing:
        raise ValueError(
            f"Gene reference missing columns: {sorted(missing)} "
            f"from {gene_tsv_path}"
        )
    gene_df["identifier_str"] = gene_df["identifier"].astype(str)
    gene_df["symbol_upper"] = gene_df["name"].astype(str).str.upper()
    return gene_df


def _map_input_gene_to_identifier(
    gene_token: str,
    identifier_set: set[str],
    symbol_to_identifier: dict[str, str],
) -> Optional[str]:
    if not gene_token:
        return None
    if gene_token in identifier_set:
        return gene_token
    upper_token = gene_token.upper()
    if upper_token in symbol_to_identifier:
        return symbol_to_identifier[upper_token]
    return None


def extract_top_lv_genes(
    lv_loadings_path: Path,
    gene_reference_path: Path,
    output_top_genes_path: Path,
    output_summary_path: Path,
    requested_lvs: Optional[Iterable[str]] = None,
    top_fraction: float = 0.01,
    lv_column: Optional[str] = None,
    gene_column: Optional[str] = None,
    loading_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load LV loadings, map genes to Hetionet, and select top genes per LV.

    Parameters
    ----------
    lv_loadings_path
        Path to LV loadings table.
    gene_reference_path
        Path to `data/nodes/Gene.tsv`.
    output_top_genes_path
        CSV path for selected top genes.
    output_summary_path
        CSV path for per-LV summary metrics.
    requested_lvs
        Optional iterable of LV IDs to keep.
    top_fraction
        Fraction of mapped genes to keep per LV (e.g., 0.01 for top 1%).
    lv_column, gene_column, loading_column
        Optional explicit column names in the LV loadings table.
    """
    if top_fraction <= 0 or top_fraction > 1:
        raise ValueError("top_fraction must be in (0, 1].")

    normalized_requested = (
        {_normalize_lv_id(item) for item in requested_lvs}
        if requested_lvs
        else None
    )

    raw_df = _read_lv_loadings(Path(lv_loadings_path))
    try:
        resolved_lv_col = _resolve_column(raw_df, LV_COLUMN_CANDIDATES, lv_column)
        resolved_gene_col = _resolve_column(raw_df, GENE_COLUMN_CANDIDATES, gene_column)
        resolved_loading_col = _resolve_column(
            raw_df, LOADING_COLUMN_CANDIDATES, loading_column
        )
        working = raw_df[[resolved_lv_col, resolved_gene_col, resolved_loading_col]].copy()
        working.columns = ["lv_raw", "gene_raw", "loading_raw"]
    except ValueError:
        if lv_column or loading_column:
            raise
        coerced = _coerce_wide_z_to_long(
            raw_df,
            requested_lvs=normalized_requested,
            explicit_gene_column=gene_column,
        )
        if coerced is None:
            raise
        working = coerced.copy()

    working["lv_id"] = working["lv_raw"].map(_normalize_lv_id)
    working["gene_token"] = working["gene_raw"].map(_normalize_gene_token)
    working["loading"] = pd.to_numeric(working["loading_raw"], errors="coerce")
    working = working.dropna(subset=["lv_id", "gene_token", "loading"]).copy()

    if normalized_requested:
        working = working[working["lv_id"].isin(normalized_requested)].copy()

    if working.empty:
        raise ValueError("No rows left after loading/filtering LV data.")

    gene_df = _load_gene_reference(Path(gene_reference_path))
    identifier_set = set(gene_df["identifier_str"].tolist())
    symbol_to_identifier = dict(
        zip(gene_df["symbol_upper"], gene_df["identifier_str"])
    )
    identifier_to_symbol = dict(
        zip(gene_df["identifier_str"], gene_df["name"])
    )

    working["gene_identifier"] = working["gene_token"].map(
        lambda token: _map_input_gene_to_identifier(
            token, identifier_set, symbol_to_identifier
        )
    )
    mapped = working.dropna(subset=["gene_identifier"]).copy()
    if mapped.empty:
        raise ValueError("No genes mapped to Hetionet Gene.tsv identifiers.")

    # Keep strongest loading if duplicated gene appears within one LV.
    mapped = mapped.sort_values("loading", ascending=False)
    mapped = mapped.drop_duplicates(subset=["lv_id", "gene_identifier"], keep="first")

    mapped["n_mapped_genes"] = mapped.groupby("lv_id")["gene_identifier"].transform("size")
    mapped["n_select"] = mapped["n_mapped_genes"].map(
        lambda count: max(1, int(math.ceil(top_fraction * float(count))))
    )
    mapped["rank_within_lv"] = mapped.groupby("lv_id")["loading"].rank(
        method="first", ascending=False
    )
    selected = mapped[mapped["rank_within_lv"] <= mapped["n_select"]].copy()
    selected["gene_symbol"] = selected["gene_identifier"].map(identifier_to_symbol)
    selected = selected.sort_values(["lv_id", "rank_within_lv"])

    selected_output = selected[
        [
            "lv_id",
            "gene_identifier",
            "gene_symbol",
            "loading",
            "rank_within_lv",
            "n_mapped_genes",
            "n_select",
        ]
    ].copy()
    selected_output = selected_output.rename(
        columns={"rank_within_lv": "rank", "n_select": "n_top_genes"}
    )

    input_counts = (
        working.groupby("lv_id", as_index=False)
        .size()
        .rename(columns={"size": "n_rows_input"})
    )
    mapped_counts = (
        mapped.groupby("lv_id", as_index=False)["gene_identifier"]
        .nunique()
        .rename(columns={"gene_identifier": "n_unique_genes_mapped"})
    )
    selected_counts = (
        selected_output.groupby("lv_id", as_index=False)["gene_identifier"]
        .nunique()
        .rename(columns={"gene_identifier": "n_unique_genes_selected"})
    )
    threshold_by_lv = (
        selected_output.groupby("lv_id", as_index=False)["loading"]
        .min()
        .rename(columns={"loading": "loading_threshold"})
    )

    summary = input_counts.merge(mapped_counts, on="lv_id", how="left")
    summary = summary.merge(selected_counts, on="lv_id", how="left")
    summary = summary.merge(threshold_by_lv, on="lv_id", how="left")
    summary["top_fraction"] = top_fraction
    summary = summary.sort_values("lv_id")

    output_top_genes_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    selected_output.to_csv(output_top_genes_path, index=False)
    summary.to_csv(output_summary_path, index=False)

    return selected_output, summary
