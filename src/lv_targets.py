"""
Target construction for LV multi-DWPC analysis.

Each LV maps to a single target node. The user provides:
- A set of genes (via LV loadings or direct input)
- A target node (node_type + target_id)

This module builds the LV-to-target mapping and resolves target metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd


class LVTarget(NamedTuple):
    """Single target definition for an LV."""

    lv_id: str
    target_id: str
    target_name: str
    node_type: str
    target_position: int


# Default LV-to-target mappings (can be overridden via config)
DEFAULT_LV_TARGETS: dict[str, tuple[str, str]] = {
    # lv_id -> (node_type, target_id)
    "LV603": ("Side Effect", "C3665444"),  # Neutrophilia
    "LV246": ("Anatomy", "UBERON:0001013"),  # Adipose tissue
    "LV57": ("Disease", "DOID:1459"),  # Hypothyroidism
}


def _normalize_lv_id(value: object) -> str:
    """Normalize LV IDs to canonical form (e.g., lv603 -> LV603)."""
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


def _load_nodes_table(path: Path) -> pd.DataFrame:
    """Load a hetnet nodes TSV file."""
    df = pd.read_csv(path, sep="\t")
    required_cols = {"position", "identifier", "name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Node table missing required columns {sorted(missing)}: {path}")
    return df


def resolve_target(
    nodes_dir: Path,
    node_type: str,
    target_id: str,
) -> LVTarget | None:
    """
    Resolve a target node from the hetnet nodes directory.

    Returns None if the target is not found.
    """
    node_file = nodes_dir / f"{node_type}.tsv"
    if not node_file.exists():
        return None

    nodes_df = _load_nodes_table(node_file)
    match = nodes_df[nodes_df["identifier"].astype(str) == str(target_id)]
    if match.empty:
        return None

    row = match.iloc[0]
    return LVTarget(
        lv_id="",  # Will be set by caller
        target_id=str(row["identifier"]),
        target_name=str(row["name"]),
        node_type=node_type,
        target_position=int(row["position"]),
    )


def build_lv_targets(
    nodes_dir: Path,
    output_path: Path,
    lv_ids: list[str],
    lv_target_overrides: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Build LV-to-target mapping for the specified LVs.

    Args:
        nodes_dir: Path to hetnet nodes directory containing {NodeType}.tsv files
        output_path: Path to write lv_targets.csv
        lv_ids: List of LV IDs to process
        lv_target_overrides: Optional dict mapping lv_id -> (node_type, target_id)
            to override default mappings

    Returns:
        DataFrame with columns: lv_id, target_id, target_name, node_type, target_position

    Outputs:
        - {output_path}: LV-to-target mapping CSV
    """
    lv_targets = lv_target_overrides or {}
    # Merge with defaults for any LVs not in overrides
    for lv_id in lv_ids:
        normalized = _normalize_lv_id(lv_id)
        if normalized not in lv_targets and normalized in DEFAULT_LV_TARGETS:
            lv_targets[normalized] = DEFAULT_LV_TARGETS[normalized]

    rows: list[dict] = []
    errors: list[str] = []

    for lv_id in lv_ids:
        normalized = _normalize_lv_id(lv_id)
        if normalized not in lv_targets:
            errors.append(f"No target mapping for {normalized}")
            continue

        node_type, target_id = lv_targets[normalized]
        target = resolve_target(nodes_dir, node_type, target_id)
        if target is None:
            errors.append(f"Target {target_id} ({node_type}) not found for {normalized}")
            continue

        rows.append({
            "lv_id": normalized,
            "target_id": target.target_id,
            "target_name": target.target_name,
            "node_type": target.node_type,
            "target_position": target.target_position,
        })

    if errors:
        raise ValueError("Target resolution errors:\n" + "\n".join(errors))

    if not rows:
        raise ValueError("No LV targets were resolved")

    targets_df = pd.DataFrame(rows)
    targets_df = targets_df.sort_values("lv_id").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets_df.to_csv(output_path, index=False)

    return targets_df


def load_lv_targets(output_dir: Path) -> pd.DataFrame:
    """Load the LV targets CSV from an output directory."""
    path = Path(output_dir) / "lv_targets.csv"
    if not path.exists():
        raise FileNotFoundError(f"LV targets file not found: {path}")
    return pd.read_csv(path)
