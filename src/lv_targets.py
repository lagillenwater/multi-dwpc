"""
Target-set construction for LV multi-DWPC analysis.

Task scope:
- Build fixed target sets for neutrophil, adipose tissue, and hypothyroidism.
- Export row-level target membership and set-level summary tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

## NB need to parameterize these to avoid hardcoding in the target-set construction function. 3-3-2026
NEUTROPHILIA_TARGET_SET = "neutrophilia_se"
ADIPOSE_TARGET_SET = "adipose_tissue"
HYPOTHYROIDISM_TARGET_SET = "hypothyroidism"

NEUTROPHILIA_SE_ID = "C3665444"
ADIPOSE_ID = "UBERON:0001013"
BROWN_ADIPOSE_ID = "UBERON:0001348"
HYPOTHYROIDISM_ID = "DOID:1459"


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
    df = pd.read_csv(path, sep="\t")
    required_cols = {"position", "identifier", "name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Node table missing required columns {sorted(missing)}: {path}"
        )
    return df


def _build_rows(
    node_df: pd.DataFrame,
    node_type: str,
    target_set_id: str,
    target_set_label: str,
) -> pd.DataFrame:
    out = node_df[["identifier", "name", "position"]].copy()
    out = out.rename(
        columns={
            "identifier": "target_id",
            "name": "target_name",
            "position": "target_position",
        }
    )
    out["node_type"] = node_type
    out["target_set_id"] = target_set_id
    out["target_set_label"] = target_set_label
    return out[
        [
            "target_set_id",
            "target_set_label",
            "node_type",
            "target_id",
            "target_name",
            "target_position",
        ]
    ]


def _build_neutrophilia_se_rows(se_df: pd.DataFrame) -> pd.DataFrame:
    neut_df = se_df[se_df["identifier"].astype(str) == NEUTROPHILIA_SE_ID].copy()
    if neut_df.empty:
        raise ValueError(
            f"Neutrophilia side effect {NEUTROPHILIA_SE_ID} not found in Side Effect.tsv"
        )
    return _build_rows(
        node_df=neut_df,
        node_type="Side Effect",
        target_set_id=NEUTROPHILIA_TARGET_SET,
        target_set_label="Neutrophilia side effect term",
    )


def _build_adipose_rows(
    anatomy_df: pd.DataFrame,
    include_brown_adipose: bool,
) -> pd.DataFrame:
    wanted_ids = [ADIPOSE_ID]
    if include_brown_adipose:
        wanted_ids.append(BROWN_ADIPOSE_ID)

    adipose_df = anatomy_df[anatomy_df["identifier"].isin(wanted_ids)].copy()
    missing_ids = [node_id for node_id in wanted_ids if node_id not in set(adipose_df["identifier"])]
    if missing_ids:
        raise ValueError(
            f"Missing anatomy target IDs in Anatomy.tsv: {missing_ids}"
        )

    label = "Adipose tissue anatomy terms"
    if include_brown_adipose:
        label = "Adipose tissue anatomy terms (including brown adipose tissue)"
    return _build_rows(
        node_df=adipose_df,
        node_type="Anatomy",
        target_set_id=ADIPOSE_TARGET_SET,
        target_set_label=label,
    )


def _build_hypothyroidism_rows(disease_df: pd.DataFrame) -> pd.DataFrame:
    hypo_df = disease_df[disease_df["identifier"] == HYPOTHYROIDISM_ID].copy()
    if hypo_df.empty:
        fallback = disease_df[
            disease_df["name"].astype(str).str.contains(
                "hypothyroidism", case=False, regex=False
            )
        ].copy()
        if fallback.empty:
            raise ValueError("Hypothyroidism disease node not found in Disease.tsv")
        hypo_df = fallback

    return _build_rows(
        node_df=hypo_df,
        node_type="Disease",
        target_set_id=HYPOTHYROIDISM_TARGET_SET,
        target_set_label="Hypothyroidism disease term",
    )


def _build_summary(target_sets_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        target_sets_df.groupby(
            ["target_set_id", "target_set_label", "node_type"], as_index=False
        )["target_id"]
        .nunique()
        .rename(columns={"target_id": "n_targets"})
    )
    return summary.sort_values("target_set_id")


def _build_lv_target_map_rows(lv_ids: Iterable[str]) -> pd.DataFrame:
    lv_to_target = {
        "LV603": NEUTROPHILIA_TARGET_SET,
        "LV246": ADIPOSE_TARGET_SET,
        "LV57": HYPOTHYROIDISM_TARGET_SET,
    }

    requested_lvs = []
    seen = set()
    for lv_id in lv_ids:
        normalized = _normalize_lv_id(lv_id)
        if not normalized or normalized in seen:
            continue
        requested_lvs.append(normalized)
        seen.add(normalized)

    rows = []
    unmapped = []
    for lv_id in requested_lvs:
        mapped_target = lv_to_target.get(lv_id)
        if mapped_target is None:
            unmapped.append(lv_id)
            continue
        rows.append(
            {
                "lv_id": lv_id,
                "target_set_id": mapped_target,
            }
        )

    if unmapped:
        supported = sorted(lv_to_target.keys())
        raise ValueError(
            "No target mapping configured for LV IDs: "
            f"{sorted(unmapped)}. Supported LV IDs: {supported}"
        )

    return pd.DataFrame(rows)


def build_target_sets(
    nodes_dir: Path,
    output_target_sets_path: Path,
    output_summary_path: Path,
    output_lv_map_path: Path,
    lv_ids: Iterable[str],
    include_brown_adipose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construct fixed biological target sets and write outputs.
    """
    se_df = _load_nodes_table(nodes_dir / "Side Effect.tsv")
    anatomy_df = _load_nodes_table(nodes_dir / "Anatomy.tsv")
    disease_df = _load_nodes_table(nodes_dir / "Disease.tsv")

    targets_df = pd.concat(
        [
            _build_neutrophilia_se_rows(se_df),
            _build_adipose_rows(anatomy_df, include_brown_adipose),
            _build_hypothyroidism_rows(disease_df),
        ],
        ignore_index=True,
    )
    targets_df = targets_df.sort_values(["target_set_id", "target_id"]).reset_index(
        drop=True
    )
    summary_df = _build_summary(targets_df)

    lv_map_df = _build_lv_target_map_rows(lv_ids)
    if lv_map_df.empty:
        raise ValueError("No LV-to-target mappings were generated.")
    lv_map_df = lv_map_df.sort_values("lv_id").reset_index(drop=True)

    output_target_sets_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_lv_map_path.parent.mkdir(parents=True, exist_ok=True)
    targets_df.to_csv(output_target_sets_path, index=False)
    summary_df.to_csv(output_summary_path, index=False)
    lv_map_df.to_csv(output_lv_map_path, index=False)

    return targets_df, summary_df, lv_map_df
