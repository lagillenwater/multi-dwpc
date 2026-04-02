"""Shared workspace helpers for replicate-summary analyses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CORE_MANIFEST_COLUMNS = [
    "domain",
    "name",
    "control",
    "replicate",
    "source_path",
    "result_path",
    "summary_path",
]


def resolve_manifest_path(path: Path | str) -> Path:
    """Return the manifest path for a workspace directory or manifest file."""
    path = Path(path)
    if path.is_dir():
        return path / "replicate_manifest.csv"
    return path


def validate_manifest_columns(
    manifest_df: pd.DataFrame,
    required_cols: list[str] | None = None,
) -> None:
    """Validate that a manifest includes the required core columns."""
    required = set(required_cols or CORE_MANIFEST_COLUMNS)
    missing = required - set(manifest_df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")


def load_manifest(path: Path | str, required_cols: list[str] | None = None) -> pd.DataFrame:
    """Load a replicate manifest from a workspace directory or manifest file."""
    manifest_path = resolve_manifest_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Replicate manifest not found: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        return manifest_df
    validate_manifest_columns(manifest_df, required_cols=required_cols)
    return manifest_df


def load_summary_bank_from_manifest(
    manifest_df: pd.DataFrame,
    required_summary_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load all summary CSVs referenced by a manifest."""
    if manifest_df.empty:
        return pd.DataFrame()

    frames = []
    for row in manifest_df.itertuples(index=False):
        summary_path = Path(row.summary_path)
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary CSV referenced by manifest: {summary_path}")
        frame = pd.read_csv(summary_path)
        if required_summary_cols:
            missing = set(required_summary_cols) - set(frame.columns)
            if missing:
                raise ValueError(
                    f"Summary CSV {summary_path} missing required columns: {sorted(missing)}"
                )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_summary_bank(
    path: Path | str,
    required_summary_cols: list[str] | None = None,
    summary_glob: str = "summary_*.csv",
) -> pd.DataFrame:
    """
    Load summary CSVs from a manifest-aware workspace or a raw summaries directory.

    This preserves backward compatibility for older year workspaces that only
    expose a `replicate_summaries/` directory with no manifest-driven loader.
    """
    path = Path(path)
    manifest_path = resolve_manifest_path(path)
    if manifest_path.exists():
        manifest_df = load_manifest(manifest_path)
        return load_summary_bank_from_manifest(
            manifest_df,
            required_summary_cols=required_summary_cols,
        )

    summary_dir = path
    if summary_dir.is_file():
        raise FileNotFoundError(
            f"Expected a workspace directory or manifest file, got file without manifest: {summary_dir}"
        )

    files = sorted(summary_dir.glob(summary_glob))
    if not files:
        raise FileNotFoundError(f"No summary CSVs found under {summary_dir}")

    frames = []
    for csv_path in files:
        frame = pd.read_csv(csv_path)
        if required_summary_cols:
            missing = set(required_summary_cols) - set(frame.columns)
            if missing:
                raise ValueError(
                    f"Summary CSV {csv_path} missing required columns: {sorted(missing)}"
                )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
