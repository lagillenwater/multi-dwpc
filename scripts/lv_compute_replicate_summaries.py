#!/usr/bin/env python3
"""Compute per-replicate LV metapath summary artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.lv_explicit_replicates import compute_summary_for_artifact, list_artifact_names, load_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/lv_experiment")
    parser.add_argument("--list-artifacts", action="store_true")
    parser.add_argument("--artifact-name", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.list_artifacts:
        for name in list_artifact_names(output_dir):
            print(name)
        return

    if args.artifact_name:
        path = compute_summary_for_artifact(output_dir, args.artifact_name, force=args.force)
        print(f"[saved] {path}")
        return

    manifest = load_manifest(output_dir)
    for name in manifest["name"].astype(str).tolist():
        path = compute_summary_for_artifact(output_dir, name, force=args.force)
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
