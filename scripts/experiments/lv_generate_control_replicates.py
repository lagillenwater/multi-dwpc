#!/usr/bin/env python3
"""Generate explicit LV null replicate artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.lv_explicit_replicates import generate_control_artifact, write_manifest, write_real_artifact  # noqa: E402


def _parse_ids(arg: str) -> list[int]:
    values = [int(tok.strip()) for tok in str(arg).split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one replicate id")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/lv_experiment")
    parser.add_argument("--control", required=True, choices=["permuted", "random"])
    parser.add_argument("--replicate-id", type=int, default=None)
    parser.add_argument("--replicate-ids", default=None)
    parser.add_argument("--n-replicates", type=int, default=20)
    parser.add_argument("--random-state-base", type=int, default=42)
    parser.add_argument("--promiscuity-tolerance", type=int, default=2)
    parser.add_argument("--n-swap-attempts-per-edge", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    write_real_artifact(output_dir, force=False)

    if args.replicate_id is not None:
        replicate_ids = [int(args.replicate_id)]
    elif args.replicate_ids:
        replicate_ids = _parse_ids(args.replicate_ids)
    else:
        replicate_ids = list(range(1, int(args.n_replicates) + 1))

    for replicate in replicate_ids:
        path = generate_control_artifact(
            output_dir=output_dir,
            control=args.control,
            replicate=replicate,
            random_state_base=args.random_state_base,
            promiscuity_tolerance=args.promiscuity_tolerance,
            n_swap_attempts_per_edge=args.n_swap_attempts_per_edge,
            force=args.force,
        )
        print(f"[saved] {path}")

    manifest = write_manifest(output_dir)
    print(f"Updated manifest: {output_dir / 'replicate_manifest.csv'} ({len(manifest)} rows)")


if __name__ == "__main__":
    main()
