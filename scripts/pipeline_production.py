"""
Run the full production pipeline (no GO hierarchy analysis).

This script executes the production-ready sequence of pipeline steps using the
per-script entry points under scripts/.
"""

from pathlib import Path
import subprocess
import sys


PIPELINE_STEPS = [
    "scripts/load_data.py",
    "scripts/percent_change_and_filtering.py",
    "scripts/jaccard_similarity_and_filtering.py",
    "scripts/permutation_null_datasets.py",
    "scripts/random_null_datasets.py",
]


def run_step(repo_root: Path, script_path: str) -> None:
    """Run a pipeline step and raise on failure."""
    script_full_path = repo_root / script_path
    print(f"\nRunning: {script_full_path}")
    subprocess.run([sys.executable, str(script_full_path)], check=True)


def main() -> None:
    """Execute all production pipeline steps."""
    repo_root = Path(__file__).resolve().parent.parent
    print(f"Repository root: {repo_root}")

    for step in PIPELINE_STEPS:
        run_step(repo_root, step)

    print("\nProduction pipeline complete.")


if __name__ == "__main__":
    main()
