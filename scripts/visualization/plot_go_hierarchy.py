#!/usr/bin/env python3
"""Plot-only sibling of scripts/data_prep/go_hierarchy_analysis.py figure.

Reads the cached hierarchy metrics CSV and re-renders the 3-panel scatter
without re-running OBO parsing or hierarchy traversal.

Outputs:
    <output-dir>/hierarchy_vs_percent_change.{pdf,jpeg}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Spearman rho + (very rough) two-sided p-value via t approximation.

    We avoid scipy.stats here so the script runs in environments without scipy.
    The plotted r value is exact; the p-value is an approximation accurate
    enough for an annotation label.
    """
    a = x.astype(float).rank(method="average")
    b = y.astype(float).rank(method="average")
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return float("nan"), float("nan")
    a = a[mask].to_numpy()
    b = b[mask].to_numpy()
    n = len(a)
    ra = a - a.mean()
    rb = b - b.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if denom == 0.0:
        return float("nan"), float("nan")
    rho = float((ra * rb).sum() / denom)

    if abs(rho) >= 1.0 or n <= 2:
        p = 0.0 if abs(rho) >= 1.0 else float("nan")
    else:
        # Student-t p-value via complementary normal approx (good for n>=20).
        t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
        # 1 - Phi(|t|) * 2; standard normal CDF approximated by erf.
        from math import erf, sqrt
        p = float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0)))))
    return rho, p


def _save_dual(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.jpeg", dpi=300, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="output/cached_hierarchy_metrics.csv",
        help="Path to cached hierarchy metrics CSV (default: output/cached_hierarchy_metrics.csv).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Where to write hierarchy_vs_percent_change.{pdf,jpeg}.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Error if the cached hierarchy metrics CSV is missing. Default behavior is to skip with a warning, "
        "since go_hierarchy_analysis.py is a manual exploration step and is not part of submit_end_to_end.sh.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        msg = (
            f"{input_path} not found. go_hierarchy_analysis.py is a manual step "
            "outside the e2e pipeline; run it once to populate this cache."
        )
        if args.strict:
            raise FileNotFoundError(msg)
        print(f"WARNING: {msg} Skipping hierarchy_vs_percent_change figure.")
        return

    df = pd.read_csv(input_path)
    needed = ["normalized_depth_2016", "normalized_depth_2024", "norm_depth_change", "pct_change_genes"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{input_path} missing columns: {missing}")

    work = df.dropna(subset=needed).copy()

    rho_2016, p_2016 = _spearman(work["normalized_depth_2016"], work["pct_change_genes"])
    rho_2024, p_2024 = _spearman(work["normalized_depth_2024"], work["pct_change_genes"])
    rho_change, p_change = _spearman(work["norm_depth_change"], work["pct_change_genes"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)

    axes[0].scatter(work["normalized_depth_2016"], work["pct_change_genes"], alpha=0.3, s=20)
    axes[0].set_xlabel("Normalized Depth (2016)", fontsize=11)
    axes[0].set_ylabel("Percent Change in Gene Count", fontsize=11)
    axes[0].set_title(f"2016 Depth vs Change\nSpearman r={rho_2016:.3f}, p={p_2016:.2e}", fontsize=11)
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].scatter(work["normalized_depth_2024"], work["pct_change_genes"], alpha=0.3, s=20, color="orange")
    axes[1].set_xlabel("Normalized Depth (2024)", fontsize=11)
    axes[1].set_ylabel("Percent Change in Gene Count", fontsize=11)
    axes[1].set_title(f"2024 Depth vs Change\nSpearman r={rho_2024:.3f}, p={p_2024:.2e}", fontsize=11)
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].scatter(work["norm_depth_change"], work["pct_change_genes"], alpha=0.3, s=20, color="green")
    axes[2].set_xlabel("Change in Normalized Depth", fontsize=11)
    axes[2].set_ylabel("Percent Change in Gene Count", fontsize=11)
    axes[2].set_title(f"Depth Change vs Gene Change\nSpearman r={rho_change:.3f}, p={p_change:.2e}", fontsize=11)
    axes[2].axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].axvline(x=0, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.tight_layout()
    _save_dual(fig, output_dir, "hierarchy_vs_percent_change")
    plt.close(fig)
    print(f"Saved {output_dir / 'hierarchy_vs_percent_change.pdf'}")


if __name__ == "__main__":
    main()
