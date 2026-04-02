#!/usr/bin/env python3
"""Validate direct DWPC concordance against historical API outputs with plots."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT / "src"))

from dwpc_direct import HetMat  # noqa: E402
from dwpc_validation import list_api_metapaths, load_api_results, sample_metapath_concordance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-results",
        default="output/dwpc_com/all_GO_positive_growth/results/res_all_GO_positive_growth_2016_real.csv",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output/dwpc_validation/all_GO_positive_growth")
    parser.add_argument("--n-samples-per-metapath", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metapath-api", default=None)
    parser.add_argument("--list-metapaths", action="store_true")
    return parser.parse_args()


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))


def _plot_metapath_scatter(samples_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(
        samples_df["api_dwpc"].astype(float),
        samples_df["direct_dwpc"].astype(float),
        s=14,
        alpha=0.45,
        edgecolors="none",
    )
    combined = pd.concat(
        [samples_df["api_dwpc"].astype(float), samples_df["direct_dwpc"].astype(float)],
        ignore_index=True,
    )
    lower = float(combined.min())
    upper = float(combined.max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.0, color="#444444")
    ax.set_xlabel("API DWPC")
    ax.set_ylabel("Direct DWPC")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_overall_hexbin(samples_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    hb = ax.hexbin(
        samples_df["api_dwpc"].astype(float),
        samples_df["direct_dwpc"].astype(float),
        gridsize=45,
        mincnt=1,
        cmap="Blues",
    )
    combined = pd.concat(
        [samples_df["api_dwpc"].astype(float), samples_df["direct_dwpc"].astype(float)],
        ignore_index=True,
    )
    lower = float(combined.min())
    upper = float(combined.max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.0, color="#444444")
    ax.set_xlabel("API DWPC")
    ax.set_ylabel("Direct DWPC")
    ax.set_title("Direct vs API DWPC concordance")
    ax.grid(alpha=0.15)
    fig.colorbar(hb, ax=ax, label="Sample count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_abs_diff_hist(samples_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.hist(samples_df["abs_diff"].astype(float), bins=50, color="#4c78a8", alpha=0.8)
    ax.set_xlabel("Absolute difference |direct - api|")
    ax.set_ylabel("Count")
    ax.set_title("DWPC absolute difference distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    api_results_path = REPO_ROOT / args.api_results
    data_dir = REPO_ROOT / args.data_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    api_df = load_api_results(api_results_path)
    metapaths = list_api_metapaths(api_df)
    if args.list_metapaths:
        for metapath in metapaths:
            print(metapath)
        return 0

    if args.metapath_api:
        metapaths = [str(args.metapath_api)]

    hetmat = HetMat(data_dir, damping=0.5, use_disk_cache=True, write_disk_cache=False)
    all_samples = []
    summary_rows = []
    for idx, metapath_api in enumerate(metapaths):
        samples_df, summary = sample_metapath_concordance(
            hetmat,
            api_df,
            data_dir=data_dir,
            metapath_api=metapath_api,
            n_samples=int(args.n_samples_per_metapath),
            seed=int(args.seed) + idx,
        )
        summary_rows.append(summary)
        if samples_df.empty:
            continue
        all_samples.append(samples_df)
        metapath_slug = _safe_slug(summary["metapath_api"])
        _plot_metapath_scatter(
            samples_df,
            output_dir / f"scatter_{metapath_slug}.png",
            title=f"{summary['metapath_api']} direct vs API",
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "concordance_summary.csv", index=False)
    print(f"Saved: {output_dir / 'concordance_summary.csv'}")

    if not all_samples:
        print("No valid DWPC concordance samples were produced.")
        return 0

    all_samples_df = pd.concat(all_samples, ignore_index=True)
    all_samples_df.to_csv(output_dir / "concordance_samples.csv", index=False)
    print(f"Saved: {output_dir / 'concordance_samples.csv'}")

    _plot_overall_hexbin(all_samples_df, output_dir / "concordance_overall_hexbin.png")
    print(f"Saved: {output_dir / 'concordance_overall_hexbin.png'}")
    _plot_abs_diff_hist(all_samples_df, output_dir / "concordance_abs_diff_hist.png")
    print(f"Saved: {output_dir / 'concordance_abs_diff_hist.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
