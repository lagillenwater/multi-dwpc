#!/usr/bin/env python3
"""Render documentation flowcharts:

    output/flowcharts/year_go_term_filtering_flow.pdf
    output/flowcharts/sensitivity_analyses_flow.pdf

Run from the repo root so the repo `matplotlibrc` is loaded (TrueType output):

    python3 scripts/visualization/make_documentation_flowcharts.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "output" / "flowcharts"

DATA_FACE = "#FFF4E1"
DATA_EDGE = "#B06C00"
PROCESS_FACE = "#E6F0FA"
PROCESS_EDGE = "#1F4E79"
DECISION_FACE = "#EAF7E6"
DECISION_EDGE = "#2E7D32"
OUTPUT_FACE = "#F3E5F5"
OUTPUT_EDGE = "#6A1B9A"
TEXT_COLOR = "#212121"
ARROW_COLOR = "#5A5A5A"


def draw_box(ax, xy, width, height, label, face, edge,
             fontsize=10, fontweight="normal", pad=0.02):
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width, height,
        boxstyle=f"round,pad={pad},rounding_size=0.08",
        linewidth=1.2, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=TEXT_COLOR)


def arrow(ax, start, end):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=12, linewidth=1.0,
        color=ARROW_COLOR,
    )
    ax.add_patch(arr)


def add_legend(ax):
    patches = [
        mpatches.Patch(facecolor=DATA_FACE, edgecolor=DATA_EDGE, label="Data / artifact"),
        mpatches.Patch(facecolor=PROCESS_FACE, edgecolor=PROCESS_EDGE, label="Processing step"),
        mpatches.Patch(facecolor=DECISION_FACE, edgecolor=DECISION_EDGE, label="Filter / selection"),
        mpatches.Patch(facecolor=OUTPUT_FACE, edgecolor=OUTPUT_EDGE, label="Output"),
    ]
    ax.legend(
        handles=patches,
        loc="lower center", bbox_to_anchor=(0.5, -0.04),
        fontsize=9, frameon=True, framealpha=0.95, edgecolor="#999",
        ncol=4,
    )


# ---------------------------------------------------------------------------
# Filtering chart
# ---------------------------------------------------------------------------

def make_filtering_chart() -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5, 12.4, "GO term filtering pipeline",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color=TEXT_COLOR)

    # Top inputs
    draw_box(ax, (2.6, 11.0), 3.4, 0.65, "2016 GO-gene pairs", DATA_FACE, DATA_EDGE)
    draw_box(ax, (7.4, 11.0), 3.4, 0.65, "2024 GO-gene pairs", DATA_FACE, DATA_EDGE)

    # Intersect
    draw_box(ax, (5, 9.7), 3.6, 0.6, "Intersect", PROCESS_FACE, PROCESS_EDGE)
    arrow(ax, (2.6, 10.66), (4.0, 10.0))
    arrow(ax, (7.4, 10.66), (6.0, 10.0))

    # Filters
    draw_box(ax, (5, 8.4), 5.0, 0.6,
             "Positive percent-change filter", DECISION_FACE, DECISION_EDGE)
    arrow(ax, (5, 9.39), (5, 8.7))

    draw_box(ax, (5, 7.1), 5.0, 0.6,
             "IQR filter on percent change", DECISION_FACE, DECISION_EDGE)
    arrow(ax, (5, 8.1), (5, 7.4))

    draw_box(ax, (5, 5.6), 6.0, 0.95,
             "Jaccard similarity filter\n"
             "greedy pairwise removal of GO terms with Jaccard > 0.1",
             DECISION_FACE, DECISION_EDGE)
    arrow(ax, (5, 6.79), (5, 6.08))

    # Outputs (NEW): branch from Jaccard step into 2016 / 2024 filtered.
    draw_box(ax, (2.6, 3.8), 3.4, 0.65, "2016 filtered", OUTPUT_FACE, OUTPUT_EDGE,
             fontweight="bold")
    draw_box(ax, (7.4, 3.8), 3.4, 0.65, "2024 filtered", OUTPUT_FACE, OUTPUT_EDGE,
             fontweight="bold")
    arrow(ax, (4.0, 5.12), (2.7, 4.13))
    arrow(ax, (6.0, 5.12), (7.3, 4.13))

    add_legend(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "year_go_term_filtering_flow.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Sensitivity chart
# ---------------------------------------------------------------------------

def make_sensitivity_chart() -> Path:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(6.5, 9.5, "Sensitivity analyses overview",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color=TEXT_COLOR)

    # Inputs
    draw_box(ax, (4.5, 8.4), 3.4, 0.65, "Real DWPC scores", DATA_FACE, DATA_EDGE)
    draw_box(ax, (8.5, 8.4), 3.4, 0.65, "Null DWPC scores", DATA_FACE, DATA_EDGE)

    # DWPC compute
    draw_box(ax, (6.5, 7.2), 4.6, 0.6, "DWPC compute", PROCESS_FACE, PROCESS_EDGE)
    arrow(ax, (4.5, 8.07), (5.5, 7.5))
    arrow(ax, (8.5, 8.07), (7.5, 7.5))

    # Four parallel sensitivity sweeps
    sweep_y = 5.7
    sweeps = [
        (1.9, "Null variance\nsweep"),
        (5.4, "Rank stability\nsweep"),
        (8.9, "DWPC z-filter\nsweep"),
        (11.7, "Path enumeration\nsweep"),
    ]
    for x, label in sweeps:
        draw_box(ax, (x, sweep_y), 2.4, 1.0, label, DECISION_FACE, DECISION_EDGE)
        arrow(ax, (6.5, 6.9), (x, sweep_y + 0.5))

    # Decisions: B and z-thresholds
    draw_box(ax, (3.6, 3.6), 3.4, 0.65, "Optimal B", DECISION_FACE, DECISION_EDGE)
    draw_box(ax, (10.3, 3.6), 3.4, 0.65, "z-thresholds", DECISION_FACE, DECISION_EDGE)
    arrow(ax, (1.9, 5.2), (2.8, 3.93))
    arrow(ax, (5.4, 5.2), (4.4, 3.93))
    arrow(ax, (8.9, 5.2), (9.7, 3.93))
    arrow(ax, (11.7, 5.2), (10.9, 3.93))

    # Output
    draw_box(ax, (6.5, 1.9), 4.6, 0.7, "Intermediate sharing",
             OUTPUT_FACE, OUTPUT_EDGE, fontweight="bold")
    arrow(ax, (3.6, 3.27), (5.4, 2.25))
    arrow(ax, (10.3, 3.27), (7.6, 2.25))

    add_legend(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "sensitivity_analyses_flow.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print(f"wrote {make_filtering_chart()}")
    print(f"wrote {make_sensitivity_chart()}")


if __name__ == "__main__":
    main()
