"""Shared plotting palette and axis styling for benchmark figures."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt


# =====================================================================
# Colours
# =====================================================================

METHOD_PALETTE: dict[str, str] = {
    # Canonical names
    "OVOS":       "#1f77b4",
    "COVO":       "#2ca02c",
    "RHF":        "#d62728",
    "AVAS":       "#ff7f0e",
    "NO-MP2":     "#9467bd",
    "BASELINE":   "#8c564b",
    # Aliases for the folder names in the data directory
    "COVO_ORB":   "#2ca02c",
    "NO_MP2":     "#9467bd",
    "NOMP2":      "#9467bd",
    "MP2-NO":     "#9467bd",
    "MP2_NO":     "#9467bd",
}
DEFAULT_METHOD_COLOR = "#7f7f7f"

ALGO_PALETTE: dict[str, str] = {
    "QSE":       "#1f77b4",
    "SQD":       "#ff7f0e",
    "SKQD":      "#2ca02c",
    "HSB-QSCI":  "#d62728",
    "TE-QSCI":   "#9467bd",
}
DEFAULT_ALGO_COLOR = "#7f7f7f"

ALGO_MARKER: dict[str, str] = {
    "QSE":       "o",
    "SQD":       "s",
    "SKQD":      "^",
    "HSB-QSCI":  "D",
    "TE-QSCI":   "v",
}
DEFAULT_ALGO_MARKER = "o"


def method_color(method: str) -> str:
    return METHOD_PALETTE.get(method.upper(), DEFAULT_METHOD_COLOR)


def algo_color(algorithm: str) -> str:
    return ALGO_PALETTE.get(algorithm.upper(), DEFAULT_ALGO_COLOR)


def algo_marker(algorithm: str) -> str:
    return ALGO_MARKER.get(algorithm.upper(), DEFAULT_ALGO_MARKER)


# =====================================================================
# Axes
# =====================================================================

def style_axes(
    ax: plt.Axes,
    xlabel: str = "",
    ylabel: str = "",
    title: Optional[str] = None,
    logx: bool = False,
    logy: bool = False,
    grid: bool = True,
) -> None:
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if grid:
        ax.grid(True, alpha=0.3, which="both")


def add_chemical_accuracy_line(ax: plt.Axes, y: float = 1.0) -> None:
    ax.axhline(y, color="grey", linestyle=":", linewidth=1.2,
               label=f"{y:g} mHa")


def save_figure(fig: plt.Figure, path, dpi: int = 150) -> None:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)