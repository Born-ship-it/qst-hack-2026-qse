"""Shared plotting palette and axis styling for benchmark figures."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

# Consistent colours across all figures
METHOD_PALETTE: dict[str, str] = {
    "OVOS":     "#1f77b4",
    "COVO":     "#2ca02c",
    "RHF":      "#d62728",
    "AVAS":     "#ff7f0e",
    "NO-MP2":   "#9467bd",
    "NOMP2":    "#9467bd",  # alt spelling
    "MP2-NO":   "#9467bd",  # alt spelling
}
DEFAULT_METHOD_COLOR = "#7f7f7f"

ALGO_PALETTE: dict[str, str] = {
    "QSE":       "#1f77b4",
    "SQD":       "#ff7f0e",
    "HSB-QSCI":  "#2ca02c",
    "TE-QSCI":   "#d62728",
}
DEFAULT_ALGO_COLOR = "#7f7f7f"


def method_color(method: str) -> str:
    """Colour for a given orbital method, falling back to grey."""
    return METHOD_PALETTE.get(method.upper(), DEFAULT_METHOD_COLOR)


def algo_color(algorithm: str) -> str:
    """Colour for a given QSD algorithm, falling back to grey."""
    return ALGO_PALETTE.get(algorithm.upper(), DEFAULT_ALGO_COLOR)


def style_axes(
    ax: plt.Axes,
    xlabel: str = "",
    ylabel: str = "",
    title: Optional[str] = None,
    logx: bool = False,
    logy: bool = False,
    grid: bool = True,
) -> None:
    """Apply a consistent style to a matplotlib Axes object."""
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
    """Draw a horizontal reference line at chemical accuracy (1 mHa by default)."""
    ax.axhline(y, color="grey", linestyle=":", linewidth=1.2,
               label=f"{y:g} mHa")


def save_figure(fig: plt.Figure, path, dpi: int = 150) -> None:
    """Save a figure with sensible defaults and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
