# media/generate_simple_gif.py
"""
Minimal GIF generator using Pillow — no manim, no LaTeX.

Useful when you want a lightweight animation of an existing figure
(e.g. the QSE convergence curve growing with Krylov dimension).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def render_curve_gif(out_path: Path, fps: int = 2) -> None:
    """Animate the QSE convergence curve growing one R at a time."""
    r_vals = np.arange(1, 9)
    qse = np.array([
        -1.1287, -1.1553, -1.1570, -1.1600, -1.1607,
        -1.1607, -1.1607, -1.1611,
    ])
    exact = -1.1627

    frames = []
    for end in range(1, len(r_vals) + 1):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(r_vals[:end], qse[:end], "o-", color="#1f77b4",
                linewidth=2, markersize=6)
        ax.axhline(exact, color="red", linestyle="--",
                   label=f"Exact = {exact:.4f}")
        ax.set_xlim(0.5, 8.5)
        ax.set_ylim(-1.165, -1.125)
        ax.set_xlabel("Krylov dimension R")
        ax.set_ylabel("Energy (Ha)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )


if __name__ == "__main__":
    render_curve_gif(Path("media/output/qse_curve.gif"))