"""
One function per figure in the orbital-method benchmarking study.

All plot functions take two pandas DataFrames (runs, sweeps) and write a
PNG to the provided path.  See the README for the study description and
the paper each figure is inspired by.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    add_chemical_accuracy_line,
    algo_color,
    method_color,
    save_figure,
    style_axes,
)


# =====================================================================
# Fig 1 — Accuracy vs N_A per method
# =====================================================================

def fig01_accuracy_vs_ncas(runs: pd.DataFrame, out: Path) -> None:
    """Active-space CASCI error vs number of active orbitals."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, sub in runs.groupby("method"):
        sub = sub.sort_values("num_qubits")
        ncas = sub["num_qubits"] / 2
        err_mha = (sub["casci_total"] - sub["fci_total"]) * 1000.0
        ax.plot(ncas, err_mha, marker="o", linewidth=2,
                color=method_color(method), label=method)
    add_chemical_accuracy_line(ax, 1.0)
    style_axes(ax, "Active orbitals (N_A)", "E_CASCI − E_FCI (mHa)",
               "Active-space accuracy vs orbital method", logy=True)
    ax.legend(fontsize=9)
    save_figure(fig, out)


# =====================================================================
# Fig 2 — Quantum resource scaling
# =====================================================================

def fig02_resource_scaling(runs: pd.DataFrame, out: Path) -> None:
    """Pauli-term count, commuting groups, and Trotter cost vs qubits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("num_pauli_terms", "Pauli terms"),
        ("num_commuting_groups", "Commuting groups"),
        ("trotter_2q_gates", "Trotter 2Q gates"),
    ]
    for ax, (col, label) in zip(axes, panels):
        for method, sub in runs.groupby("method"):
            sub = sub.sort_values("num_qubits")
            ax.plot(sub["num_qubits"], sub[col], marker="o", linewidth=2,
                    color=method_color(method), label=method)
        style_axes(ax, "Qubits", label, logx=True, logy=True)
    axes[0].legend(fontsize=9)
    fig.suptitle("Quantum resource scaling", fontsize=12, y=1.02)
    save_figure(fig, out)


# =====================================================================
# Fig 3 — Convergence at fixed N_A
# =====================================================================

def fig03_convergence_fixed_ncas(
    runs: pd.DataFrame, sweeps: pd.DataFrame, out: Path, ncas: int = 6,
) -> None:
    """
    Two-panel figure: absolute quality (vs FCI) and within-method
    convergence (vs CASCI), for each method at a fixed active-space size.

    The two panels answer different questions:
    - left:  "which method gives the best answer overall?"
    - right: "how fast does each method converge to its own CASCI limit?"
    """
    target_qubits = 2 * ncas
    subset_runs = runs[runs["num_qubits"] == target_qubits]
    if subset_runs.empty:
        return

    methods = sorted(subset_runs["method"].unique())
    n = len(methods)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        2, rows * cols,
        figsize=(5 * rows * cols / 2, 8), squeeze=False,
    )

    for col, method in enumerate(methods):
        keys = subset_runs[subset_runs["method"] == method]["run_key"]
        sub = sweeps[sweeps["run_key"].isin(keys)]

        for algo, algo_sub in sub.groupby("algorithm"):
            algo_sub = algo_sub.sort_values("krylov_dim")
            for row, (metric, label) in enumerate((
                ("err_vs_fci_mha",    "Error vs FCI (mHa)"),
                ("err_vs_casci_mha",  "Error vs CASCI (mHa)"),
            )):
                ax = axes[row, col]
                ax.plot(algo_sub["krylov_dim"], algo_sub[metric],
                        marker="o", linewidth=2, color=algo_color(algo),
                        label=algo)

        for row in range(2):
            ax = axes[row, col]
            add_chemical_accuracy_line(ax, 1.0)
            style_axes(
                ax, "Krylov dimension R",
                "Error vs FCI (mHa)" if row == 0 else "Error vs CASCI (mHa)",
                title=f"{method}  (N_A={ncas})" if row == 0 else None,
                logy=True,
            )
            ax.legend(fontsize=9)

    # Hide unused subplots
    for i in range(n, rows * cols):
        for row in range(2):
            axes[row, i].set_visible(False)

    fig.tight_layout()
    save_figure(fig, out)



# =====================================================================
# Fig 4 — R_min / R_GS scaling
# =====================================================================

def fig04_rmin_over_rgs(
    runs: pd.DataFrame, sweeps: pd.DataFrame, out: Path,
) -> None:
    """Subspace dimension needed for 1 mHa accuracy, normalised to GS-QSCI."""
    # R_min from QSE sweep at 1 mHa
    rows = []
    for key, sub in sweeps[sweeps["algorithm"] == "QSE"].groupby("run_key"):
        hit = sub[sub["err_vs_casci_mha"] < 1.0]
        if hit.empty:
            continue
        r_min = hit["krylov_dim"].min()
        meta = runs[runs["run_key"] == key].iloc[0]
        rows.append({
            "run_key": key,
            "method": meta["method"],
            "num_qubits": meta["num_qubits"],
            "R_min": r_min,
            # Proxy for R_GS: full 2e dimension is C(n_q, n_e); we use a
            # conservative estimate if the exact value isn't stored.
            "R_GS": _rgs_estimate(meta["num_qubits"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    df["ratio"] = df["R_min"] / df["R_GS"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, sub in df.groupby("method"):
        ax.scatter(sub["num_qubits"], sub["ratio"],
                   color=method_color(method), label=method, s=60)
        if len(sub) >= 2:
            coef = np.polyfit(sub["num_qubits"], sub["ratio"], 1)
            xs = np.linspace(sub["num_qubits"].min(), sub["num_qubits"].max(), 20)
            ax.plot(xs, np.polyval(coef, xs), linestyle="--", alpha=0.5,
                    color=method_color(method))
    style_axes(ax, "Qubits", "R_min / R_GS",
               "Subspace overhead vs system size")
    ax.legend(fontsize=9)
    save_figure(fig, out)


def _rgs_estimate(num_qubits: int) -> float:
    """Cheap upper bound for the 2-electron subspace dimension."""
    from math import comb
    return float(comb(num_qubits, 2))


# =====================================================================
# Fig 5 — Condition number of the overlap matrix
# =====================================================================

def fig05_condition_number(
    runs: pd.DataFrame, sweeps: pd.DataFrame, out: Path,
) -> None:
    """log10(κ(S)) vs Krylov dimension, one curve per method."""
    fig, ax = plt.subplots(figsize=(7, 5))
    subset = sweeps[sweeps["algorithm"] == "QSE"].copy()
    for method, sub in subset.groupby("method"):
        avg = sub.groupby("krylov_dim")["condition_number"].median()
        ax.plot(avg.index, np.log10(avg.values), marker="o", linewidth=2,
                color=method_color(method), label=method)
    ax.axhline(np.log10(1e8), color="grey", linestyle=":", linewidth=1.2,
               label="κ = 1e8")
    style_axes(ax, "Krylov dimension R", "log10 κ(S)",
               "Overlap-matrix conditioning")
    ax.legend(fontsize=9)
    save_figure(fig, out)


# =====================================================================
# Fig 6 — Noise robustness (single-bit-flip model)
# =====================================================================

def fig06_noise_robustness(
    runs: pd.DataFrame, sweeps: pd.DataFrame, out: Path,
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.20),
) -> None:
    """Placeholder — noise sweep is run by scripts/noise_sweep.py."""
    # This figure is generated by scripts/noise_sweep.py, which writes
    # a separate noise.parquet with columns [run_key, method, noise,
    # min_err_mha].  If that file exists next to sweeps.parquet, plot it.
    noise_path = out.parent / "noise.parquet"
    if not noise_path.exists():
        return
    noise_df = pd.read_parquet(noise_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, sub in noise_df.groupby("method"):
        agg = sub.groupby("noise")["min_err_mha"].median()
        ax.plot(agg.index, agg.values, marker="o", linewidth=2,
                color=method_color(method), label=method)
    style_axes(ax, "Noise level δ", "min_R err vs CASCI (mHa)",
               "SQD robustness to readout noise", logy=True)
    ax.legend(fontsize=9)
    save_figure(fig, out)


# =====================================================================
# Fig 7 — Initial overlap vs R_min
# =====================================================================

def fig07_overlap_vs_rmin(
    runs: pd.DataFrame, sweeps: pd.DataFrame, out: Path,
) -> None:
    """R_min vs initial ground-state overlap, one colour per method."""
    rows = []
    for key, sub in sweeps[sweeps["algorithm"] == "QSE"].groupby("run_key"):
        hit = sub[sub["err_vs_casci_mha"] < 1.0]
        if hit.empty:
            continue
        meta = runs[runs["run_key"] == key].iloc[0]
        rows.append({
            "method": meta["method"],
            "initial_overlap": meta["initial_overlap"],
            "R_min": hit["krylov_dim"].min(),
            "num_qubits": meta["num_qubits"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, sub in df.groupby("method"):
        ax.scatter(sub["initial_overlap"], sub["R_min"],
                   s=sub["num_qubits"] * 3,
                   color=method_color(method), label=method, alpha=0.7)
    style_axes(ax, "|<HF|GS>|²", "R_min for 1 mHa",
               "Reference overlap vs convergence cost")
    ax.legend(fontsize=9)
    save_figure(fig, out)


# =====================================================================
# Fig 8 — Pareto frontier
# =====================================================================

def fig08_pareto_frontier(
    runs: pd.DataFrame, sweeps: pd.DataFrame, out: Path,
    num_samples: int = 20_000,
) -> None:
    """Accuracy vs total resource cost, one point per (run, algorithm, R)."""
    merged = sweeps.merge(
        runs[["run_key", "method", "trotter_2q_gates"]],
        on="run_key", how="left",
    )
    # Cost estimate: 2Q gates × shots × subspace dimension
    merged["cost"] = (
        merged["trotter_2q_gates"] * num_samples * merged["krylov_dim"]
    )
    merged = merged.dropna(subset=["cost", "err_vs_fci_mha"])

    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for method, sub in merged.groupby("method"):
        for algo, algo_sub in sub.groupby("algorithm"):
            ax.scatter(algo_sub["cost"], algo_sub["err_vs_fci_mha"],
                       color=method_color(method),
                       marker="o" if algo == "QSE" else "s",
                       s=40, alpha=0.7,
                       label=f"{method}·{algo}")
    # Pareto frontier
    df = merged.sort_values("cost").reset_index(drop=True)
    pareto = []
    best = float("inf")
    for _, row in df.iterrows():
        if row["err_vs_fci_mha"] < best:
            pareto.append((row["cost"], row["err_vs_fci_mha"]))
            best = row["err_vs_fci_mha"]
    if pareto:
        xs, ys = zip(*pareto)
        ax.plot(xs, ys, "k--", linewidth=2, label="Pareto frontier")

    style_axes(ax, "Total resource cost (arb. units)",
               "Error vs FCI (mHa)", "Accuracy vs total resource cost",
               logx=True, logy=True)
    # Deduplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, ncol=2)
    save_figure(fig, out)


# =====================================================================
# Fig 9 — Runtime breakdown
# =====================================================================

def fig09_runtime_breakdown(
    runs: pd.DataFrame, out: Path,
) -> None:
    """Stacked bar chart of synthesis time per method."""
    if runs.empty:
        return
    agg = runs.groupby("method")["synthesis_seconds"].median().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(agg.index, agg.values,
            color=[method_color(m) for m in agg.index])
    ax.set_xlabel("Median Trotter synthesis time (s)", fontsize=11)
    ax.set_title("Synthesis cost per orbital method", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    save_figure(fig, out)


# =====================================================================
# Fig 10 — Summary heatmap
# =====================================================================

def fig10_summary_heatmap(
    runs: pd.DataFrame, sweeps: pd.DataFrame, out: Path,
) -> None:
    """min_R err_vs_casci per (method, algorithm) at fixed resource budget."""
    df = sweeps.copy()
    df = df.merge(runs[["run_key", "num_qubits"]], on="run_key", how="left")

    # Fixed budget: max error over the smallest 8-qubit runs
    budget_qubits = int(df["num_qubits"].min())
    df = df[df["num_qubits"] == budget_qubits]

    table = (
        df.groupby(["method", "algorithm"])["err_vs_casci_mha"]
        .min()
        .unstack(fill_value=np.nan)
    )
    if table.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 0.6 * len(table) + 2))
    im = ax.imshow(np.log10(table.values + 1e-6),
                   cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            v = table.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="log10 min error vs CASCI (mHa)")
    ax.set_title(f"Best achievable accuracy at {budget_qubits} qubits",
                 fontsize=12)
    save_figure(fig, out)


# =====================================================================
# Driver
# =====================================================================

ALL_FIGURES = [
    ("fig01_accuracy_vs_ncas.png",  lambda r, s, o: fig01_accuracy_vs_ncas(r, o)),
    ("fig02_resource_scaling.png",  lambda r, s, o: fig02_resource_scaling(r, o)),
    ("fig03_convergence_fixed_ncas.png", fig03_convergence_fixed_ncas),
    ("fig04_rmin_over_rgs.png",     fig04_rmin_over_rgs),
    ("fig05_condition_number.png",  fig05_condition_number),
    ("fig06_noise_robustness.png",  fig06_noise_robustness),
    ("fig07_overlap_vs_rmin.png",   fig07_overlap_vs_rmin),
    ("fig08_pareto_frontier.png",   fig08_pareto_frontier),
    ("fig09_runtime_breakdown.png", lambda r, s, o: fig09_runtime_breakdown(r, o)),
    ("fig10_summary_heatmap.png",   fig10_summary_heatmap),
]


# Narrow the exceptions caught by generate_all
_OPTIONAL_FIGURE_EXCEPTIONS = (
    FileNotFoundError,        # optional inputs like noise.parquet
    KeyError,                 # a column missing from an optional dataframe
    ValueError,               # e.g. no rows at the requested N_A
)


def generate_all(
    runs: pd.DataFrame, sweeps: pd.DataFrame, out_dir: Path,
) -> None:
    """
    Generate every figure, skipping only the ones whose optional inputs
    are missing. Programming errors (TypeError, AttributeError, etc.)
    propagate so they are not lost.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in ALL_FIGURES:
        path = out_dir / name
        try:
            fn(runs, sweeps, path)
        except _OPTIONAL_FIGURE_EXCEPTIONS as exc:
            print(f"[skip] {name}: {type(exc).__name__}: {exc}")