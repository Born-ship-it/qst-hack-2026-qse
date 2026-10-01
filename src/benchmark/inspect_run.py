#!/usr/bin/env python3
"""
Inspect a saved run without re-executing the quantum sweep.

Examples
--------
    # Summary of one run
    python scripts/inspect_run.py \\
        --artifacts results/h2_ccpvdz \\
        --run-key h2/cc-pvdz/OVOS/initfoh_virtual

    # Export raw arrays to CSV
    python scripts/inspect_run.py \\
        --artifacts results/h2_ccpvdz \\
        --run-key h2/cc-pvdz/OVOS/initfoh_virtual \\
        --dump-csv /tmp/ovos_initfoh

    # Plot fidelity of the QSE solution with the exact GS vs Krylov R
    python scripts/inspect_run.py \\
        --artifacts results/h2_ccpvdz \\
        --run-key h2/cc-pvdz/OVOS/initfoh_virtual \\
        --plot-fidelity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.artifacts import load_exact, load_sweep  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, required=True,
                   help="Directory containing runs.parquet, exact/, sweeps_raw/")
    p.add_argument("--run-key", type=str, required=True,
                   help="e.g. h2/cc-pvdz/OVOS/initfoh_virtual")
    p.add_argument("--algorithm", type=str, default=None,
                   choices=["QSE", "SQD"])
    p.add_argument("--dump-csv", type=Path, default=None)
    p.add_argument("--plot-fidelity", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Summary row
    runs = pd.read_parquet(args.artifacts / "runs.parquet")
    row = runs[runs["run_key"] == args.run_key]
    if row.empty:
        print(f"Run key not found in runs.parquet: {args.run_key}")
        return 1
    print("--- Run summary ---")
    for k, v in row.iloc[0].items():
        print(f"  {k:>28}: {v}")

    # Exact eigendecomposition
    exact = load_exact(args.artifacts, args.run_key)
    print(f"\n--- Exact 2e spectrum ---")
    print(f"  dim:                {len(exact.evals)}")
    print(f"  E0 (electronic):    {exact.evals[0]:+.8f}")
    print(f"  E1 (electronic):    {exact.evals[1]:+.8f}")
    print(f"  E_max (electronic): {exact.evals[-1]:+.8f}")
    print(f"  HF index:           {exact.ref_index}")
    print(f"  |<HF|GS>|^2:        {exact.hf_gs_overlap:.6f}")

    # Sweep artifacts
    algorithms = [args.algorithm] if args.algorithm else ["QSE", "SQD"]
    for algo in algorithms:
        try:
            art = load_sweep(args.artifacts, args.run_key, algo)
        except FileNotFoundError:
            print(f"\n[no {algo} artifact]")
            continue
        print(f"\n--- {algo} sweep ---")
        print(f"  krylov dims: {art.krylov_dims}")
        print(f"  energies:    {np.round(art.energies_total, 6)}")
        print(f"  cond nums:   {np.round(art.condition_numbers, 2)}")

        if algo == "QSE" and art.s_row is not None:
            print(f"  s_row (first 5): {np.round(art.s_row[:5], 4)}")
            print(f"  h_row (first 5): {np.round(art.h_row[:5], 4)}")

        if algo == "SQD" and art.subspace_bitstrings is not None:
            for r, (bs, counts) in zip(art.krylov_dims, zip(
                art.subspace_bitstrings, art.sampled_counts or [],
            )):
                print(f"  R={r}: {len(bs)} configs")

    # Optional CSV dump
    if args.dump_csv:
        args.dump_csv.mkdir(parents=True, exist_ok=True)
        _dump_csv(args, algorithms)

    # Optional fidelity plot
    if args.plot_fidelity:
        _plot_fidelity(args, algorithms)

    return 0


def _dump_csv(args, algorithms):
    for algo in algorithms:
        try:
            art = load_sweep(args.artifacts, args.run_key, algo)
        except FileNotFoundError:
            continue
        df = pd.DataFrame({
            "krylov_dim": art.krylov_dims,
            "energy_total": art.energies_total,
            "condition_number": art.condition_numbers,
            "retained_dim": art.retained_dims,
        })
        out = args.dump_csv / f"{algo.lower()}_summary.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {out}")

        # Coeffs
        coeffs_rows = []
        for r, c in zip(art.krylov_dims, art.coeffs):
            for i, ci in enumerate(c):
                coeffs_rows.append({"R": r, "index": i,
                                    "re": ci.real, "im": ci.imag})
        pd.DataFrame(coeffs_rows).to_csv(
            args.dump_csv / f"{algo.lower()}_coeffs.csv", index=False,
        )


def _plot_fidelity(args, algorithms):
    """Plot |<QSD Ritz vector | exact GS>|^2 vs Krylov dimension."""
    exact = load_exact(args.artifacts, args.run_key)

    fig, ax = plt.subplots(figsize=(7, 5))
    for algo in algorithms:
        try:
            art = load_sweep(args.artifacts, args.run_key, algo)
        except FileNotFoundError:
            continue

        fidelities = []
        for r, coeffs in zip(art.krylov_dims, art.coeffs):
            fidelities.append(_qsd_fidelity(
                exact, art, r, coeffs, algo,
            ))
        ax.plot(art.krylov_dims, fidelities,
                marker="o", linewidth=2, label=algo)

    ax.set_xlabel("Krylov dimension R")
    ax.set_ylabel("|<QSD Ritz | exact GS>|²")
    ax.set_title(f"Fidelity with exact GS — {args.run_key}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = args.artifacts / "figures" / f"inspect_fidelity_{args.run_key.replace('/', '__')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


def _qsd_fidelity(exact, art, r, coeffs, algorithm):
    """
    Compute |<QSD Ritz vector | exact GS>|^2.

    For QSE: the Ritz vector is a Krylov combination; we project the exact
    GS onto the Krylov basis and take the inner product with `coeffs`.

    For SQD: the Ritz vector is a subspace combination; we project the exact
    GS onto the sampled basis and take the inner product.
    """
    if len(coeffs) == 0:
        return np.nan
    if algorithm == "SQD":
        bs = art.subspace_bitstrings[list(art.krylov_dims).index(r)]
        gs_vector = _gs_on_bitstrings(exact, bs)
        return float(abs(coeffs.conj() @ gs_vector) ** 2)
    # QSE: Krylov basis is not directly orthogonal; a proper projection
    # would need the Krylov Gram matrix.  We approximate by the S-projection
    # of the exact GS onto the Krylov span, then take overlap with coeffs.
    # (See the README for the exact procedure.)
    return np.nan  # placeholder; extend once the Krylov Gram is stored


def _gs_on_bitstrings(exact, bitstrings):
    """Coordinates of the exact GS restricted to a set of basis bitstrings."""
    index_map = {bs: i for i, bs in enumerate(exact.bitstrings)}
    coords = np.zeros(len(bitstrings), dtype=complex)
    for j, bs in enumerate(bitstrings):
        coords[j] = exact.evecs[index_map[bs], 0]
    return coords


if __name__ == "__main__":
    raise SystemExit(main())
