"""
Utility functions: GEVP, exact diagonalization, plotting.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from qiskit.quantum_info import SparsePauliOp


# =====================================================================
# GEVP
# =====================================================================

def solve_thresholded_gevp(
    h_matrix: np.ndarray,
    s_matrix: np.ndarray,
    threshold: float = 1e-10,
) -> tuple[float, np.ndarray, int]:
    """
    Solve H c = E S c via canonical orthogonalization with thresholding.

    Returns
    -------
    e_min : float
        Lowest eigenvalue.
    coeffs : np.ndarray
        Corresponding eigenvector in the original basis.
    retained : int
        Number of overlap eigenvalues above threshold.
    """
    s_vals, s_vecs = la.eigh(s_matrix)
    valid = s_vals > threshold
    if not np.any(valid):
        raise ValueError("All overlap eigenvalues were removed by thresholding.")

    ortho = s_vecs[:, valid] @ np.diag(1.0 / np.sqrt(s_vals[valid]))
    h_orth = ortho.conj().T @ h_matrix @ ortho
    h_orth = 0.5 * (h_orth + h_orth.conj().T)

    eigvals, eigvecs = la.eigh(h_orth)
    coeffs = ortho @ eigvecs[:, 0]

    norm = np.sqrt(np.real(coeffs.conj().T @ s_matrix @ coeffs))
    if norm > 0:
        coeffs = coeffs / norm

    return float(np.real(eigvals[0])), coeffs, int(np.sum(valid))


# =====================================================================
# Exact benchmarks
# =====================================================================

def compute_exact_ground_state(
    hamiltonian: SparsePauliOp,
    num_qubits: Optional[int] = None,
) -> float:
    """Full diagonalization (only feasible up to ~15 qubits)."""
    if num_qubits is None:
        num_qubits = hamiltonian.num_qubits
    if num_qubits > 15:
        raise ValueError(
            f"Full diagonalization requires 2^{num_qubits} memory. "
            f"Current limit is 15 qubits."
        )
    return float(np.min(la.eigvalsh(hamiltonian.to_matrix())))


def compute_exact_ground_state_subspace(
    hamiltonian: SparsePauliOp,
    subspace_bitstrings: list[str],
) -> float:
    """Exact ground state within a specified subspace."""
    num_qubits = hamiltonian.num_qubits
    dim = len(subspace_bitstrings)
    h = np.zeros((dim, dim), dtype=complex)
    for i, bra in enumerate(subspace_bitstrings):
        for j, ket in enumerate(subspace_bitstrings):
            h[i, j] = _matrix_element(hamiltonian, bra, ket)
    h = 0.5 * (h + h.conj().T)
    return float(np.min(la.eigvalsh(h)))


def _matrix_element(
    hamiltonian: SparsePauliOp,
    bra_bs: str,
    ket_bs: str,
) -> complex:
    n = hamiltonian.num_qubits
    bra, ket = int(bra_bs, 2), int(ket_bs, 2)
    val = 0.0 + 0.0j
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = pauli.to_label()
        new_state = ket
        phase = 1.0 + 0.0j
        for q in range(n):
            p = label[n - 1 - q]
            if p == "I":
                continue
            elif p == "Z":
                if (new_state >> q) & 1:
                    phase *= -1
            elif p == "X":
                new_state ^= 1 << q
            elif p == "Y":
                phase *= 1j if not ((new_state >> q) & 1) else -1j
                new_state ^= 1 << q
        if new_state == bra:
            val += coeff * phase
    return val


# =====================================================================
# Plotting
# =====================================================================

def plot_energy_convergence(
    energies: list[float],
    exact_energy: Optional[float] = None,
    krylov_dim: Optional[int] = None,
    title: str = "Energy Convergence",
    label: str = "QSE estimate",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if krylov_dim is None:
        krylov_dim = len(energies)
    dims = range(1, krylov_dim + 1)

    ax.plot(dims, energies, marker="o", linewidth=2, markersize=8, label=label)
    if exact_energy is not None:
        ax.axhline(
            exact_energy, linestyle="--", color="red", linewidth=2,
            label=f"Exact = {exact_energy:.6f}",
        )

    ax.set_xlabel("Krylov dimension", fontsize=12)
    ax.set_ylabel("Ground-state energy (Ha)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    return ax


def plot_comparison(
    qse_energies: list[float],
    sqd_energies: list[float],
    exact_energy: float,
    title: str = "QSE vs SQD Comparison",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(qse_energies) + 1), qse_energies,
            marker="o", linewidth=2, markersize=8, label="QSE")
    ax.plot(range(1, len(sqd_energies) + 1), sqd_energies,
            marker="s", linewidth=2.5, markersize=10,
            color="orange", zorder=5, label="SQD")
    ax.axhline(exact_energy, linestyle="--", color="red", linewidth=2,
               label=f"Exact = {exact_energy:.6f}")
    ax.set_xlabel("Iteration / Krylov dimension", fontsize=12)
    ax.set_ylabel("Ground-state energy (Ha)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig


def compute_energy_error(estimated: float, exact: float) -> tuple[float, float]:
    abs_err = abs(estimated - exact)
    rel_err = abs_err / abs(exact) if exact != 0 else float("inf")
    return abs_err, rel_err
