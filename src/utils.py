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


# =====================================================================
# Shared subspace helpers
# =====================================================================

def subspace_matrix_elements(
    hamiltonian: SparsePauliOp,
    bitstrings: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the projected H and S matrices for a list of computational basis states.

    This is the single source of truth used by SQDSolver, the sweep layer,
    and the metrics layer. Do not duplicate elsewhere.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
    bitstrings : list[str]
        Basis states in Qiskit display convention (leftmost = highest qubit).

    Returns
    -------
    h_mat : np.ndarray, shape (d, d), complex
    s_mat : np.ndarray, shape (d, d), complex
        `s_mat` is the identity because the basis is orthonormal.
    """
    dim = len(bitstrings)
    h_mat = np.zeros((dim, dim), dtype=complex)
    s_mat = np.eye(dim, dtype=complex)
    n = hamiltonian.num_qubits

    for i, bra_bs in enumerate(bitstrings):
        for j, ket_bs in enumerate(bitstrings):
            if i == j:
                h_mat[i, j] = _diag_energy(hamiltonian, bra_bs)
            else:
                h_mat[i, j] = _off_diag_energy(hamiltonian, n, bra_bs, ket_bs)
    return h_mat, s_mat


def diagonalize_two_electron_subspace(
    hamiltonian: SparsePauliOp,
    num_electrons: int = 2,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Enumerate the n_electron subspace and diagonalize the Hamiltonian in it.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
    num_electrons : int

    Returns
    -------
    bitstrings : list[str]
        Basis-state ordering, length = C(num_qubits, num_electrons).
    evals : np.ndarray, shape (d,)
        Sorted eigenvalues in the electronic frame.
    evecs : np.ndarray, shape (d, d)
        Columns are eigenvectors, matching `evals`.
    """
    from itertools import combinations

    n = hamiltonian.num_qubits
    bitstrings: list[str] = []
    for occ in combinations(range(n), num_electrons):
        bits = ["0"] * n
        for q in occ:
            bits[n - 1 - q] = "1"
        bitstrings.append("".join(bits))

    h_mat, _ = subspace_matrix_elements(hamiltonian, bitstrings)
    h_mat = 0.5 * (h_mat + h_mat.conj().T)
    evals, evecs = np.linalg.eigh(h_mat)
    return bitstrings, evals, evecs


def _diag_energy(op: SparsePauliOp, bitstring: str) -> complex:
    state = int(bitstring, 2)
    e = 0.0 + 0.0j
    for pauli, coeff in zip(op.paulis, op.coeffs):
        label = pauli.to_label()
        if all(c in "IZ" for c in label):
            sign = 1.0
            for i, c in enumerate(reversed(label)):
                if c == "Z":
                    sign *= (-1.0) ** ((state >> i) & 1)
            e += coeff * sign
    return e


def _off_diag_energy(
    op: SparsePauliOp, n: int, bra_bs: str, ket_bs: str,
) -> complex:
    bra, ket = int(bra_bs, 2), int(ket_bs, 2)
    val = 0.0 + 0.0j
    for pauli, coeff in zip(op.paulis, op.coeffs):
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