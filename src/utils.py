"""
Utility functions: GEVP, exact diagonalization, plotting.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from qiskit.quantum_info import SparsePauliOp

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

_POPCOUNT_BYTE = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint64
)


def _popcount_u64(x: np.ndarray) -> np.ndarray:
    """Element-wise popcount of a uint64 numpy array."""
    x = np.asarray(x, dtype=np.uint64)
    result = np.zeros_like(x, dtype=np.uint64)
    for shift in range(0, 64, 8):
        byte = (x >> np.uint64(shift)) & np.uint64(0xFF)
        result += _POPCOUNT_BYTE[byte.astype(np.uint8)]
    return result

# =====================================================================
# GEVP
# =====================================================================

def solve_thresholded_gevp(
    h_matrix,
    s_matrix,
    threshold: float = 1e-10,
) -> tuple[float, np.ndarray, int]:
    """Solve H c = E S c.

    Handles dense and sparse input. In this codebase S is the identity
    (orthonormal basis), so the sparse path reduces to a standard
    symmetric eigenvalue problem and uses eigsh.

    Returns
    -------
    e_min : float
    coeffs : np.ndarray, shape (M,)
    retained : int
    """
    from scipy.sparse import issparse
    from scipy.sparse.linalg import eigsh
    import scipy.linalg as la

    is_sparse = issparse(h_matrix) or issparse(s_matrix)

    if is_sparse:
        M = h_matrix.shape[0]
        if M < 4:
            # eigsh needs k < M - 1; fall back to dense for tiny M
            h_dense = h_matrix.toarray() if issparse(h_matrix) else h_matrix
            s_dense = s_matrix.toarray() if issparse(s_matrix) else s_matrix
            return _solve_gevp_dense(h_dense, s_dense, threshold)

        h_csr = h_matrix if issparse(h_matrix) else csr_matrix(h_matrix)
        evals, evecs = eigsh(
            h_csr.astype(complex),
            k=1,
            which="SA",
            tol=max(threshold, 1e-12),
            maxiter=10_000,
        )
        coeffs = evecs[:, 0]
        return float(np.real(evals[0])), coeffs, M

    return _solve_gevp_dense(h_matrix, s_matrix, threshold)


def _solve_gevp_dense(
    h_matrix: np.ndarray,
    s_matrix: np.ndarray,
    threshold: float,
) -> tuple[float, np.ndarray, int]:
    """Dense canonical-orthogonalization GEVP (existing algorithm)."""
    import scipy.linalg as la

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
    """Exact ground state within a specified subspace (vectorized)."""
    h, _ = subspace_matrix_elements(hamiltonian, subspace_bitstrings)
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
    sparse: bool | None = None,
) -> tuple:
    """Build the projected H and S matrices for a list of basis states.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
    bitstrings : list[str]
    sparse : bool, optional
        If None (default), returns a sparse CSR matrix when
        ``len(bitstrings) > 2000`` and a dense ndarray otherwise.
        Force either with an explicit True/False.

    Returns
    -------
    h_mat : ndarray or scipy.sparse.csr_matrix, shape (M, M)
    s_mat : ndarray or scipy.sparse.csr_matrix, shape (M, M)
    """
    from scipy.sparse import csr_matrix, eye as sparse_eye

    M = len(bitstrings)
    n = hamiltonian.num_qubits
    N = len(hamiltonian.coeffs)

    if sparse is None:
        sparse = M > 2000

    ket_ints = np.array([int(bs, 2) for bs in bitstrings], dtype=np.uint64)
    sort_idx = np.argsort(ket_ints)
    sorted_ints = ket_ints[sort_idx]

    x = hamiltonian.paulis.x
    z = hamiltonian.paulis.z
    coeffs = hamiltonian.coeffs

    bit_weights = np.array(
        [np.uint64(1) << np.uint64(n - 1 - q) for q in range(n)],
        dtype=np.uint64,
    )

    if not sparse:
        # Dense path (small M)
        h_mat = np.zeros((M, M), dtype=complex)
        for p in range(N):
            xp, zp, c = x[p], z[p], coeffs[p]
            flip_int = np.uint64(0)
            z_int = np.uint64(0)
            n_y = 0
            for q in range(n):
                if xp[q]:
                    flip_int |= bit_weights[q]
                if zp[q]:
                    z_int |= bit_weights[q]
                if xp[q] and zp[q]:
                    n_y += 1
            i_pow = (1j) ** (n_y % 4)
            targets = ket_ints ^ flip_int
            pos = np.searchsorted(sorted_ints, targets)
            pos_clip = np.clip(pos, 0, M - 1)
            matches = sorted_ints[pos_clip] == targets
            bra_idx = np.where(matches, sort_idx[pos_clip], np.int64(-1))
            pc = _popcount_u64(z_int & ket_ints)
            signs = np.where((pc & np.uint64(1)) != np.uint64(0), -1.0, 1.0)
            contrib = c * i_pow * signs
            valid = bra_idx >= 0
            js = np.where(valid)[0]
            if js.size == 0:
                continue
            is_ = bra_idx[js]
            h_mat[is_, js] += contrib[js]
        s_mat = np.eye(M, dtype=complex)
        return h_mat, s_mat

    # Sparse path (large M). Accumulate COO parts, then build CSR.
    # csr_matrix sums duplicate (i, j) entries, which handles the
    # multiple-Pauli contributions to the same matrix element.
    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    for p in range(N):
        xp, zp, c = x[p], z[p], coeffs[p]
        flip_int = np.uint64(0)
        z_int = np.uint64(0)
        n_y = 0
        for q in range(n):
            if xp[q]:
                flip_int |= bit_weights[q]
            if zp[q]:
                z_int |= bit_weights[q]
            if xp[q] and zp[q]:
                n_y += 1
        i_pow = (1j) ** (n_y % 4)
        targets = ket_ints ^ flip_int
        pos = np.searchsorted(sorted_ints, targets)
        pos_clip = np.clip(pos, 0, M - 1)
        matches = sorted_ints[pos_clip] == targets
        bra_idx = np.where(matches, sort_idx[pos_clip], np.int64(-1))
        pc = _popcount_u64(z_int & ket_ints)
        signs = np.where((pc & np.uint64(1)) != np.uint64(0), -1.0, 1.0)
        contrib = c * i_pow * signs
        valid = bra_idx >= 0
        js = np.where(valid)[0]
        if js.size == 0:
            continue
        rows_parts.append(bra_idx[js].astype(np.int64))
        cols_parts.append(js.astype(np.int64))
        data_parts.append(contrib[js])

    if rows_parts:
        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        data = np.concatenate(data_parts)
    else:
        rows = np.zeros(0, dtype=np.int64)
        cols = np.zeros(0, dtype=np.int64)
        data = np.zeros(0, dtype=complex)

    h_mat = csr_matrix((data, (rows, cols)), shape=(M, M), dtype=complex)
    s_mat = sparse_eye(M, format="csr", dtype=complex)
    return h_mat, s_mat

def _condition_number(s_mat) -> float:
    """Condition number of the overlap matrix.

    Handles both dense and sparse input. S is the identity in this
    codebase (orthonormal computational basis), so the sparse path
    short-circuits to 1.0 — no need to iterate eigenvalues of I.
    """
    from scipy.sparse import issparse
    if issparse(s_mat):
        # S == I here, cond(I) == 1
        return 1.0

    vals = np.linalg.eigvalsh(0.5 * (s_mat + s_mat.conj().T))
    pos = vals[vals > 1e-12]
    if len(pos) < 2:
        return float("inf")
    return float(pos[-1] / pos[0])


def diagonalize_two_electron_subspace(
    hamiltonian,
    num_electrons: int = 2,
    n_states: int = 8,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    from itertools import combinations
    from scipy.sparse import issparse, csr_matrix
    from scipy.sparse.linalg import eigsh

    n = hamiltonian.num_qubits
    bitstrings: list[str] = []
    for occ in combinations(range(n), num_electrons):
        bits = ["0"] * n
        for q in occ:
            bits[n - 1 - q] = "1"
        bitstrings.append("".join(bits))

    h_mat, _ = subspace_matrix_elements(hamiltonian, bitstrings)

    M = len(bitstrings)
    k = min(n_states, max(1, M - 2))

    if issparse(h_mat):
        h_csr = h_mat.tocsr()
    else:
        h_mat = 0.5 * (h_mat + h_mat.conj().T)
        h_csr = csr_matrix(h_mat)

    evals, evecs = eigsh(h_csr.astype(complex), k=k, which="SA", tol=1e-10, maxiter=10_000)
    order = np.argsort(evals)
    return bitstrings, evals[order], evecs[:, order]


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