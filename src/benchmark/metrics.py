"""Accuracy, overlap, and spectral-gap metrics for a LoadedRun."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..utils import compute_exact_ground_state_subspace
from .loader import LoadedRun


@dataclass
class AccuracyMetrics:
    """Per-run accuracy metrics (one row per JSON)."""

    hf_total: float
    mp2_total: float | None
    casci_total: float
    fci_total: float | None
    correlation_energy: float           # HF - CASCI (mHa would be ×1000)
    initial_overlap: float              # |<HF|GS_active>|^2
    spectral_gap_ratio: float           # (E1 - E0) / (Emax - E0)
    active_gs_energy: float             # exact 2e GS energy in the active space

    def to_row(self) -> dict:
        return asdict(self)


def measure_accuracy(loaded: LoadedRun) -> AccuracyMetrics:
    """
    Compute accuracy metrics, including the exact 2e ground state.

    The 2-electron subspace is diagonalized once to extract:
      - the exact active-space GS energy
      - the HF coefficient on the GS (giving the initial overlap)
      - the spectral gap ratio used in the Epperly bound
    """
    h = loaded.hamiltonian
    num_qubits = loaded.num_qubits
    n_elec = loaded.num_electrons

    # Enumerate 2-electron bitstrings and build the projected Hamiltonian
    from itertools import combinations

    bitstrings = []
    for occ in combinations(range(num_qubits), n_elec):
        bits = ["0"] * num_qubits
        for q in occ:
            bits[num_qubits - 1 - q] = "1"
        bitstrings.append("".join(bits))

    # We call the existing helper for the ground energy (it re-builds H and S)
    e_active_gs_elec = compute_exact_ground_state_subspace(h, bitstrings)
    e_active_gs_total = e_active_gs_elec + loaded.nuclear_repulsion

    # We also need the HF coefficient on the GS and the full spectrum.
    # Re-build the projected H once so we can use eigh directly.
    h_mat, _ = _projected_h(h, bitstrings)
    evals, evecs = np.linalg.eigh(0.5 * (h_mat + h_mat.conj().T))

    ref_idx = bitstrings.index(loaded.ref_bitstring)
    gs_vec = evecs[:, 0]
    initial_overlap = float(abs(gs_vec[ref_idx]) ** 2)

    spectral_gap_ratio = _gap_ratio(evals)

    correlation_energy = loaded.hf_total - loaded.casci_total

    return AccuracyMetrics(
        hf_total=loaded.hf_total,
        mp2_total=loaded.mp2_total,
        casci_total=loaded.casci_total,
        fci_total=loaded.fci_total,
        correlation_energy=correlation_energy,
        initial_overlap=initial_overlap,
        spectral_gap_ratio=spectral_gap_ratio,
        active_gs_energy=e_active_gs_total,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _projected_h(hamiltonian, bitstrings) -> tuple[np.ndarray, int]:
    """Return the dense matrix <b_i|H|b_j> over the given basis states."""
    from qiskit.quantum_info import SparsePauliOp  # noqa: F401  (type only)
    from .sweeps import _subspace_matrix_elements  # local import to avoid cycles

    return _subspace_matrix_elements(hamiltonian, bitstrings), 0


def _gap_ratio(evals: np.ndarray) -> float:
    """Spectral gap ratio (E1 - E0) / (Emax - E0); NaN if degenerate."""
    if len(evals) < 2:
        return float("nan")
    e0 = float(evals[0])
    e1 = float(evals[1])
    e_max = float(evals[-1])
    denom = e_max - e0
    return (e1 - e0) / denom if denom > 0 else float("nan")
