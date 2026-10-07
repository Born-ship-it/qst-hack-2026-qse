"""
Sample-Based Krylov Quantum Diagonalization (SKQD).

References
----------
Yu et al., "Quantum-Centric Algorithm for Sample-Based Krylov
Diagonalization", arXiv:2501.09702 (2025).

SKQD prepares a sequence of Krylov states |psi_k> = e^{-i k dt H} |psi_0>,
samples M bitstrings from each in the computational basis, unions all
samples into a subspace, and diagonalizes the projected Hamiltonian.

Unlike QSE, no Hadamard or swap test is required — only projective
measurements. Unlike SQD, the subspace is enriched by multiple time
steps of evolution, which captures configurations not reachable from
the reference in a single Trotter step.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector

from .circuits import build_trotter_circuit
from .utils import solve_thresholded_gevp, subspace_matrix_elements


# =====================================================================
# Result container
# =====================================================================

@dataclass
class SKQDResult:
    """Results from a single SKQD calculation."""

    energies: list[float] = field(default_factory=list)
    coefficients: list[np.ndarray] = field(default_factory=list)
    krylov_steps: list[int] = field(default_factory=list)
    subspace_dims: list[int] = field(default_factory=list)
    retained_dims: list[int] = field(default_factory=list)
    condition_numbers: list[float] = field(default_factory=list)
    sampled_bitstrings: list[list[str]] = field(default_factory=list)
    sampled_counts: list[dict[str, int]] = field(default_factory=list)
    runtime_seconds: float = 0.0


# =====================================================================
# Solver
# =====================================================================

class SKQDSolver:
    """
    Sample-based Krylov quantum diagonalization.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
    reference_circuit : QuantumCircuit
        Prepares the reference state.
    reference_bitstring : str
        HF bitstring in Qiskit display convention.
    krylov_dim : int
        Number of Krylov basis states to sample from.  k = 0, ..., d-1.
    dt : float, optional
        Time step.  Defaults to ``pi / sum(|c_i|)`` (Epperly heuristic).
    num_trotter_steps : int
        Trotter repetitions per time step.
    trotter_order : int
        1 (Lie), 2 or 4 (Suzuki).
    num_samples_per_k : int
        Number of shots per Krylov state.
    threshold : float
        GEVP regularization threshold.
    seed : int
        RNG seed for reproducibility.
    max_subspace_dim : int
        Cap on the subspace dimension; if exceeded, keeps the most
        frequently sampled bitstrings.
    """

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        reference_circuit: QuantumCircuit,
        reference_bitstring: str,
        krylov_dim: int = 8,
        dt: Optional[float] = None,
        num_trotter_steps: int = 2,
        trotter_order: int = 2,
        num_samples_per_k: int = 20_000,
        threshold: float = 1e-8,
        seed: int = 42,
        max_subspace_dim: int = 5000,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.num_qubits = hamiltonian.num_qubits
        self.reference_circuit = reference_circuit
        self.reference_bitstring = reference_bitstring
        self.krylov_dim = krylov_dim
        self.num_trotter_steps = num_trotter_steps
        self.trotter_order = trotter_order
        self.num_samples_per_k = num_samples_per_k
        self.threshold = threshold
        self.max_subspace_dim = max_subspace_dim
        self.rng = np.random.default_rng(seed)

        if dt is None:
            norm_bound = float(sum(abs(c) for c in hamiltonian.coeffs))
            self.dt = np.pi / norm_bound if norm_bound > 0 else np.pi
        else:
            self.dt = float(dt)

        self.reference_energy = self._compute_reference_energy()

    # ------------------------------------------------------------------
    # Reference energy
    # ------------------------------------------------------------------

    def _compute_reference_energy(self) -> float:
        state = int(self.reference_bitstring, 2)
        energy = 0.0
        for pauli, coeff in zip(
            self.hamiltonian.paulis, self.hamiltonian.coeffs
        ):
            label = pauli.to_label()
            if all(c in "IZ" for c in label):
                sign = 1.0
                for i, c in enumerate(reversed(label)):
                    if c == "Z":
                        sign *= (-1.0) ** ((state >> i) & 1)
                energy += coeff.real * sign
        return float(energy)

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _build_parameterized_circuit(self) -> QuantumCircuit:
        """
        Reference prep followed by a parameterized Trotter step.

        Note: we use a fixed number of Trotter repetitions and vary the
        parameter ``t`` across Krylov states.  This is a simplification
        of the paper's Eq. (26) which uses k separate Trotter steps for
        |psi_k>; the difference is a Trotter-error correction at higher
        order that does not affect the qualitative comparison.
        """
        evolution = build_trotter_circuit(
            self.hamiltonian,
            time=0.0,
            num_trotter_steps=self.num_trotter_steps,
            order=self.trotter_order,
            parameterized=True,
        )
        params = sorted(evolution.parameters, key=lambda p: p.name)
        if len(params) != 1:
            raise ValueError(f"expected 1 Trotter parameter, got {len(params)}")
        evolution = evolution.assign_parameters({params[0]: t})

        qc = QuantumCircuit(self.num_qubits)
        qc.compose(self.reference_circuit, inplace=True)
        qc.compose(evolution, inplace=True)
        return qc

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(self, verbose: bool = True) -> SKQDResult:
        """Run SKQD up to ``krylov_dim`` Krylov states."""
        t_start = time.time()

        circuit = self._build_parameterized_circuit()
        t_param = sorted(circuit.parameters, key=lambda p: p.name)[0]

        result = SKQDResult()
        union_bitstrings: set[str] = set()
        per_k_counts: dict[int, Counter] = {}
        aggregated_counts: Counter = Counter()

        if verbose:
            print(f"SKQD: sampling from {self.krylov_dim} Krylov states")

        # --- Sample from each Krylov state ---
        for k in range(self.krylov_dim):
            t_k = k * self.dt
            bound = circuit.assign_parameters({t_param: t_k})
            psi = Statevector(bound)
            probs = psi.probabilities_dict()

            keys = list(probs.keys())
            vals = np.array([probs[key] for key in keys])
            vals = vals / vals.sum()

            sampled = self.rng.choice(
                keys, size=self.num_samples_per_k, p=vals,
            )
            counts = Counter(sampled)
            per_k_counts[k] = counts
            union_bitstrings.update(counts.keys())
            aggregated_counts.update(counts)

            if verbose:
                print(
                    f"  k={k:2d}  unique: {len(counts):4d}  "
                    f"union: {len(union_bitstrings):4d}"
                )

        # --- Cap the subspace if needed ---
        if len(union_bitstrings) > self.max_subspace_dim:
            keep = [bs for bs, _ in aggregated_counts.most_common(
                self.max_subspace_dim
            )]
            union_bitstrings = set(keep)

        bitstrings = sorted(union_bitstrings)

        # --- Build the projected H matrix ---
        if verbose:
            print(f"  Total unique bitstrings: {len(bitstrings)}")
        h_mat, s_mat = subspace_matrix_elements(
            self.hamiltonian, bitstrings,
        )

        # Condition number
        s_vals = np.linalg.eigvalsh(0.5 * (s_mat + s_mat.conj().T))
        pos = s_vals[s_vals > 1e-12]
        cond = float(pos[-1] / pos[0]) if len(pos) >= 2 else float("inf")

        # --- Solve GEVP ---
        try:
            e_elec, coeffs, retained = solve_thresholded_gevp(
                h_mat, s_mat, threshold=self.threshold,
            )
        except ValueError:
            e_elec = float("nan")
            coeffs = np.array([])
            retained = 0

        result.energies.append(e_elec)
        result.coefficients.append(coeffs)
        result.krylov_steps.append(self.krylov_dim)
        result.subspace_dims.append(len(bitstrings))
        result.retained_dims.append(retained)
        result.condition_numbers.append(cond)
        result.sampled_bitstrings.append(bitstrings)
        result.sampled_counts.append(dict(aggregated_counts))
        result.runtime_seconds = time.time() - t_start

        return result


# =====================================================================
# Convenience wrapper
# =====================================================================

def run_skqd(
    hamiltonian: SparsePauliOp,
    reference_circuit: QuantumCircuit,
    reference_bitstring: str,
    krylov_dim: int = 8,
    dt: Optional[float] = None,
    num_trotter_steps: int = 2,
    trotter_order: int = 2,
    num_samples_per_k: int = 20_000,
) -> SKQDResult:
    """One-line SKQD run."""
    solver = SKQDSolver(
        hamiltonian=hamiltonian,
        reference_circuit=reference_circuit,
        reference_bitstring=reference_bitstring,
        krylov_dim=krylov_dim,
        dt=dt,
        num_trotter_steps=num_trotter_steps,
        trotter_order=trotter_order,
        num_samples_per_k=num_samples_per_k,
    )
    return solver.solve()