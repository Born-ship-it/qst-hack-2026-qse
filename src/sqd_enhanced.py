"""
Sample-Based Quantum Diagonalization (SQD).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector

from .circuits import build_trotter_circuit
from .utils import solve_thresholded_gevp


# =====================================================================
# Result container
# =====================================================================

@dataclass
class SQDResult:
    energies:        list[float] = field(default_factory=list)
    coefficients:    list[np.ndarray] = field(default_factory=list)
    subspace_dims:   list[int] = field(default_factory=list)
    sampled_bitstrings: list[list[str]] = field(default_factory=list)
    counts:          list[Counter] = field(default_factory=list)
    runtime_seconds: float = 0.0


# =====================================================================
# SQD solver
# =====================================================================

class SQDSolver:
    """Sample-Based Quantum Diagonalization solver."""

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        reference_circuit: QuantumCircuit,
        reference_bitstring: str,
        krylov_dim: int = 7,
        dt: Optional[float] = None,
        num_trotter_steps: int = 2,
        trotter_order: int = 2,
        num_samples: int = 10_000,
        max_subspace_dim: int = 2000,
        threshold: float = 1e-8,
        seed: Optional[int] = 42,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.num_qubits = hamiltonian.num_qubits
        self.reference_circuit = reference_circuit
        self.reference_bitstring = reference_bitstring
        self.krylov_dim = krylov_dim
        self.num_trotter_steps = num_trotter_steps
        self.trotter_order = trotter_order
        self.num_samples = num_samples
        self.max_subspace_dim = max_subspace_dim
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)

        if dt is None:
            norm_bound = float(sum(abs(c) for c in hamiltonian.coeffs))
            self.dt = np.pi / norm_bound
        else:
            self.dt = float(dt)

        self.reference_energy = self._compute_reference_energy()

    def _compute_reference_energy(self) -> float:
        state = int(self.reference_bitstring, 2)
        energy = 0.0
        for pauli, coeff in zip(self.hamiltonian.paulis, self.hamiltonian.coeffs):
            label = pauli.to_label()
            if all(c in "IZ" for c in label):
                sign = 1.0
                for i, c in enumerate(reversed(label)):
                    if c == "Z":
                        sign *= (-1.0) ** ((state >> i) & 1)
                energy += coeff.real * sign
        return float(energy)

    # ------------------------------------------------------------------
    # Circuit building
    # ------------------------------------------------------------------
    def _build_parameterized_circuit(self) -> QuantumCircuit:
        evolution = build_trotter_circuit(
            self.hamiltonian,
            time=0.0,
            num_trotter_steps=self.num_trotter_steps,
            order=self.trotter_order,
            parameterized=True,
        )
        qc = QuantumCircuit(self.num_qubits)
        qc.compose(self.reference_circuit, inplace=True)
        qc.compose(evolution, inplace=True)
        return qc

    # ------------------------------------------------------------------
    # Subspace matrices
    # ------------------------------------------------------------------
    def _compute_subspace_matrix_elements(self, bitstrings):
        """Delegate to the shared helper in utils."""
        from .utils import subspace_matrix_elements
        return subspace_matrix_elements(self.hamiltonian, bitstrings)

    # ------------------------------------------------------------------
    # Fast solve
    # ------------------------------------------------------------------
    def solve(self, verbose: bool = True) -> SQDResult:
        """
        Fast-path SQD: cached synthesis + sampling from the exact
        statevector probability distribution.
        """
        t_start = time.time()

        circuit = self._build_parameterized_circuit()
        if verbose:
            print("Synthesizing SQD circuit ...", flush=True)
        t_synth = time.time()
        circuit_synth = transpile(
            circuit,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=1,
        )
        if verbose:
            print(f"  done in {time.time() - t_synth:.1f} s "
                  f"(depth {circuit_synth.depth()}, "
                  f"size {circuit_synth.size()})")

        t_param = sorted(circuit_synth.parameters, key=lambda p: p.name)[0]

        all_bitstrings: set[str] = set()
        per_d_counts: list[Counter] = []

        for d in range(1, self.krylov_dim):
            bound = circuit_synth.assign_parameters({t_param: d * self.dt})
            psi = Statevector(bound)
            probs = psi.probabilities_dict()

            keys = list(probs.keys())
            vals = np.array([probs[k] for k in keys])
            vals = vals / vals.sum()

            sampled = self.rng.choice(keys, size=self.num_samples, p=vals)
            counts = Counter(sampled)
            per_d_counts.append(counts)
            all_bitstrings.update(counts.keys())

            if verbose:
                print(f"  d={d}  unique bitstrings: {len(counts)}")

        # Cap the subspace if needed
        all_bitstrings.discard("")
        if len(all_bitstrings) > self.max_subspace_dim:
            total = Counter()
            for c in per_d_counts:
                total.update(c)
            keep = [bs for bs, _ in total.most_common(self.max_subspace_dim)]
            all_bitstrings = set(keep)

        bitstrings = sorted(all_bitstrings)
        h_mat, s_mat = self._compute_subspace_matrix_elements(bitstrings)

        result = SQDResult()
        result.sampled_bitstrings.append(bitstrings)
        result.counts.append(per_d_counts)

        try:
            e_elec, coeffs, retained = solve_thresholded_gevp(
                h_mat, s_mat, threshold=self.threshold,
            )
            result.energies.append(e_elec)
            result.coefficients.append(coeffs)
            result.subspace_dims.append(retained)
        except ValueError:
            result.energies.append(np.nan)
            result.coefficients.append(np.array([]))
            result.subspace_dims.append(0)

        result.runtime_seconds = time.time() - t_start
        return result


# =====================================================================
# Convenience wrapper
# =====================================================================

def run_sqd(
    hamiltonian: SparsePauliOp,
    reference_circuit: QuantumCircuit,
    reference_bitstring: str,
    krylov_dim: int = 7,
    dt: Optional[float] = None,
    num_trotter_steps: int = 2,
    trotter_order: int = 2,
    num_samples: int = 10_000,
) -> SQDResult:
    """One-line SQD run."""
    solver = SQDSolver(
        hamiltonian=hamiltonian,
        reference_circuit=reference_circuit,
        reference_bitstring=reference_bitstring,
        krylov_dim=krylov_dim,
        dt=dt,
        num_trotter_steps=num_trotter_steps,
        trotter_order=trotter_order,
        num_samples=num_samples,
    )
    return solver.solve()
