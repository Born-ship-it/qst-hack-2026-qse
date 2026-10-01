"""
Quantum Subspace Expansion (QSE) / Krylov Quantum Diagonalization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.linalg as la
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator

from .circuits import build_control_free_swap_test, build_trotter_circuit
from .utils import solve_thresholded_gevp


# =====================================================================
# Result container
# =====================================================================

@dataclass
class QSEResult:
    """Results from a QSE calculation."""
    energies:        list[float] = field(default_factory=list)
    coefficients:    list[np.ndarray] = field(default_factory=list)
    krylov_dims:     list[int] = field(default_factory=list)
    retained_dims:   list[int] = field(default_factory=list)
    s_row:           Optional[np.ndarray] = None
    h_row:           Optional[np.ndarray] = None
    runtime_seconds: float = 0.0


# =====================================================================
# QSE solver
# =====================================================================

class QSESolver:
    """
    Quantum Subspace Expansion solver.

    By default ``solve()`` uses a cached-synthesis path that transpiles the
    parameterized swap-test circuit once, then rebinds parameters for each
    ``d`` value and extracts the statevector. This is substantially faster
    than going through ``StatevectorEstimator`` on every PUB.
    """

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        reference_circuit: QuantumCircuit,
        reference_bitstring: str,
        krylov_dim: int = 10,
        dt: Optional[float] = None,
        num_trotter_steps: int = 3,
        trotter_order: int = 2,
        threshold: float = 1e-10,
        use_shifting: bool = False,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.num_qubits = hamiltonian.num_qubits
        self.reference_circuit = reference_circuit
        self.reference_bitstring = reference_bitstring
        self.krylov_dim = krylov_dim
        self.num_trotter_steps = num_trotter_steps
        self.trotter_order = trotter_order
        self.threshold = threshold
        self.use_shifting = use_shifting

        if dt is None:
            norm_bound = float(sum(abs(c) for c in hamiltonian.coeffs))
            self.dt = np.pi / norm_bound
        else:
            self.dt = float(dt)

        self.reference_energy = self._compute_reference_energy()

        # Shifting is only reliable for Heisenberg-like Hamiltonians where
        # the reference is a computational-basis state and the shift terms
        # can be identified cleanly. Off by default for molecules.
        if self.use_shifting:
            self.shift_tau = self._setup_shifting()
        else:
            self.shift_tau = 0.0

    # ------------------------------------------------------------------
    # Reference state
    # ------------------------------------------------------------------
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

    def _setup_shifting(self) -> float:
        """Return the shift eigenvalue τ such that <ψ_ref|T|ψ_ref> = τ."""
        state = int(self.reference_bitstring, 2)
        tau = 0.0
        for pauli, coeff in zip(self.hamiltonian.paulis, self.hamiltonian.coeffs):
            label = pauli.to_label()
            if all(c in "IZ" for c in label):
                sign = 1.0
                for i, c in enumerate(reversed(label)):
                    if c == "Z":
                        sign *= (-1.0) ** ((state >> i) & 1)
                tau += coeff.real * sign
        return tau

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit:
        evolution_circuit = build_trotter_circuit(
            self.hamiltonian,
            time=0.0,
            num_trotter_steps=self.num_trotter_steps,
            order=self.trotter_order,
            parameterized=True,
        )
        vacuum_energy = self._compute_vacuum_energy()
        return build_control_free_swap_test(
            self.num_qubits,
            self.reference_bitstring,
            evolution_circuit,
            vacuum_energy=vacuum_energy,
        )

    def _compute_vacuum_energy(self) -> float:
        energy = 0.0
        for pauli, coeff in zip(self.hamiltonian.paulis, self.hamiltonian.coeffs):
            if all(c in "IZ" for c in pauli.to_label()):
                energy += coeff.real
        return float(energy)

    def _build_observables(self) -> list[SparsePauliOp]:
        n = self.num_qubits
        obs_x_id = SparsePauliOp("I" * n + "X")
        obs_y_id = SparsePauliOp("I" * n + "Y")

        labels = list(self.hamiltonian.paulis.to_labels())
        coeffs = self.hamiltonian.coeffs
        obs_x_h = SparsePauliOp.from_list(
            [(lbl + "X", c) for lbl, c in zip(labels, coeffs)]
        )
        obs_y_h = SparsePauliOp.from_list(
            [(lbl + "Y", c) for lbl, c in zip(labels, coeffs)]
        )
        return [obs_x_id, obs_y_id, obs_x_h, obs_y_h]

    # ------------------------------------------------------------------
    # Fast solve (default)
    # ------------------------------------------------------------------
    def solve(self, verbose: bool = True) -> QSEResult:
        """Run QSE using the cached-synthesis fast path."""
        t_start = time.time()

        circuit = self._build_circuit()
        observables = self._build_observables()
        d_values = list(range(1, self.krylov_dim))

        if verbose:
            print(f"Synthesizing QSE circuit ...", flush=True)
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

        s_row = np.zeros(self.krylov_dim, dtype=complex)
        h_shifted_row = np.zeros(self.krylov_dim, dtype=complex)
        s_row[0] = 1.0
        h_shifted_row[0] = self.reference_energy - self.shift_tau

        for d in d_values:
            bound = circuit_synth.assign_parameters({t_param: d * self.dt})
            psi = Statevector(bound)
            evs = [float(np.real(psi.expectation_value(obs))) for obs in observables]

            s_row[d] = evs[0] + 1j * evs[1]
            h_shifted_row[d] = evs[2] + 1j * evs[3]

            if verbose:
                print(f"  d={d:2d}  "
                      f"S={s_row[d].real:+.4f}{s_row[d].imag:+.4f}j  "
                      f"H={h_shifted_row[d].real:+.4f}{h_shifted_row[d].imag:+.4f}j")

        h_row = h_shifted_row + self.shift_tau * s_row

        # Reconstruct full Toeplitz matrices
        s_mat = la.toeplitz(s_row.conj(), s_row)
        h_mat = la.toeplitz(h_row.conj(), h_row)

        # Solve GEVP for growing dimensions
        result = QSEResult(s_row=s_row, h_row=h_row)
        for r in range(1, self.krylov_dim + 1):
            try:
                e_elec, coeffs, retained = solve_thresholded_gevp(
                    h_mat[:r, :r], s_mat[:r, :r], threshold=self.threshold,
                )
            except ValueError:
                # All overlap eigenvalues below threshold — carry forward
                e_elec = result.energies[-1] if result.energies else np.nan
                coeffs = np.array([])
                retained = 0
            result.energies.append(e_elec)
            result.coefficients.append(coeffs)
            result.krylov_dims.append(r)
            result.retained_dims.append(retained)

        result.runtime_seconds = time.time() - t_start
        return result

    # ------------------------------------------------------------------
    # Legacy primitive-based solve
    # ------------------------------------------------------------------
    def solve_primitive(
        self,
        estimator: Optional[StatevectorEstimator] = None,
    ) -> QSEResult:
        """
        Legacy path that uses ``StatevectorEstimator`` for every PUB.

        Slower than ``solve()`` by roughly 5–15× on large circuits, but
        useful as a reference for validating the fast path.
        """
        if estimator is None:
            estimator = StatevectorEstimator()

        t_start = time.time()
        circuit = self._build_circuit()
        observables = self._build_observables()

        d_values = list(range(1, self.krylov_dim))
        pubs = []
        for d in d_values:
            for obs in observables:
                pubs.append((circuit, obs, [d * self.dt]))

        job = estimator.run(pubs)
        result = job.result()

        s_row = np.zeros(self.krylov_dim, dtype=complex)
        h_shifted_row = np.zeros(self.krylov_dim, dtype=complex)
        s_row[0] = 1.0
        h_shifted_row[0] = self.reference_energy - self.shift_tau

        for idx, pub_result in enumerate(result):
            d_index, obs_index = divmod(idx, len(observables))
            ev = float(np.asarray(pub_result.data.evs).reshape(-1)[0])
            if obs_index == 0:
                s_row[d_index + 1] = ev
            elif obs_index == 1:
                s_row[d_index + 1] += 1j * ev
            elif obs_index == 2:
                h_shifted_row[d_index + 1] = ev
            elif obs_index == 3:
                h_shifted_row[d_index + 1] += 1j * ev

        h_row = h_shifted_row + self.shift_tau * s_row
        s_mat = la.toeplitz(s_row.conj(), s_row)
        h_mat = la.toeplitz(h_row.conj(), h_row)

        qse = QSEResult(s_row=s_row, h_row=h_row)
        for r in range(1, self.krylov_dim + 1):
            e_elec, coeffs, retained = solve_thresholded_gevp(
                h_mat[:r, :r], s_mat[:r, :r], threshold=self.threshold,
            )
            qse.energies.append(e_elec)
            qse.coefficients.append(coeffs)
            qse.krylov_dims.append(r)
            qse.retained_dims.append(retained)

        qse.runtime_seconds = time.time() - t_start
        return qse


# =====================================================================
# Convenience wrapper
# =====================================================================

def run_baseline_qse(
    hamiltonian: SparsePauliOp,
    reference_circuit: QuantumCircuit,
    reference_bitstring: str,
    krylov_dim: int = 10,
    dt: Optional[float] = None,
    num_trotter_steps: int = 3,
    trotter_order: int = 2,
    use_shifting: bool = False,
) -> QSEResult:
    """One-line QSE run."""
    solver = QSESolver(
        hamiltonian=hamiltonian,
        reference_circuit=reference_circuit,
        reference_bitstring=reference_bitstring,
        krylov_dim=krylov_dim,
        dt=dt,
        num_trotter_steps=num_trotter_steps,
        trotter_order=trotter_order,
        use_shifting=use_shifting,
    )
    return solver.solve()
