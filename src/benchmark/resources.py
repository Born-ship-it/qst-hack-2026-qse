"""Measure quantum-circuit and Pauli-term resource requirements."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from .loader import LoadedRun


BASIS_GATES = ["rz", "sx", "x", "cx"]


@dataclass
class ResourceMetrics:
    """Per-run quantum-resource metrics (one row per JSON)."""

    num_qubits: int
    num_pauli_terms: int
    num_commuting_groups: int
    trotter_depth: int
    trotter_2q_gates: int
    trotter_1q_gates: int
    pauli_max_locality: int
    pauli_mean_locality: float
    synthesis_seconds: float

    def to_row(self) -> dict:
        return asdict(self)


def measure_resources(
    loaded: LoadedRun,
    num_trotter_steps: int = 2,
    trotter_order: int = 2,
    transpile_level: int = 1,
) -> ResourceMetrics:
    """
    Compute resource metrics for a single loaded run.

    Uses a *numeric* (non-parameterized) Trotter circuit so that transpilation
    can be applied once and the resulting gate counts are directly meaningful
    for hardware-cost estimation.
    """
    hamiltonian = loaded.hamiltonian
    num_qubits = hamiltonian.num_qubits
    num_terms = len(hamiltonian.paulis)

    # Commuting groups
    groups = hamiltonian.group_commuting(qubit_wise=True)
    num_groups = len(groups)

    # Pauli string locality
    localities = _pauli_localities(hamiltonian)

    # Trotter circuit (numeric, dt scaled for one step)
    synthesis = _synthesis(trotter_order, num_trotter_steps)
    evol = PauliEvolutionGate(hamiltonian, time=loaded.dt, synthesis=synthesis)

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_qubits)
    qc.append(evol, range(num_qubits))

    t0 = time.time()
    qc_synth = transpile(qc, basis_gates=BASIS_GATES,
                         optimization_level=transpile_level)
    synth_seconds = time.time() - t0

    depth = qc_synth.depth()
    two_q = sum(1 for inst in qc_synth.data if inst.operation.num_qubits == 2)
    one_q = sum(1 for inst in qc_synth.data if inst.operation.num_qubits == 1)

    return ResourceMetrics(
        num_qubits=num_qubits,
        num_pauli_terms=num_terms,
        num_commuting_groups=num_groups,
        trotter_depth=depth,
        trotter_2q_gates=two_q,
        trotter_1q_gates=one_q,
        pauli_max_locality=int(np.max(localities)) if localities else 0,
        pauli_mean_locality=float(np.mean(localities)) if localities else 0.0,
        synthesis_seconds=float(synth_seconds),
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _synthesis(order: int, reps: int):
    if order == 1:
        return LieTrotter(reps=reps)
    if order in (2, 4):
        return SuzukiTrotter(order=order, reps=reps)
    raise ValueError(f"Unsupported Trotter order: {order}")


def _pauli_localities(op: SparsePauliOp) -> list[int]:
    """Return the number of non-identity characters in each Pauli string."""
    return [sum(1 for c in label if c != "I")
            for label in op.paulis.to_labels()]
