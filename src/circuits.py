"""
Quantum circuit construction utilities for QSE / SQD.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter


# =====================================================================
# Trotterized time evolution
# =====================================================================

def build_trotter_circuit(
    hamiltonian: SparsePauliOp,
    time: float = 0.0,
    num_trotter_steps: int = 1,
    order: int = 1,
    parameterized: bool = False,
) -> QuantumCircuit:
    """
    Build a Trotterized time-evolution circuit.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
    time : float
        Evolution time (ignored if ``parameterized=True``).
    num_trotter_steps : int
    order : int
        1 → Lie–Trotter, 2 or 4 → Suzuki–Trotter.
    parameterized : bool
        If True, the returned circuit has one free parameter ``t``.
    """
    num_qubits = hamiltonian.num_qubits
    time_value = Parameter("t") if parameterized else time

    if order == 1:
        synthesis = LieTrotter(reps=num_trotter_steps)
    elif order in (2, 4):
        synthesis = SuzukiTrotter(order=order, reps=num_trotter_steps)
    else:
        raise ValueError(f"Unsupported Trotter order: {order}")

    evol_gate = PauliEvolutionGate(
        hamiltonian,
        time=time_value,
        synthesis=synthesis,
    )

    qc = QuantumCircuit(num_qubits)
    qc.append(evol_gate, range(num_qubits))
    return qc


# =====================================================================
# Swap-test circuits
# =====================================================================

def build_extended_swap_test(
    num_qubits: int,
    reference_circuit: QuantumCircuit,
    evolution_circuit: QuantumCircuit,
) -> QuantumCircuit:
    """
    Extended swap test with a *controlled* time evolution.

    Prepares (|0>|ψ_0> + |1>U(t)|ψ_0>) / √2.
    Uses ancilla + `num_qubits` system qubits.
    """
    ancilla = 0
    system_qubits = list(range(1, num_qubits + 1))

    qc = QuantumCircuit(num_qubits + 1)
    qc.compose(reference_circuit, system_qubits, inplace=True)
    qc.h(ancilla)
    qc.append(evolution_circuit.to_gate().control(1),
              [ancilla] + system_qubits)
    return qc


def build_control_free_swap_test(
    num_qubits: int,
    reference_bitstring: str,
    evolution_circuit: QuantumCircuit,
    vacuum_energy: float,
) -> QuantumCircuit:
    """
    Control-free extended swap test exploiting U(1) symmetry.

    Because the Hamiltonian conserves particle number, the vacuum |0…0> picks
    up only a phase under the evolution operator. We can therefore apply the
    evolution *uncontrolled* and cancel the vacuum phase on the ancilla.

    The ``evolution_circuit`` **must** be parameterized — its single parameter
    is reused to build the phase-correction gate.
    """
    ancilla = 0
    system_qubits = list(range(1, num_qubits + 1))

    qc = QuantumCircuit(num_qubits + 1)
    qc.h(ancilla)

    # Prepare reference state only on the |1> branch (CNOTs suffice for
    # computational-basis references)
    for i, bit in enumerate(reversed(reference_bitstring)):
        if bit == "1":
            qc.cx(ancilla, system_qubits[i])

    qc.barrier()
    qc.compose(evolution_circuit, system_qubits, inplace=True)
    qc.barrier()

    # Uncompute: map |0>|0…0> to |0>|ψ_ref>
    qc.x(ancilla)
    for i, bit in enumerate(reversed(reference_bitstring)):
        if bit == "1":
            qc.cx(ancilla, system_qubits[i])
    qc.x(ancilla)

    # Cancel the known vacuum phase, reusing the evolution parameter
    if not evolution_circuit.parameters:
        raise ValueError(
            "evolution_circuit must be parameterized (it has no parameters)."
        )
    t = sorted(evolution_circuit.parameters, key=lambda p: p.name)[0]
    qc.p(-vacuum_energy * t, ancilla)

    return qc


# =====================================================================
# Hadamard test (single-ancilla variant)
# =====================================================================

def build_hadamard_test_circuit(
    num_qubits: int,
    state_prep_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    measure_real: bool = True,
) -> QuantumCircuit:
    """
    Hadamard test circuit for measuring expectation values.

    Note: for production use, prefer the extended swap test, which gives all
    four real/imaginary components from a single circuit family.
    """
    ancilla = 0
    system_qubits = list(range(1, num_qubits + 1))

    qc = QuantumCircuit(num_qubits + 1)
    qc.h(ancilla)

    controlled_prep = state_prep_circuit.to_gate().control(1)
    qc.append(controlled_prep, [ancilla] + system_qubits)

    if measure_real:
        qc.h(ancilla)
    else:
        qc.sdg(ancilla)
        qc.h(ancilla)

    return qc
