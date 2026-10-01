"""
Hamiltonian generation and manipulation utilities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


# =====================================================================
# Heisenberg chain (testbed)
# =====================================================================

def create_heisenberg_hamiltonian(
    num_qubits: int,
    coupling: float = 1.0,
    periodic: bool = False,
) -> SparsePauliOp:
    """1D Heisenberg chain: H = Σ (XX + YY + ZZ) over nearest neighbours."""
    terms: list[tuple[str, complex]] = []

    def append_term(q0: int, q1: int, pauli: str) -> None:
        label = ["I"] * num_qubits
        # Qiskit uses little-endian: qubit 0 is rightmost
        label[num_qubits - 1 - q0] = pauli[0]
        label[num_qubits - 1 - q1] = pauli[1]
        terms.append(("".join(label), coupling))

    for pauli in ("XX", "YY", "ZZ"):
        for q in range(num_qubits - 1):
            append_term(q, q + 1, pauli)
        if periodic and num_qubits > 2:
            append_term(num_qubits - 1, 0, pauli)

    return SparsePauliOp.from_list(terms).simplify()


# =====================================================================
# Molecular Hamiltonians (via qiskit-nature / PySCF)
# =====================================================================

def create_molecular_hamiltonian(
    molecule: str = "LiH",
    bond_distance: float = 1.5,
    basis: str = "sto-3g",
    active_electrons: Optional[int] = None,
    active_orbitals: Optional[int] = None,
) -> tuple[SparsePauliOp, int]:
    """
    Build a molecular qubit Hamiltonian from a name, using PySCF + qiskit-nature.

    Requires ``qiskit-nature`` and ``pyscf`` to be installed.
    """
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
    except ImportError as exc:
        raise ImportError(
            "qiskit-nature and pyscf are required for molecular Hamiltonians. "
            "Install with: pip install qiskit-nature pyscf"
        ) from exc

    geometry_by_molecule = {
        "H2":  f"H 0 0 0; H 0 0 {bond_distance}",
        "LiH": f"Li 0 0 0; H 0 0 {bond_distance}",
        "BeH2": (
            f"Be 0 0 0; "
            f"H 0 0 {bond_distance}; "
            f"H 0 0 {-bond_distance}"
        ),
    }
    if molecule not in geometry_by_molecule:
        raise ValueError(f"Unsupported molecule: {molecule}")
    geometry = geometry_by_molecule[molecule]

    driver = PySCFDriver(atom=geometry, basis=basis, charge=0, spin=0)
    problem = driver.run()

    if active_electrons is not None and active_orbitals is not None:
        problem = ActiveSpaceTransformer(
            num_electrons=active_electrons,
            num_spatial_orbitals=active_orbitals,
        ).transform(problem)

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(problem.second_q_ops()[0])
    if not isinstance(qubit_op, SparsePauliOp):
        qubit_op = SparsePauliOp.from_list(qubit_op.to_list()).simplify()

    return qubit_op, qubit_op.num_qubits


# =====================================================================
# Reference states
# =====================================================================

def get_reference_state_circuit(
    num_qubits: int,
    num_electrons: int,
) -> QuantumCircuit:
    """
    Prepare a Hartree–Fock reference by occupying the lowest spin orbitals.

    Uses interleaved spin-orbital ordering (α₀, β₀, α₁, β₁, …). For a
    different convention, pass the bitstring directly to
    ``ovos_bridge.find_hf_bitstring`` and build the circuit manually.
    """
    if num_electrons > num_qubits:
        raise ValueError(
            f"Reference has {num_electrons} electrons but only "
            f"{num_qubits} qubits."
        )
    qc = QuantumCircuit(num_qubits)
    for i in range(num_electrons):
        qc.x(i)
    return qc


def create_single_excitation_reference(
    num_qubits: int,
    excitation_qubit: Optional[int] = None,
) -> tuple[QuantumCircuit, str]:
    """
    Single-excitation reference (for Heisenberg testbeds).

    Returns the circuit and its bitstring in Qiskit's display convention
    (leftmost character = highest qubit).
    """
    if excitation_qubit is None:
        excitation_qubit = num_qubits // 2

    qc = QuantumCircuit(num_qubits)
    qc.x(excitation_qubit)

    bitstring = ["0"] * num_qubits
    bitstring[num_qubits - 1 - excitation_qubit] = "1"

    return qc, "".join(bitstring)
