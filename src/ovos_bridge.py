"""
Bridge between OVOS compressed molecular data and the QSE/SQD solvers.
=========================================================================

The OVOS pipeline produces a `data_final` JSON dictionary containing the
compressed one- and two-electron integrals, the active-space HF energy, the
CASCI reference, and the full FCI reference. This module converts that data
into a qubit Hamiltonian plus the HF reference state.

Public API
----------
load_ovos_data(path)               Load a JSON file.
ovos_to_qubit_problem(ovos)        Build the qubit Hamiltonian + HF reference.
find_hf_bitstring(H, E, n_elec)    Auto-detect the HF bitstring.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp


# =====================================================================
# Loading
# =====================================================================

def load_ovos_data(path: str | Path) -> dict[str, Any]:
    """Load an OVOS ``data_final`` JSON file and return the dict."""
    with open(path, "r") as handle:
        return json.load(handle)


# =====================================================================
# Hamiltonian construction
# =====================================================================

def ovos_to_qubit_problem(
    ovos_data: dict[str, Any],
    mapper_name: str = "jordan_wigner",
    transpose_h2: bool = False,
) -> tuple[SparsePauliOp, int, str, float, float]:
    """
    Convert OVOS data into a qubit Hamiltonian plus the HF reference.

    Parameters
    ----------
    ovos_data : dict
        Dictionary returned by ``load_ovos_data``.
    mapper_name : str
        ``"jordan_wigner"`` (default), ``"parity"``, or ``"bravyi_kitaev"``.
    transpose_h2 : bool
        Set to True if the two-electron integrals use physicist notation
        (⟨pr|qs⟩) rather than chemist notation ((pq|rs)).

    Returns
    -------
    hamiltonian : SparsePauliOp
    num_qubits : int
    ref_bitstring : str
        HF reference in Qiskit's display convention (leftmost = highest).
    ref_energy : float
        Active-space HF energy in the *electronic* frame (no nuclear repulsion).
    vacuum_energy : float
        <0…0|H|0…0> — the constant shift the control-free swap test cancels.
    """
    # --- Validate -------------------------------------------------------
    required = [
        "n_active_occ", "n_active_vir", "n_active_electrons",
        "one_electron_integrals", "two_electron_integrals",
        "active_hf_energy",
    ]
    missing = [k for k in required if k not in ovos_data]
    if missing:
        raise ValueError(f"OVOS data missing required fields: {missing}")

    # --- Integrals ------------------------------------------------------
    h1 = np.asarray(ovos_data["one_electron_integrals"], dtype=float)
    h2 = np.asarray(ovos_data["two_electron_integrals"], dtype=float)
    if transpose_h2:
        h2 = h2.transpose(0, 2, 1, 3)

    n_occ = int(ovos_data["n_active_occ"])
    n_vir = int(ovos_data["n_active_vir"])
    n_orb = n_occ + n_vir

    # --- Build fermionic operator --------------------------------------
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.mappers import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParityMapper,
    )

    # Restricted closed-shell → all spin blocks identical
    energy_op = ElectronicEnergy.from_raw_integrals(
        h1, h2,
        h1, h2, h2,
    )
    fermionic_op = energy_op.second_q_op()

    # --- Map to qubits --------------------------------------------------
    mapper_name = mapper_name.lower().replace("-", "_")
    if mapper_name in ("jordan_wigner", "jw"):
        mapper = JordanWignerMapper()
    elif mapper_name in ("parity",):
        mapper = ParityMapper()
    elif mapper_name in ("bravyi_kitaev", "bk"):
        mapper = BravyiKitaevMapper()
    else:
        raise ValueError(f"Unknown mapper: {mapper_name}")

    qubit_op = mapper.map(fermionic_op)
    if not isinstance(qubit_op, SparsePauliOp):
        qubit_op = SparsePauliOp.from_list(qubit_op.to_list()).simplify()

    num_qubits = qubit_op.num_qubits
    n_electrons = int(ovos_data["n_active_electrons"])
    target_hf = float(ovos_data["active_hf_energy"])

    # --- Auto-detect HF bitstring --------------------------------------
    ref_bitstring, diff = find_hf_bitstring(
        qubit_op, target_hf, n_electrons,
    )
    if diff > 1e-6:
        raise RuntimeError(
            f"No {n_electrons}-electron configuration matches OVOS HF energy "
            f"({target_hf:.8f} Ha). Best was |{ref_bitstring}> with diff "
            f"{diff:.2e} Ha. Try transpose_h2=True."
        )

    ref_energy = _diag_energy(qubit_op, ref_bitstring)
    vacuum_energy = _diag_energy(qubit_op, "0" * num_qubits)

    return qubit_op, num_qubits, ref_bitstring, ref_energy, vacuum_energy


# =====================================================================
# HF auto-detection
# =====================================================================

def find_hf_bitstring(
    hamiltonian: SparsePauliOp,
    target_energy: float,
    n_electrons: int,
) -> tuple[str, float]:
    """
    Find the computational basis state whose <bs|H|bs> best matches
    ``target_energy``.

    Returns
    -------
    bitstring : str
        In Qiskit's display convention (leftmost = highest qubit).
    diff : float
        |<bs|H|bs> - target_energy|.
    """
    num_qubits = hamiltonian.num_qubits

    best_bs, best_diff = None, float("inf")
    for occ in combinations(range(num_qubits), n_electrons):
        bits = ["0"] * num_qubits
        for q in occ:
            bits[num_qubits - 1 - q] = "1"
        bs = "".join(bits)

        diff = abs(_diag_energy(hamiltonian, bs) - target_energy)
        if diff < best_diff:
            best_diff, best_bs = diff, bs

    return best_bs, best_diff


# =====================================================================
# Internal helpers
# =====================================================================

def _diag_energy(op: SparsePauliOp, bitstring: str) -> float:
    """<bitstring|H|bitstring> (diagonal Pauli terms only)."""
    state = int(bitstring, 2)
    energy = 0.0
    for pauli, coeff in zip(op.paulis, op.coeffs):
        label = pauli.to_label()
        if all(c in "IZ" for c in label):
            sign = 1.0
            for i, c in enumerate(reversed(label)):
                if c == "Z":
                    sign *= (-1.0) ** ((state >> i) & 1)
            energy += coeff.real * sign
    return energy
