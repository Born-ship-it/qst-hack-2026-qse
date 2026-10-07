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
# Internal helpers
# =====================================================================

def _diag_energy(pauli_op, bitstring: str) -> float:
    """<bitstring|H|bitstring> for a SparsePauliOp, vectorized.

    Only Z-only Pauli terms contribute; everything else is off-diagonal
    in the computational basis and gives 0. The surviving phase is
    (-1)^popcount(z_mask & b).
    """
    n = pauli_op.num_qubits
    # Qiskit's to_label() reads qubit 0 as the leftmost character.
    # Pauli.x / Pauli.z index qubit q along axis 1.
    bits = np.frombuffer(bitstring.encode(), dtype=np.uint8) - ord('0')
    bits = bits.astype(bool)                # shape (n,), qubit 0 first

    x = pauli_op.paulis.x                   # (N, n) bool
    z = pauli_op.paulis.z                   # (N, n) bool
    coeffs = pauli_op.coeffs                # (N,) complex

    diagonal = ~x.any(axis=1)               # (N,) bool, True for Z-only terms
    parity = (z & bits).sum(axis=1) & 1     # (N,) int
    signs = 1.0 - 2.0 * parity              # (N,) float

    return float(np.sum(coeffs.real * signs * diagonal))

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
    Convert evaluator JSON into a qubit Hamiltonian plus HF reference.

    Handles two schema variants emitted by the evaluator:

    - Full schema (OVOS, COVO): has ``n_active_occ``, ``n_active_vir``
      and ``active_hf_energy``.  The HF reference is auto-detected by
      matching ``<bs|H|bs>`` against ``active_hf_energy``.
    - Reduced schema (baseline, no_mp2): omits those three keys.  We
      derive ``n_active_occ = n_active_electrons // 2`` (closed shell),
      ``n_active_vir = n_orb - n_occ``, and construct the HF bitstring
      from the blocked spin-orbital ordering used by ``qiskit-nature``.

    Returns
    -------
    hamiltonian : SparsePauliOp
    num_qubits : int
    ref_bitstring : str
        HF reference in Qiskit display convention (leftmost = highest).
    ref_energy : float
        ``<ref|H|ref>`` in the *electronic* frame (no nuclear repulsion).
    vacuum_energy : float
        ``<0...0|H|0...0>``, the constant shift cancelled by the
        control-free swap test.
    """
    # --- Minimal required fields ---
    required = [
        "n_active_electrons",
        "n_active_orbitals",
        "one_electron_integrals",
        "two_electron_integrals",
    ]
    missing = [k for k in required if k not in ovos_data]
    if missing:
        raise ValueError(
            f"JSON data missing required fields: {missing}. "
            f"Available keys: {sorted(ovos_data.keys())}"
        )

    # --- Integrals ---
    h1 = np.asarray(ovos_data["one_electron_integrals"], dtype=float)
    h2 = np.asarray(ovos_data["two_electron_integrals"], dtype=float)
    if transpose_h2:
        h2 = h2.transpose(0, 2, 1, 3)

    n_orb = int(ovos_data["n_active_orbitals"])
    n_electrons = int(ovos_data["n_active_electrons"])

    # --- Build fermionic operator ---
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.mappers import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParityMapper,
    )

    energy_op = ElectronicEnergy.from_raw_integrals(h1, h2, h1, h2, h2)
    fermionic_op = energy_op.second_q_op()

    # --- Map to qubits ---
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

    # --- HF reference bitstring ---
    target_hf = ovos_data.get("active_hf_energy")
    if target_hf is not None:
        ref_bitstring, diff = find_hf_bitstring(
            qubit_op, float(target_hf), n_electrons,
        )
        if diff > 1e-6:
            raise RuntimeError(
                f"No {n_electrons}-electron configuration matches "
                f"active_hf_energy ({target_hf:.8f} Ha). Best was "
                f"|{ref_bitstring}> with diff {diff:.2e} Ha. "
                f"Try transpose_h2=True."
            )
    else:
        ref_bitstring = _structural_hf_bitstring(
            num_qubits, n_electrons, n_orb,
        )

    ref_energy = _diag_energy(qubit_op, ref_bitstring)
    vacuum_energy = _diag_energy(qubit_op, "0" * num_qubits)

    return qubit_op, num_qubits, ref_bitstring, ref_energy, vacuum_energy


# =====================================================================
# Structural fallback for reduced-schema JSONs
# =====================================================================

def _structural_hf_bitstring(
    num_qubits: int,
    n_electrons: int,
    n_active_orbitals: int,
) -> str:
    """
    Blocked-order HF reference for a closed-shell RHF calculation.

    ``qiskit-nature``'s ``ElectronicEnergy`` uses blocked spin-orbital
    ordering: spin orbitals ``0 .. n_orb - 1`` are alpha, ``n_orb ..``
    are beta.  The HF determinant occupies the lowest ``n_alpha`` alpha
    orbitals and the lowest ``n_beta`` beta orbitals.

    Qiskit bitstring convention: leftmost character = highest qubit.
    """
    n_occ_alpha = n_electrons // 2
    n_occ_beta = n_electrons - n_occ_alpha

    bits = ["0"] * num_qubits
    for i in range(n_occ_alpha):
        bits[num_qubits - 1 - i] = "1"
    for i in range(n_occ_beta):
        bits[num_qubits - 1 - (n_active_orbitals + i)] = "1"
    return "".join(bits)

# =====================================================================
# HF auto-detection
# =====================================================================
def find_hf_bitstring(
    qubit_op: SparsePauliOp,
    target_hf_energy: float,
    n_electrons: int,
) -> tuple[str, float]:
    """Brute-force the N-electron computational basis state whose diagonal
    energy is closest to ``target_hf_energy``.

    With the vectorized ``_diag_energy`` this is ~C(n, k) * 10 µs, i.e.
    tens of milliseconds even for C(14, 8) = 3005. The old form was slow
    only because ``Pauli.to_label()`` was called per term per candidate.
    """
    num_qubits = qubit_op.num_qubits

    best_bs = None
    best_diff = float("inf")
    for occ in combinations(range(num_qubits), n_electrons):
        bits = ["0"] * num_qubits
        for q in occ:
            bits[q] = "1"           # position q in the string == qubit q
        bs = "".join(bits)
        d = abs(_diag_energy(qubit_op, bs) - target_hf_energy)
        if d < best_diff:
            best_diff = d
            best_bs = bs
            if d < 1e-12:
                break
    return best_bs, best_diff