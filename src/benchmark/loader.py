"""
Load a single OVOS/COVO/... JSON file into a normalised LoadedRun.

Also performs the sanity check requested by the study design:

    |E_CASCI  -  (E_HF_active  +  E_MP2_corr_active  +  E_nuc)|  <  0.1 Ha

The threshold is loose by design: it catches unit-mismatch bugs (electronic
vs total frame, missing nuclear repulsion) without rejecting legitimate
cases where the active space is too small for MP2 to capture all correlation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from ..ovos_bridge import load_ovos_data, ovos_to_qubit_problem
from .discovery import OrbitalRun

logger = logging.getLogger(__name__)

# Sanity check parameters
_SANITY_THRESHOLD_HA = 0.1


@dataclass
class LoadedRun:
    """A fully-loaded run ready for measurement and sweeping."""

    meta: OrbitalRun
    ovos: dict
    hamiltonian: SparsePauliOp
    num_qubits: int
    num_electrons: int
    reference_circuit: QuantumCircuit
    ref_bitstring: str

    # Energies, all in the *total* frame (nuclear repulsion included)
    hf_total: float
    mp2_total: float | None
    casci_total: float
    fci_total: float | None
    sanity_diff_ha: float          # |CASCI - (HF + MP2_corr + E_nuc)|

    # Integrals and metadata for later use
    dt: float  # auto time step pi / sum(|coeffs|)
    nuclear_repulsion: float


def load_run(run: OrbitalRun, strict: bool = False) -> LoadedRun:
    """
    Load and normalise a single OVOS/COVO/... JSON file.

    Parameters
    ----------
    run : OrbitalRun
    strict : bool
        If True, raise on sanity check failure; otherwise log a warning.

    Returns
    -------
    LoadedRun

    Raises
    ------
    RuntimeError
        If the JSON cannot be parsed or the Hamiltonian has no qubits.
    ValueError
        If ``strict=True`` and the sanity check fails.
    """
    ovos = load_ovos_data(run.json_path)

    hamiltonian, num_qubits, ref_bitstring, ref_energy, _ = (
        ovos_to_qubit_problem(ovos)
    )

    nuclear_repulsion = float(ovos.get("nuclear_repulsion_energy", 0.0))
    num_electrons = int(ovos.get("n_active_electrons", 2))

    # Build the reference circuit once
    ref_circuit = QuantumCircuit(num_qubits)
    for i, bit in enumerate(reversed(ref_bitstring)):
        if bit == "1":
            ref_circuit.x(i)

    hf_total = ref_energy + nuclear_repulsion
    mp2_total = _mp2_total(ovos, nuclear_repulsion)
    casci_total = float(ovos.get("casci_energy", np.nan))
    fci_total = _optional_float(ovos.get("full_fci_energy"))

    # Sanity check
    sanity_diff = _check_sanity(
        json_path=run.json_path,
        casci=casci_total,
        hf=hf_total,
        mp2=mp2_total,
        strict=strict,
    )

    # Auto time step from the Hamiltonian's coefficient norm bound
    norm_bound = float(sum(abs(c) for c in hamiltonian.coeffs))
    dt = np.pi / norm_bound if norm_bound > 0 else np.pi

    return LoadedRun(
        meta=run,
        ovos=ovos,
        hamiltonian=hamiltonian,
        num_qubits=num_qubits,
        num_electrons=num_electrons,
        reference_circuit=ref_circuit,
        ref_bitstring=ref_bitstring,
        hf_total=hf_total,
        mp2_total=mp2_total,
        casci_total=casci_total,
        fci_total=fci_total,
        sanity_diff_ha=sanity_diff,
        dt=dt,
        nuclear_repulsion=nuclear_repulsion,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _mp2_total(ovos: dict, nuclear_repulsion: float) -> float | None:
    """Return the active-space MP2 total energy if the pieces are present."""
    hf_elec = ovos.get("active_hf_energy")
    mp2_corr = ovos.get("active_mp2_correlation_energy")
    if hf_elec is None or mp2_corr is None:
        return None
    return float(hf_elec) + float(mp2_corr) + nuclear_repulsion


def _optional_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _check_sanity(
    json_path: Path,
    casci: float,
    hf: float,
    mp2: float | None,
    strict: bool,
) -> float:
    """
    Compare CASCI to the HF+MP2+E_nuc estimate and log the difference.

    Always logs the difference at DEBUG so a sweep summary shows outliers.
    Logs at WARNING above the threshold; raises if strict=True.

    Returns the difference (nan if not computable).
    """
    if np.isnan(casci) or mp2 is None:
        return float("nan")

    diff = abs(casci - mp2)
    logger.debug("%s: |CASCI - (HF+MP2+E_nuc)| = %.4f Ha", json_path, diff)

    if diff <= _SANITY_THRESHOLD_HA:
        return diff

    msg = (
        f"{json_path}: CASCI ({casci:.6f}) deviates from active-space "
        f"MP2+HF+E_nuc ({mp2:.6f}) by {diff:.4f} Ha "
        f"(threshold {_SANITY_THRESHOLD_HA})"
    )
    if strict:
        raise ValueError(msg)
    logger.warning(msg)
    return diff


