"""Accuracy, overlap, and spectral-gap metrics for a LoadedRun."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..utils import diagonalize_two_electron_subspace
from .loader import LoadedRun


# Degeneracy window for grouping the ground manifold
_GS_DEGENERACY_TOL = 1e-6


@dataclass
class AccuracyMetrics:
    hf_total: float
    mp2_total: float | None
    casci_total: float
    fci_total: float | None
    correlation_energy: float
    hf_gs_projection: float         # |<HF|P_0|HF>|^2, projector onto GS manifold
    gs_degeneracy: int
    spectral_gap_ratio: float
    active_gs_energy: float
    sanity_diff_ha: float

    def to_row(self) -> dict:
        return asdict(self)


def measure_accuracy(loaded: LoadedRun) -> AccuracyMetrics:
    """
    Compute accuracy metrics for a single run.

    The ground-state overlap is reported as the spectral projection
    |<HF|P_0|HF>|^2 where P_0 projects onto the (possibly degenerate)
    ground manifold. This is well-defined even when the eigensolver makes
    an arbitrary choice within a degenerate subspace.
    """
    bitstrings, evals, evecs = diagonalize_two_electron_subspace(
        loaded.hamiltonian, num_electrons=loaded.num_electrons,
    )

    e0 = float(evals[0])
    gs_degeneracy = int(np.sum(np.abs(evals - e0) < _GS_DEGENERACY_TOL))

    ref_idx = bitstrings.index(loaded.ref_bitstring)
    hf_row = evecs[ref_idx, :gs_degeneracy]          # shape (d_gs,)
    hf_projection = float(np.sum(np.abs(hf_row) ** 2))

    spectral_gap_ratio = _gap_ratio(evals, gs_degeneracy)
    e_active_gs_total = e0 + loaded.nuclear_repulsion
    correlation_energy = loaded.hf_total - loaded.casci_total

    return AccuracyMetrics(
        hf_total=loaded.hf_total,
        mp2_total=loaded.mp2_total,
        casci_total=loaded.casci_total,
        fci_total=loaded.fci_total,
        correlation_energy=correlation_energy,
        hf_gs_projection=hf_projection,
        gs_degeneracy=gs_degeneracy,
        spectral_gap_ratio=spectral_gap_ratio,
        active_gs_energy=e_active_gs_total,
        sanity_diff_ha=loaded.sanity_diff_ha,
    )


def _gap_ratio(evals: np.ndarray, gs_degeneracy: int) -> float:
    """
    Spectral gap ratio (E1 - E0) / (Emax - E0), where E1 is the first
    eigenvalue *above* the ground manifold. Returns NaN if the spectrum
    is too degenerate or truncated to be meaningful.
    """
    if len(evals) <= gs_degeneracy:
        return float("nan")
    e0 = float(evals[0])
    e1 = float(evals[gs_degeneracy])
    e_max = float(evals[-1])
    denom = e_max - e0
    return (e1 - e0) / denom if denom > 0 else float("nan")