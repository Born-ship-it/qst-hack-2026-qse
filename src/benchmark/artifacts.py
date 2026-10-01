"""
Save and load raw intermediate data from a QSD sweep.

The Parquet files (runs.parquet, sweeps.parquet) hold summary metrics.
The artifacts in this module hold the *numerical* intermediate data that
you would want for post-hoc analysis:

- exact/           : exact 2-electron eigendecomposition per run
- sweeps_raw/      : s_row, h_row, coeffs, sampled bitstrings, projected
                     matrices per (run, algorithm)

Both are saved as .npz so they survive pandas-free tooling and are
inspectable from a REPL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# =====================================================================
# Directory layout
# =====================================================================

EXACT_DIR = "exact"
SWEEPS_RAW_DIR = "sweeps_raw"


def run_key_to_filename(run_key: str) -> str:
    """Convert a run_key like 'h2/cc-pvdz/OVOS/initfoh_virtual' to a filename."""
    return run_key.replace("/", "__").replace("::", "__")


def exact_path(base_dir: Path, run_key: str) -> Path:
    return base_dir / EXACT_DIR / f"{run_key_to_filename(run_key)}.npz"


def sweep_path(base_dir: Path, run_key: str, algorithm: str) -> Path:
    fname = f"{run_key_to_filename(run_key)}__{algorithm}.npz"
    return base_dir / SWEEPS_RAW_DIR / fname


# =====================================================================
# Exact eigendecomposition (per run)
# =====================================================================

@dataclass
class ExactArtifact:
    """Exact 2-electron spectrum + reference overlap for a single run."""
    bitstrings: list[str]         # 2e basis state ordering
    evals: np.ndarray             # sorted eigenvalues (electronic frame)
    evecs: np.ndarray             # columns = eigenvectors
    ref_index: int                # index of the HF bitstring in `bitstrings`
    hf_gs_overlap: float          # |<HF|GS>|^2


def save_exact(base_dir: Path, run_key: str, art: ExactArtifact) -> Path:
    path = exact_path(base_dir, run_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bitstrings=np.array(art.bitstrings),
        evals=art.evals,
        evecs=art.evecs,
        ref_index=np.array(art.ref_index),
        hf_gs_overlap=np.array(art.hf_gs_overlap),
    )
    return path


def load_exact(base_dir: Path, run_key: str) -> ExactArtifact:
    data = np.load(exact_path(base_dir, run_key), allow_pickle=False)
    return ExactArtifact(
        bitstrings=list(data["bitstrings"]),
        evals=data["evals"],
        evecs=data["evecs"],
        ref_index=int(data["ref_index"]),
        hf_gs_overlap=float(data["hf_gs_overlap"]),
    )


# =====================================================================
# Sweep artifacts (per run + algorithm)
# =====================================================================

@dataclass
class SweepArtifact:
    """Raw intermediate data from a QSE or SQD sweep."""
    algorithm: str
    krylov_dims: np.ndarray            # shape (K,) — which R's were run
    energies_total: np.ndarray         # shape (K,) — total frame
    coeffs: list[np.ndarray]           # shape (K,) list of R-dim eigenvectors
    condition_numbers: np.ndarray      # shape (K,)
    retained_dims: np.ndarray          # shape (K,)

    # QSE-only (empty arrays if SQD)
    s_row: np.ndarray | None = None    # shape (max_R,) complex
    h_row: np.ndarray | None = None    # shape (max_R,) complex

    # SQD-only (empty lists if QSE)
    subspace_bitstrings: list[list[str]] | None = None
    subspace_H: list[np.ndarray] | None = None
    subspace_S: list[np.ndarray] | None = None
    sampled_counts: list[dict[str, int]] | None = None


def save_sweep(base_dir: Path, run_key: str, art: SweepArtifact) -> Path:
    path = sweep_path(base_dir, run_key, art.algorithm)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "algorithm": np.array(art.algorithm),
        "krylov_dims": art.krylov_dims,
        "energies_total": art.energies_total,
        "condition_numbers": art.condition_numbers,
        "retained_dims": art.retained_dims,
        "coeffs": np.array(art.coeffs, dtype=object),
    }
    if art.s_row is not None:
        payload["s_row"] = art.s_row
    if art.h_row is not None:
        payload["h_row"] = art.h_row
    if art.subspace_bitstrings is not None:
        payload["subspace_bitstrings"] = np.array(art.subspace_bitstrings, dtype=object)
    if art.subspace_H is not None:
        payload["subspace_H"] = np.array(art.subspace_H, dtype=object)
    if art.subspace_S is not None:
        payload["subspace_S"] = np.array(art.subspace_S, dtype=object)
    if art.sampled_counts is not None:
        payload["sampled_counts"] = np.array(art.sampled_counts, dtype=object)

    np.savez_compressed(path, **payload, allow_pickle=True)
    return path


def load_sweep(base_dir: Path, run_key: str, algorithm: str) -> SweepArtifact:
    data = np.load(sweep_path(base_dir, run_key, algorithm), allow_pickle=True)

    def _maybe(key):
        return data[key] if key in data.files else None

    return SweepArtifact(
        algorithm=str(data["algorithm"]),
        krylov_dims=data["krylov_dims"],
        energies_total=data["energies_total"],
        coeffs=list(data["coeffs"]),
        condition_numbers=data["condition_numbers"],
        retained_dims=data["retained_dims"],
        s_row=_maybe("s_row"),
        h_row=_maybe("h_row"),
        subspace_bitstrings=_maybe("subspace_bitstrings"),
        subspace_H=_maybe("subspace_H"),
        subspace_S=_maybe("subspace_S"),
        sampled_counts=_maybe("sampled_counts"),
    )
