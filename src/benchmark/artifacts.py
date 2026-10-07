"""
Save and load raw intermediate data from a QSD sweep.

Ragged data (coefficient vectors of varying length, variable-sized
subspaces) is padded to fixed-shape arrays so the .npz files are portable
and can be loaded with `allow_pickle=False`.

Layout::

    <base>/exact/<run_key_safe>.npz
    <base>/sweeps_raw/<run_key_safe>__<ALGO>.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXACT_DIR = "exact"
SWEEPS_RAW_DIR = "sweeps_raw"


def run_key_to_filename(run_key: str) -> str:
    return run_key.replace("/", "__").replace("::", "__")


def exact_path(base_dir: Path, run_key: str) -> Path:
    return base_dir / EXACT_DIR / f"{run_key_to_filename(run_key)}.npz"


def sweep_path(base_dir: Path, run_key: str, algorithm: str) -> Path:
    fname = f"{run_key_to_filename(run_key)}__{algorithm}.npz"
    return base_dir / SWEEPS_RAW_DIR / fname


# =====================================================================
# Exact spectrum (fixed shape, no padding needed)
# =====================================================================

@dataclass
class ExactArtifact:
    bitstrings: list[str]
    evals: np.ndarray
    evecs: np.ndarray
    ref_index: int
    hf_gs_projection: float     # |<HF|P_0|HF>|^2, projector onto GS manifold
    gs_degeneracy: int          # number of evals within 1e-6 of E0


def save_exact(base_dir: Path, run_key: str, art: ExactArtifact) -> Path:
    path = exact_path(base_dir, run_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    # <U0 is invalid numpy dtype; use a minimum of 1
    n_qubits = len(art.bitstrings[0]) if art.bitstrings else 1  # CHANGED
    np.savez_compressed(
        path,
        bitstrings=np.asarray(art.bitstrings, dtype=f"<U{n_qubits}"),
        evals=art.evals,
        evecs=art.evecs,
        ref_index=np.int64(art.ref_index),
        hf_gs_projection=np.float64(art.hf_gs_projection),
        gs_degeneracy=np.int64(art.gs_degeneracy),
    )
    return path


def load_exact(base_dir: Path, run_key: str) -> ExactArtifact:
    data = np.load(exact_path(base_dir, run_key), allow_pickle=False)
    return ExactArtifact(
        bitstrings=[str(x) for x in data["bitstrings"]],
        evals=data["evals"],
        evecs=data["evecs"],
        ref_index=int(data["ref_index"]),
        hf_gs_projection=float(data["hf_gs_projection"]),
        gs_degeneracy=int(data["gs_degeneracy"]),
    )


# =====================================================================
# Sweep artifact — padded shape
# =====================================================================

@dataclass
class SweepArtifact:
    """
    Raw intermediate data from a QSE, SQD, or SKQD sweep.

    Every array has a fixed shape so the .npz is portable.

    Padding conventions
    -------------------
    - ``coeffs``:      shape (K, max_R), zero-padded. ``coeff_lens`` gives
                       the true length of each vector.
    - ``subspace_*``:  SQD / SKQD only. ``subspace_index`` is (K, max_dim)
                       with -1 for padding; ``union_bitstrings`` holds the
                       string pool. ``subspace_H``/``subspace_S`` are
                       (K, max_dim, max_dim) with zero padding outside the
                       valid block.
    - ``sampled_counts``: (K, n_union) integer counts.
    """
    algorithm: str
    krylov_dims: np.ndarray             # (K,)
    energies_total: np.ndarray          # (K,)
    condition_numbers: np.ndarray       # (K,)
    retained_dims: np.ndarray           # (K,)
    coeffs: np.ndarray                  # (K, max_R), complex
    coeff_lens: np.ndarray              # (K,), int

    # QSE-only
    s_row: np.ndarray | None = None     # (max_R,), complex
    h_row: np.ndarray | None = None     # (max_R,), complex

    # SQD / SKQD-only
    union_bitstrings: np.ndarray | None = None    # (n_union,), '<U{n}'
    subspace_index: np.ndarray | None = None      # (K, max_dim), int, -1 padded
    subspace_sizes: np.ndarray | None = None      # (K,), int
    subspace_H: np.ndarray | None = None          # (K, max_dim, max_dim), complex
    subspace_S: np.ndarray | None = None          # (K, max_dim, max_dim), complex
    sampled_counts: np.ndarray | None = None      # (K, n_union), int


def save_sweep(base_dir: Path, run_key: str, art: SweepArtifact) -> Path:
    path = sweep_path(base_dir, run_key, art.algorithm)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "algorithm": np.asarray(art.algorithm),
        "krylov_dims": art.krylov_dims,
        "energies_total": art.energies_total,
        "condition_numbers": art.condition_numbers,
        "retained_dims": art.retained_dims,
        "coeffs": art.coeffs,
        "coeff_lens": art.coeff_lens,
    }
    if art.s_row is not None:
        payload["s_row"] = art.s_row
        payload["h_row"] = art.h_row
    if art.union_bitstrings is not None:
        payload["union_bitstrings"] = art.union_bitstrings
        payload["subspace_index"] = art.subspace_index
        payload["subspace_sizes"] = art.subspace_sizes
        payload["subspace_H"] = art.subspace_H
        payload["subspace_S"] = art.subspace_S
        payload["sampled_counts"] = art.sampled_counts

    np.savez_compressed(path, **payload, allow_pickle=False)
    return path


def load_sweep(base_dir: Path, run_key: str, algorithm: str) -> SweepArtifact:
    data = np.load(sweep_path(base_dir, run_key, algorithm), allow_pickle=False)

    def _maybe(key):
        return data[key] if key in data.files else None

    return SweepArtifact(
        algorithm=str(data["algorithm"]),
        krylov_dims=data["krylov_dims"],
        energies_total=data["energies_total"],
        condition_numbers=data["condition_numbers"],
        retained_dims=data["retained_dims"],
        coeffs=data["coeffs"],
        coeff_lens=data["coeff_lens"],
        s_row=_maybe("s_row"),
        h_row=_maybe("h_row"),
        union_bitstrings=_maybe("union_bitstrings"),
        subspace_index=_maybe("subspace_index"),
        subspace_sizes=_maybe("subspace_sizes"),
        subspace_H=_maybe("subspace_H"),
        subspace_S=_maybe("subspace_S"),
        sampled_counts=_maybe("sampled_counts"),
    )


# =====================================================================
# Padding helpers (used by sweeps.py)
# =====================================================================

def pack_coeffs(                                                     # RENAMED
    coeffs_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a ragged list of coefficient vectors to a fixed-shape array."""
    K = len(coeffs_list)
    if K == 0:
        return np.zeros((0, 0), dtype=complex), np.zeros(0, dtype=int)
    max_R = max((len(c) for c in coeffs_list), default=0)
    padded = np.zeros((K, max_R), dtype=complex)
    lens = np.zeros(K, dtype=int)
    for i, c in enumerate(coeffs_list):
        c = np.asarray(c, dtype=complex)
        padded[i, : len(c)] = c
        lens[i] = len(c)
    return padded, lens


def pack_subspaces(
    per_step_bitstrings: list[list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack a list of variable-size bitstring sets into a fixed-shape index.

    Returns
    -------
    union_bitstrings : np.ndarray, shape (n_union,), fixed-length '<U{n}'
    indices          : np.ndarray, shape (K, max_dim), int, -1 padded
    sizes            : np.ndarray, shape (K,), int
    """
    union = sorted({bs for step in per_step_bitstrings for bs in step})
    K = len(per_step_bitstrings)
    if not union:
        return (np.zeros(0, dtype="<U1"),
                np.zeros((K, 0), dtype=int),
                np.zeros(K, dtype=int))
    bit_len = max(len(bs) for bs in union)                            # CHANGED
    lookup = {bs: i for i, bs in enumerate(union)}
    max_dim = max((len(step) for step in per_step_bitstrings), default=0)
    indices = np.full((K, max_dim), -1, dtype=int)
    sizes = np.zeros(K, dtype=int)
    for i, step in enumerate(per_step_bitstrings):
        indices[i, : len(step)] = [lookup[bs] for bs in step]
        sizes[i] = len(step)
    return np.asarray(union, dtype=f"<U{bit_len}"), indices, sizes


def pack_matrices(
    matrices: list[np.ndarray],
) -> np.ndarray:
    """Zero-pad a ragged list of square matrices into a 3D array."""
    if not matrices:
        return np.zeros((0, 0, 0), dtype=complex)
    max_dim = max(m.shape[0] for m in matrices)
    out = np.zeros((len(matrices), max_dim, max_dim), dtype=complex)
    for i, m in enumerate(matrices):
        d = m.shape[0]
        out[i, :d, :d] = m
    return out


def pack_counts(
    per_step_counts: list[dict[str, int]],
    union_bitstrings: np.ndarray,
) -> np.ndarray:
    """Pack per-step count dicts into a (K, n_union) int64 matrix."""
    K = len(per_step_counts)
    if union_bitstrings.size == 0:
        return np.zeros((K, 0), dtype=np.int64)
    lookup = {str(bs): i for i, bs in enumerate(union_bitstrings)}
    out = np.zeros((K, union_bitstrings.size), dtype=np.int64)
    for i, counts in enumerate(per_step_counts):
        for bs, c in counts.items():
            j = lookup.get(bs)
            if j is not None:
                out[i, j] = int(c)                                    # CHANGED
    return out