"""
Run QSE and SQD sweeps for a single LoadedRun.

Both sweeps return ``(rows, artifact)``, where ``artifact`` is a padded
:class:`SweepArtifact` ready to be written to disk.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

from ..circuits import build_trotter_circuit
from ..utils import solve_thresholded_gevp, subspace_matrix_elements
from .artifacts import (
    ExactArtifact,
    SweepArtifact,
    pack_coeffs,
    pack_counts,
    pack_matrices,
    pack_subspaces,
)
from .loader import LoadedRun


BASIS_GATES = ["rz", "sx", "x", "cx"]
DEFAULT_CACHE_DIR = Path(".cache/synthesis")


# =====================================================================
# Row schemas
# =====================================================================

@dataclass
class QSERow:
    run_key: str
    algorithm: str
    krylov_dim: int
    energy_total: float
    err_vs_casci_mha: float
    err_vs_fci_mha: float
    correlation_captured_pct: float
    condition_number: float
    retained_dim: int
    subspace_dim: int
    runtime_seconds: float

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class SQDRow(QSERow):
    pass


# =====================================================================
# Exact artifact builder
# =====================================================================

def build_exact_artifact(loaded: LoadedRun) -> ExactArtifact:
    from ..utils import diagonalize_two_electron_subspace

    bitstrings, evals, evecs = diagonalize_two_electron_subspace(
        loaded.hamiltonian, num_electrons=loaded.num_electrons,
    )
    ref_idx = bitstrings.index(loaded.ref_bitstring)
    gs_degeneracy = int(np.sum(np.abs(evals - evals[0]) < 1e-6))
    hf_gs_projection = float(
        np.sum(np.abs(evecs[ref_idx, :gs_degeneracy]) ** 2)
    )
    return ExactArtifact(
        bitstrings=bitstrings,
        evals=evals,
        evecs=evecs,
        ref_index=ref_idx,
        hf_gs_projection=hf_gs_projection,
        gs_degeneracy=gs_degeneracy,
    )


# =====================================================================
# QSE sweep
# =====================================================================

def run_qse_sweep(
    loaded: LoadedRun,
    krylov_dims: Iterable[int],
    num_trotter_steps: int = 2,
    trotter_order: int = 2,
    threshold: float = 1e-10,
    transpile_level: int = 1,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = False,
) -> tuple[list[QSERow], SweepArtifact]:
    from ..qse_baseline import QSESolver

    krylov_dims = sorted(krylov_dims)
    max_r = max(krylov_dims)

    solver = QSESolver(
        hamiltonian=loaded.hamiltonian,
        reference_circuit=loaded.reference_circuit,
        reference_bitstring=loaded.ref_bitstring,
        krylov_dim=max_r,
        num_trotter_steps=num_trotter_steps,
        trotter_order=trotter_order,
        use_shifting=False,
        threshold=threshold,
    )
    circuit = solver._build_circuit()
    circuit_synth = _load_or_build_synthesis(
        circuit,
        key=_synth_key(loaded, "qse", num_trotter_steps, trotter_order, transpile_level),
        cache_dir=cache_dir, transpile_level=transpile_level, verbose=verbose,
    )
    observables = solver._build_observables()
    t_param = sorted(circuit_synth.parameters, key=lambda p: p.name)[0]

    s_row = np.zeros(max_r, dtype=complex)
    h_shifted_row = np.zeros(max_r, dtype=complex)
    s_row[0] = 1.0
    h_shifted_row[0] = solver.reference_energy - solver.shift_tau

    rows: list[QSERow] = []
    energies: list[float] = []
    coeffs_list: list[np.ndarray] = []
    conds: list[float] = []
    retained_list: list[int] = []
    requested = set(krylov_dims)

    for d in range(1, max_r):
        t_d = time.time()
        bound = circuit_synth.assign_parameters({t_param: d * solver.dt})
        psi = Statevector(bound)
        evs = [float(np.real(psi.expectation_value(obs))) for obs in observables]
        s_row[d] = evs[0] + 1j * evs[1]
        h_shifted_row[d] = evs[2] + 1j * evs[3]
        d_seconds = time.time() - t_d

        if (d + 1) not in requested:
            continue

        r = d + 1
        h_row_r = h_shifted_row[:r] + solver.shift_tau * s_row[:r]
        s_mat = _toeplitz(s_row[:r])
        h_mat = _toeplitz(h_row_r)

        cond = _condition_number(s_mat)
        try:
            e_elec, coeffs, retained = solve_thresholded_gevp(
                h_mat, s_mat, threshold=threshold,
            )
            e_total = e_elec + loaded.nuclear_repulsion
        except ValueError:
            e_total = float("nan")
            coeffs = np.array([])
            retained = 0

        rows.append(_make_row(
            loaded, "QSE", r, e_total, cond, retained, r, d_seconds,
        ))
        energies.append(e_total)
        coeffs_list.append(coeffs)
        conds.append(cond)
        retained_list.append(retained)

    coeffs_padded, coeff_lens = pack_coeffs(coeffs_list)

    artifact = SweepArtifact(
        algorithm="QSE",
        krylov_dims=np.asarray(krylov_dims),
        energies_total=np.asarray(energies),
        condition_numbers=np.asarray(conds),
        retained_dims=np.asarray(retained_list),
        coeffs=coeffs_padded,
        coeff_lens=coeff_lens,
        s_row=s_row,
        h_row=h_shifted_row + solver.shift_tau * s_row,
    )
    return rows, artifact


# =====================================================================
# SQD sweep
# =====================================================================

def run_sqd_sweep(
    loaded: LoadedRun,
    krylov_dims: Iterable[int],
    num_trotter_steps: int = 2,
    trotter_order: int = 2,
    num_samples: int = 20_000,
    threshold: float = 1e-8,
    seed: int = 42,
    transpile_level: int = 1,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = False,
) -> tuple[list[SQDRow], SweepArtifact]:
    from ..sqd_enhanced import SQDSolver

    krylov_dims = sorted(krylov_dims)
    max_r = max(krylov_dims)

    solver = SQDSolver(
        hamiltonian=loaded.hamiltonian,
        reference_circuit=loaded.reference_circuit,
        reference_bitstring=loaded.ref_bitstring,
        krylov_dim=max_r,
        num_trotter_steps=num_trotter_steps,
        trotter_order=trotter_order,
        num_samples=num_samples,
        threshold=threshold,
        seed=seed,
    )

    evolution = build_trotter_circuit(
        loaded.hamiltonian, time=0.0,
        num_trotter_steps=num_trotter_steps,
        order=trotter_order, parameterized=True,
    )
    t_sqd = Parameter("t_sqd")
    evolution = evolution.assign_parameters(
        {list(evolution.parameters)[0]: t_sqd}
    )
    qc = QuantumCircuit(loaded.num_qubits)
    qc.compose(loaded.reference_circuit, inplace=True)
    qc.compose(evolution, inplace=True)

    qc_synth = _load_or_build_synthesis(
        qc,
        key=_synth_key(loaded, "sqd", num_trotter_steps, trotter_order, transpile_level),
        cache_dir=cache_dir, transpile_level=transpile_level, verbose=verbose,
    )

    rng = np.random.default_rng(seed)
    all_bitstrings: set[str] = set()
    aggregated_counts: Counter = Counter()

    rows: list[SQDRow] = []
    energies: list[float] = []
    coeffs_list: list[np.ndarray] = []
    conds: list[float] = []
    retained_list: list[int] = []
    per_step_bs: list[list[str]] = []
    per_step_H: list[np.ndarray] = []
    per_step_S: list[np.ndarray] = []
    per_step_counts: list[dict[str, int]] = []
    requested = set(krylov_dims)

    for d in range(1, max_r):
        t_d = time.time()
        bound = qc_synth.assign_parameters({t_sqd: d * solver.dt})
        psi = Statevector(bound)
        probs = psi.probabilities_dict()
        keys = list(probs.keys())
        vals = np.array([probs[k] for k in keys])
        vals = vals / vals.sum()
        sampled = rng.choice(keys, size=num_samples, p=vals)
        counts = Counter(sampled)
        all_bitstrings.update(counts.keys())
        aggregated_counts.update(counts)
        d_seconds = time.time() - t_d

        if (d + 1) not in requested:
            continue

        bitstrings = sorted(all_bitstrings)
        h_mat, s_mat = subspace_matrix_elements(loaded.hamiltonian, bitstrings)
        cond = _condition_number(s_mat)
        try:
            e_elec, coeffs, retained = solve_thresholded_gevp(
                h_mat, s_mat, threshold=threshold,
            )
            e_total = e_elec + loaded.nuclear_repulsion
        except ValueError:
            e_total = float("nan")
            coeffs = np.array([])
            retained = 0

        rows.append(_make_row(
            loaded, "SQD", d + 1, e_total, cond, retained,
            subspace_dim=len(bitstrings), runtime=d_seconds,
        ))
        energies.append(e_total)
        coeffs_list.append(coeffs)
        conds.append(cond)
        retained_list.append(retained)
        per_step_bs.append(bitstrings)
        per_step_H.append(h_mat)
        per_step_S.append(s_mat)
        per_step_counts.append(dict(aggregated_counts))

    coeffs_padded, coeff_lens = pack_coeffs(coeffs_list)
    union_bs, subspace_index, subspace_sizes = pack_subspaces(per_step_bs)
    subspace_H = pack_matrices(per_step_H)
    subspace_S = pack_matrices(per_step_S)
    counts_matrix = pack_counts(per_step_counts, union_bs)

    artifact = SweepArtifact(
        algorithm="SQD",
        krylov_dims=np.asarray(krylov_dims),
        energies_total=np.asarray(energies),
        condition_numbers=np.asarray(conds),
        retained_dims=np.asarray(retained_list),
        coeffs=coeffs_padded,
        coeff_lens=coeff_lens,
        union_bitstrings=union_bs,
        subspace_index=subspace_index,
        subspace_sizes=subspace_sizes,
        subspace_H=subspace_H,
        subspace_S=subspace_S,
        sampled_counts=counts_matrix,
    )
    return rows, artifact


# =====================================================================
# Synthesis cache
# =====================================================================

def synthesis_cache_path(
    cache_dir: Path, key: str,
) -> Path:
    return cache_dir / f"{key}.qpy"


def synthesis_key(
    loaded: LoadedRun, tag: str, num_trotter_steps: int,
    trotter_order: int, transpile_level: int,
) -> str:
    raw = (
        f"{loaded.meta.run_key}|{tag}|{num_trotter_steps}|"
        f"{trotter_order}|{transpile_level}|{loaded.num_qubits}"
    )
    digest = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return f"{loaded.meta.method}_{loaded.num_qubits}q_{tag}_{digest}"


_synth_key = synthesis_key  # local alias


def _load_or_build_synthesis(
    circuit: QuantumCircuit, key: str, cache_dir: Path,
    transpile_level: int, verbose: bool,
) -> QuantumCircuit:
    """Load a cached Trotter circuit, synthesizing and caching if needed.

    Callers must serialize access to the cache directory; see runner.py
    for the pre-synthesis pass that guarantees this.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = synthesis_cache_path(cache_dir, key)
    if cache_file.exists():
        if verbose:
            print(f"    [cache hit] {cache_file.name}")
        with open(cache_file, "rb") as f:
            return qpy.load(f)[0]
    if verbose:
        print(f"    [synthesizing] {key}")
    t0 = time.time()
    qc_synth = transpile(circuit, basis_gates=BASIS_GATES,
                         optimization_level=transpile_level)
    if verbose:
        print(f"      → {time.time() - t0:.1f}s, "
              f"depth {qc_synth.depth()}, size {qc_synth.size()}")
    # Atomic write: write to a tmp file then rename
    tmp = cache_file.with_suffix(".qpy.tmp")
    with open(tmp, "wb") as f:
        qpy.dump(qc_synth, f)
    tmp.replace(cache_file)
    return qc_synth


# =====================================================================
# Small helpers
# =====================================================================

def _toeplitz(first_row: np.ndarray) -> np.ndarray:
    import scipy.linalg as la
    return la.toeplitz(first_row.conj(), first_row)


def _condition_number(s_mat: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(0.5 * (s_mat + s_mat.conj().T))
    pos = vals[vals > 1e-12]
    if len(pos) < 2:
        return float("inf")
    return float(pos[-1] / pos[0])


def _make_row(
    loaded: LoadedRun, algorithm: str, krylov_dim: int,
    energy_total: float, cond: float, retained: int,
    subspace_dim: int, runtime: float,
) -> QSERow:
    err_casci = (energy_total - loaded.casci_total) * 1000.0
    err_fci = ((energy_total - loaded.fci_total) * 1000.0
               if loaded.fci_total is not None else float("nan"))
    corr = loaded.hf_total - loaded.casci_total
    captured = (100.0 * (loaded.hf_total - energy_total) / corr
                if abs(corr) > 1e-12 else float("nan"))
    return QSERow(
        run_key=loaded.meta.run_key,
        algorithm=algorithm,
        krylov_dim=krylov_dim,
        energy_total=energy_total,
        err_vs_casci_mha=err_casci,
        err_vs_fci_mha=err_fci,
        correlation_captured_pct=captured,
        condition_number=cond,
        retained_dim=retained,
        subspace_dim=subspace_dim,
        runtime_seconds=runtime,
    )