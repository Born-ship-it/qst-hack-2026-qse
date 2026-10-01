"""
Run QSE, SQD, and noise sweeps for a single LoadedRun.

All sweeps share a disk-backed synthesis cache so that expensive Trotter
transpilations are done at most once per (run, Trotter params, transpile level).

Each public function returns both the summary rows (for the Parquet files)
and a rich artifact object (for sweeps_raw/*.npz).
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
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..circuits import build_trotter_circuit
from ..utils import solve_thresholded_gevp
from .artifacts import ExactArtifact, SweepArtifact
from .loader import LoadedRun


BASIS_GATES = ["rz", "sx", "x", "cx"]
DEFAULT_CACHE_DIR = Path(".cache/synthesis")


# =====================================================================
# Row schemas (unchanged)
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
class SQDRow:
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


# =====================================================================
# Exact artifact builder
# =====================================================================

def build_exact_artifact(loaded: LoadedRun) -> ExactArtifact:
    """Diagonalize the active-space 2e Hamiltonian and package it."""
    from itertools import combinations

    n = loaded.num_qubits
    n_elec = loaded.num_electrons
    bitstrings = []
    for occ in combinations(range(n), n_elec):
        bits = ["0"] * n
        for q in occ:
            bits[n - 1 - q] = "1"
        bitstrings.append("".join(bits))

    h_mat, _ = _subspace_matrix_elements(loaded.hamiltonian, bitstrings)
    evals, evecs = np.linalg.eigh(0.5 * (h_mat + h_mat.conj().T))

    ref_idx = bitstrings.index(loaded.ref_bitstring)
    hf_gs_overlap = float(abs(evecs[ref_idx, 0]) ** 2)

    return ExactArtifact(
        bitstrings=bitstrings,
        evals=evals,
        evecs=evecs,
        ref_index=ref_idx,
        hf_gs_overlap=hf_gs_overlap,
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
    """Run a QSE sweep.  Returns (rows, artifact)."""
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
    circuit_synth = _get_cached_synthesis(
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
    energies_total: list[float] = []
    coeffs_list: list[np.ndarray] = []
    conds: list[float] = []
    retained_list: list[int] = []

    for d in range(1, max_r):
        t_d = time.time()
        bound = circuit_synth.assign_parameters({t_param: d * solver.dt})
        psi = Statevector(bound)
        evs = [float(np.real(psi.expectation_value(obs))) for obs in observables]
        s_row[d] = evs[0] + 1j * evs[1]
        h_shifted_row[d] = evs[2] + 1j * evs[3]
        d_seconds = time.time() - t_d

        if (d + 1) not in krylov_dims:
            continue

        r = d + 1
        h_row = h_shifted_row[:r] + solver.shift_tau * s_row[:r]
        s_mat = _toeplitz(s_row[:r])
        h_mat = _toeplitz(h_row)

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

        rows.append(_make_qse_row(
            loaded=loaded, krylov_dim=r, energy_total=e_total,
            cond=cond, retained=retained, runtime=d_seconds,
        ))
        energies_total.append(e_total)
        coeffs_list.append(coeffs)
        conds.append(cond)
        retained_list.append(retained)

    artifact = SweepArtifact(
        algorithm="QSE",
        krylov_dims=np.array(sorted(krylov_dims)),
        energies_total=np.array(energies_total),
        coeffs=coeffs_list,
        condition_numbers=np.array(conds),
        retained_dims=np.array(retained_list),
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
    """Run a cumulative SQD sweep.  Returns (rows, artifact)."""
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

    qc_synth = _get_cached_synthesis(
        qc,
        key=_synth_key(loaded, "sqd", num_trotter_steps, trotter_order, transpile_level),
        cache_dir=cache_dir, transpile_level=transpile_level, verbose=verbose,
    )

    rng = np.random.default_rng(seed)
    all_bitstrings: set[str] = set()
    aggregated_counts: Counter = Counter()

    rows: list[SQDRow] = []
    energies_total: list[float] = []
    coeffs_list: list[np.ndarray] = []
    conds: list[float] = []
    retained_list: list[int] = []
    per_d_bitstrings: list[list[str]] = []
    per_d_H: list[np.ndarray] = []
    per_d_S: list[np.ndarray] = []
    per_d_counts: list[dict[str, int]] = []

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

        if (d + 1) not in krylov_dims:
            continue

        bitstrings = sorted(all_bitstrings)
        h_mat, s_mat = _subspace_matrix_elements(loaded.hamiltonian, bitstrings)
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

        rows.append(_make_sqd_row(
            loaded=loaded, krylov_dim=d + 1, energy_total=e_total,
            cond=cond, retained=retained, subspace_dim=len(bitstrings),
            runtime=d_seconds,
        ))
        energies_total.append(e_total)
        coeffs_list.append(coeffs)
        conds.append(cond)
        retained_list.append(retained)
        per_d_bitstrings.append(bitstrings)
        per_d_H.append(h_mat)
        per_d_S.append(s_mat)
        per_d_counts.append(dict(aggregated_counts))

    artifact = SweepArtifact(
        algorithm="SQD",
        krylov_dims=np.array(sorted(krylov_dims)),
        energies_total=np.array(energies_total),
        coeffs=coeffs_list,
        condition_numbers=np.array(conds),
        retained_dims=np.array(retained_list),
        subspace_bitstrings=per_d_bitstrings,
        subspace_H=per_d_H,
        subspace_S=per_d_S,
        sampled_counts=per_d_counts,
    )
    return rows, artifact


# =====================================================================
# Internal helpers (unchanged from previous version)
# =====================================================================

def _synth_key(loaded, tag, num_trotter_steps, trotter_order, level) -> str:
    raw = (
        f"{loaded.meta.run_key}|{tag}|{num_trotter_steps}|"
        f"{trotter_order}|{level}|{loaded.num_qubits}"
    )
    digest = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return f"{loaded.meta.method}_{loaded.num_qubits}q_{tag}_{digest}"


def _get_cached_synthesis(
    circuit: QuantumCircuit, key: str, cache_dir: Path,
    transpile_level: int, verbose: bool,
) -> QuantumCircuit:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.qpy"
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
    with open(cache_file, "wb") as f:
        qpy.dump(qc_synth, f)
    return qc_synth


def _toeplitz(first_row: np.ndarray) -> np.ndarray:
    import scipy.linalg as la
    return la.toeplitz(first_row.conj(), first_row)


def _condition_number(s_mat: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(0.5 * (s_mat + s_mat.conj().T))
    pos = vals[vals > 1e-12]
    if len(pos) < 2:
        return float("inf")
    return float(pos[-1] / pos[0])


def _subspace_matrix_elements(
    hamiltonian: SparsePauliOp, bitstrings: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    dim = len(bitstrings)
    h_mat = np.zeros((dim, dim), dtype=complex)
    s_mat = np.eye(dim, dtype=complex)
    n = hamiltonian.num_qubits
    for i, bra_bs in enumerate(bitstrings):
        for j, ket_bs in enumerate(bitstrings):
            if i == j:
                h_mat[i, j] = _diag_energy(hamiltonian, bra_bs)
            else:
                h_mat[i, j] = _off_diag_energy(hamiltonian, n, bra_bs, ket_bs)
    return h_mat, s_mat


def _diag_energy(op: SparsePauliOp, bitstring: str) -> complex:
    state = int(bitstring, 2)
    e = 0.0 + 0.0j
    for pauli, coeff in zip(op.paulis, op.coeffs):
        label = pauli.to_label()
        if all(c in "IZ" for c in label):
            sign = 1.0
            for i, c in enumerate(reversed(label)):
                if c == "Z":
                    sign *= (-1.0) ** ((state >> i) & 1)
            e += coeff * sign
    return e


def _off_diag_energy(op, n, bra_bs, ket_bs) -> complex:
    bra, ket = int(bra_bs, 2), int(ket_bs, 2)
    val = 0.0 + 0.0j
    for pauli, coeff in zip(op.paulis, op.coeffs):
        label = pauli.to_label()
        new_state = ket
        phase = 1.0 + 0.0j
        for q in range(n):
            p = label[n - 1 - q]
            if p == "I":
                continue
            elif p == "Z":
                if (new_state >> q) & 1:
                    phase *= -1
            elif p == "X":
                new_state ^= 1 << q
            elif p == "Y":
                phase *= 1j if not ((new_state >> q) & 1) else -1j
                new_state ^= 1 << q
        if new_state == bra:
            val += coeff * phase
    return val


def _make_qse_row(loaded, krylov_dim, energy_total, cond, retained, runtime):
    err_casci = (energy_total - loaded.casci_total) * 1000.0
    err_fci = ((energy_total - loaded.fci_total) * 1000.0
               if loaded.fci_total is not None else float("nan"))
    corr = loaded.hf_total - loaded.casci_total
    captured = (100.0 * (loaded.hf_total - energy_total) / corr
                if abs(corr) > 1e-12 else float("nan"))
    return QSERow(
        run_key=loaded.meta.run_key, algorithm="QSE", krylov_dim=krylov_dim,
        energy_total=energy_total, err_vs_casci_mha=err_casci,
        err_vs_fci_mha=err_fci, correlation_captured_pct=captured,
        condition_number=cond, retained_dim=retained, subspace_dim=krylov_dim,
        runtime_seconds=runtime,
    )


def _make_sqd_row(loaded, krylov_dim, energy_total, cond, retained,
                  subspace_dim, runtime):
    err_casci = (energy_total - loaded.casci_total) * 1000.0
    err_fci = ((energy_total - loaded.fci_total) * 1000.0
               if loaded.fci_total is not None else float("nan"))
    corr = loaded.hf_total - loaded.casci_total
    captured = (100.0 * (loaded.hf_total - energy_total) / corr
                if abs(corr) > 1e-12 else float("nan"))
    return SQDRow(
        run_key=loaded.meta.run_key, algorithm="SQD", krylov_dim=krylov_dim,
        energy_total=energy_total, err_vs_casci_mha=err_casci,
        err_vs_fci_mha=err_fci, correlation_captured_pct=captured,
        condition_number=cond, retained_dim=retained, subspace_dim=subspace_dim,
        runtime_seconds=runtime,
    )
