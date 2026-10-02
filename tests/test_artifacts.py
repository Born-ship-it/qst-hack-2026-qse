"""Round-trip tests for src.benchmark.artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.benchmark.artifacts import (
    ExactArtifact, SweepArtifact,
    load_exact, load_sweep,
    save_exact, save_sweep,
    pack_coeffs, pack_subspaces, pack_matrices, pack_counts,
)


def test_exact_roundtrip(tmp_path: Path):
    art = ExactArtifact(
        bitstrings=["0011", "0101", "1001"],
        evals=np.array([-1.0, 0.5, 2.0]),
        evecs=np.eye(3),
        ref_index=0,
        hf_gs_projection=0.9,
        gs_degeneracy=1,
    )
    save_exact(tmp_path, "h2/sto-3g/OVOS/default", art)
    loaded = load_exact(tmp_path, "h2/sto-3g/OVOS/default")
    assert loaded.bitstrings == art.bitstrings
    assert np.allclose(loaded.evals, art.evals)
    assert loaded.ref_index == 0
    assert abs(loaded.hf_gs_projection - 0.9) < 1e-12


def test_sweep_padding_and_roundtrip(tmp_path: Path):
    coeffs_list = [np.array([1 + 0j, 0.5 + 0j]),
                   np.array([1 + 0j, 0.3 + 0j, 0.1 + 0j])]
    padded, lens = pack_coeffs(coeffs_list)
    assert padded.shape == (2, 3)
    assert lens.tolist() == [2, 3]

    per_step_bs = [["0011", "0101"], ["0011", "0101", "1001"]]
    union, idx, sizes = pack_subspaces(per_step_bs)
    assert len(union) == 3
    assert sizes.tolist() == [2, 3]

    H_list = [np.eye(2, dtype=complex), np.eye(3, dtype=complex)]
    H_packed = pack_matrices(H_list)
    assert H_packed.shape == (2, 3, 3)

    counts = [{"0011": 10}, {"0011": 5, "0101": 3}]
    cmat = pack_counts(counts, union)
    assert cmat.shape == (2, 3)
    assert cmat.sum() == 18

    art = SweepArtifact(
        algorithm="SQD",
        krylov_dims=np.array([2, 3]),
        energies_total=np.array([-1.1, -1.15]),
        condition_numbers=np.array([10.0, 100.0]),
        retained_dims=np.array([2, 3]),
        coeffs=padded, coeff_lens=lens,
        union_bitstrings=union, subspace_index=idx,
        subspace_sizes=sizes, subspace_H=H_packed,
        subspace_S=H_packed.copy(), sampled_counts=cmat,
    )
    save_sweep(tmp_path, "h2/sto-3g/OVOS/default", art)
    loaded = load_sweep(tmp_path, "h2/sto-3g/OVOS/default", "SQD")

    assert loaded.algorithm == "SQD"
    assert np.allclose(loaded.coeffs, padded)
    assert loaded.subspace_sizes.tolist() == [2, 3]
    assert np.allclose(loaded.sampled_counts, cmat)