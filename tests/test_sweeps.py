"""Smoke tests for src.benchmark.sweeps.

These are fast: 4-qubit H2, Krylov dim 3, 1 Trotter step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.benchmark.discovery import OrbitalRun
from src.benchmark.loader import load_run
from src.benchmark.sweeps import run_qse_sweep, run_sqd_sweep


def _loaded(fake_h2_json: Path):
    json_path = next(fake_h2_json.rglob("*.json"))
    run = OrbitalRun(json_path, "h2", "sto-3g", "OVOS", "initfoh_virtual")
    return load_run(run)


def test_qse_sweep_returns_rows_and_artifact(fake_h2_json: Path, tmp_path: Path):
    loaded = _loaded(fake_h2_json)
    rows, art = run_qse_sweep(
        loaded, krylov_dims=(2, 3),
        num_trotter_steps=1, trotter_order=1,
        cache_dir=tmp_path / "cache",
    )
    assert art.algorithm == "QSE"
    assert art.krylov_dims.tolist() == [2, 3]
    assert len(rows) == 2
    assert art.coeffs.shape[0] == 2          # K rows
    assert (art.coeff_lens > 0).all()
    # QSE-specific fields present
    assert art.s_row is not None
    assert art.h_row is not None
    # Errors should be finite
    for row in rows:
        assert np.isfinite(row.err_vs_casci_mha)


def test_sqd_sweep_returns_padded_arrays(fake_h2_json: Path, tmp_path: Path):
    loaded = _loaded(fake_h2_json)
    rows, art = run_sqd_sweep(
        loaded, krylov_dims=(2, 3),
        num_trotter_steps=1, trotter_order=1,
        num_samples=2000, seed=1,
        cache_dir=tmp_path / "cache",
    )
    assert art.algorithm == "SQD"
    # SQD-specific fields present
    assert art.union_bitstrings is not None
    assert art.subspace_index is not None
    assert art.subspace_sizes is not None
    assert art.subspace_H is not None
    assert art.sampled_counts is not None
    # Every subspace_index row has -1 padding at the tail
    for i, sz in enumerate(art.subspace_sizes):
        assert (art.subspace_index[i, :sz] >= 0).all()
        assert (art.subspace_index[i, sz:] == -1).all()


def test_synthesis_cache_reused(fake_h2_json: Path, tmp_path: Path):
    """Second run should hit the cache — check file modification time."""
    loaded = _loaded(fake_h2_json)
    cache_dir = tmp_path / "cache"

    run_qse_sweep(loaded, krylov_dims=(2,), num_trotter_steps=1,
                  trotter_order=1, cache_dir=cache_dir)
    files = list(cache_dir.glob("*.qpy"))
    assert len(files) == 1
    mtime_1 = files[0].stat().st_mtime

    run_qse_sweep(loaded, krylov_dims=(2,), num_trotter_steps=1,
                  trotter_order=1, cache_dir=cache_dir)
    mtime_2 = files[0].stat().st_mtime
    assert mtime_1 == mtime_2   # not rewritten