"""End-to-end smoke test for src.benchmark.runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmark.runner import BenchmarkConfig, run_benchmark


def _small_config(data_dir: Path, out_dir: Path, force: bool = False) -> BenchmarkConfig:
    return BenchmarkConfig(
        data_dir=data_dir,
        output_dir=out_dir,
        krylov_dims=(2, 3),
        num_trotter_steps=1,
        trotter_order=1,
        num_samples=1000,
        transpile_level=1,
        workers=1,
        force=force,
        verbose=False,
    )


def test_runner_produces_parquet_and_artifacts(
    fake_h2_json: Path, tmp_path: Path,
):
    out = tmp_path / "results"
    config = _small_config(fake_h2_json, out)
    runs_df, sweeps_df = run_benchmark(config)

    assert len(runs_df) == 1
    assert len(sweeps_df) == 4             # 2 dims × 2 algorithms
    assert (out / "runs.parquet").exists()
    assert (out / "sweeps.parquet").exists()
    assert (out / "exact").exists()
    assert (out / "sweeps_raw").exists()
    assert (out / "synthesis_cache").exists()

    # No duplicate (run_key, algorithm, krylov_dim)
    dups = sweeps_df.duplicated(
        subset=["run_key", "algorithm", "krylov_dim"],
    )
    assert not dups.any()


def test_runner_resume_is_idempotent(
    fake_h2_json: Path, tmp_path: Path,
):
    out = tmp_path / "results"
    run_benchmark(_small_config(fake_h2_json, out))
    runs_df2, sweeps_df2 = run_benchmark(_small_config(fake_h2_json, out))
    assert len(runs_df2) == 1
    assert len(sweeps_df2) == 4


def test_force_clears_artifacts(fake_h2_json: Path, tmp_path: Path):
    out = tmp_path / "results"
    run_benchmark(_small_config(fake_h2_json, out))
    # Corrupt one artifact to check --force removes it
    stray = out / "exact" / "stray.npz"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"junk")
    assert stray.exists()

    run_benchmark(_small_config(fake_h2_json, out, force=True))
    assert not stray.exists()
    # Cache is preserved
    assert (out / "synthesis_cache").exists()
    assert len(list((out / "synthesis_cache").glob("*.qpy"))) >= 1


def test_runner_parallel(fake_h2_two_methods: Path, tmp_path: Path):
    """Two runs, two workers, exercises the pre-synthesis pass."""
    out = tmp_path / "results"
    config = _small_config(fake_h2_two_methods, out)
    config.workers = 2
    runs_df, sweeps_df = run_benchmark(config)
    assert len(runs_df) == 2
    # Both runs produce the same number of sweep rows
    assert len(sweeps_df) == 8