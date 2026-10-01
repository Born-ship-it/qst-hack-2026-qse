"""
Orchestrator for the orbital-method benchmarking study.

Responsibilities
----------------
- Discover JSON runs under a data directory.
- Skip runs already present in the Parquet checkpoint (unless --force).
- Run each remaining run in a worker process.
- Write two Parquet files: runs.parquet and sweeps.parquet.
- Optionally generate all figures.
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .discovery import OrbitalRun, discover_runs
from .loader import load_run
from .metrics import measure_accuracy
from .resources import measure_resources
from .sweeps import run_qse_sweep, run_sqd_sweep

logger = logging.getLogger(__name__)


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class BenchmarkConfig:
    """All parameters that define a single benchmark run."""

    data_dir: Path
    output_dir: Path
    methods: list[str] | None = None
    krylov_dims: tuple[int, ...] = (2, 4, 6, 8, 10, 12)
    num_trotter_steps: int = 2
    trotter_order: int = 2
    num_samples: int = 20_000
    threshold_qse: float = 1e-10
    threshold_sqd: float = 1e-8
    transpile_level: int = 1
    strict_sanity: bool = False
    workers: int = 1
    force: bool = False
    verbose: bool = False


# =====================================================================
# Public entry point
# =====================================================================

def run_benchmark(config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full study and return (runs_df, sweeps_df).

    The two DataFrames are also written to
    ``config.output_dir/runs.parquet`` and ``sweeps.parquet``.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = config.output_dir / "runs.parquet"
    sweeps_path = config.output_dir / "sweeps.parquet"

    # 1. Discover runs
    all_runs = discover_runs(config.data_dir)
    all_runs = _filter_runs(all_runs, config)
    logger.info(f"Discovered {len(all_runs)} JSON files under {config.data_dir}")
    if not all_runs:
        raise SystemExit(
            f"No JSON files found under {config.data_dir}. "
            f"Expected <molecule>/<basis>/<method>/output/**/*.json"
        )

    # 2. Load existing checkpoint
    if not config.force and runs_path.exists() and sweeps_path.exists():
        existing_runs = pd.read_parquet(runs_path)
        existing_sweeps = pd.read_parquet(sweeps_path)
        done_keys = set(existing_runs["run_key"])
        logger.info(f"Resuming: {len(done_keys)} runs already complete")
    else:
        existing_runs = pd.DataFrame()
        existing_sweeps = pd.DataFrame()
        done_keys = set()

    pending = [r for r in all_runs if r.run_key not in done_keys]
    logger.info(f"Pending: {len(pending)} runs")

    # 3. Run pending
    new_run_rows: list[dict[str, Any]] = []
    new_sweep_rows: list[dict[str, Any]] = []

    if pending:
        if config.workers > 1:
            new_run_rows, new_sweep_rows = _run_parallel(pending, config)
        else:
            new_run_rows, new_sweep_rows = _run_serial(pending, config)

    # 4. Merge and save
    runs_df = _concat(existing_runs, new_run_rows)
    sweeps_df = _concat(existing_sweeps, new_sweep_rows)
    runs_df.to_parquet(runs_path, index=False)
    sweeps_df.to_parquet(sweeps_path, index=False)
    logger.info(f"Saved {len(runs_df)} run rows, {len(sweeps_df)} sweep rows")

    return runs_df, sweeps_df


# =====================================================================
# Parallel / serial execution
# =====================================================================

def _run_serial(
    runs: list[OrbitalRun], config: BenchmarkConfig,
) -> tuple[list[dict], list[dict]]:
    run_rows, sweep_rows = [], []
    for i, run in enumerate(runs, 1):
        logger.info(f"[{i}/{len(runs)}] {run.run_key}")
        try:
            rr, sr = _process_one_run(run, config)
            run_rows.append(rr)
            sweep_rows.extend(sr)
        except Exception:
            logger.error(f"Failed on {run.run_key}:\n{traceback.format_exc()}")
    return run_rows, sweep_rows


def _run_parallel(
    runs: list[OrbitalRun], config: BenchmarkConfig,
) -> tuple[list[dict], list[dict]]:
    run_rows, sweep_rows = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(_process_one_run, run, config): run for run in runs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            run = futures[fut]
            try:
                rr, sr = fut.result()
                run_rows.append(rr)
                sweep_rows.extend(sr)
                elapsed = time.time() - t0
                logger.info(
                    f"[{i}/{len(runs)}] {run.run_key}  "
                    f"({elapsed:.0f}s elapsed)"
                )
            except Exception:
                logger.error(
                    f"Failed on {run.run_key}:\n{traceback.format_exc()}"
                )
    return run_rows, sweep_rows


# =====================================================================
# Worker function
# =====================================================================

def _process_one_run(
    run: OrbitalRun, config: BenchmarkConfig,
) -> tuple[dict, list[dict]]:
    """Load, measure, sweep, and return (runs_row, sweeps_rows)."""
    loaded = load_run(run, strict=config.strict_sanity)

    # Resource metrics
    res = measure_resources(
        loaded,
        num_trotter_steps=config.num_trotter_steps,
        trotter_order=config.trotter_order,
        transpile_level=config.transpile_level,
    )

    # Accuracy metrics
    acc = measure_accuracy(loaded)

    runs_row = {
        "run_key": run.run_key,
        "molecule": run.molecule,
        "basis": run.basis,
        "method": run.method,
        "config": run.config,
        "json_path": str(run.json_path),
        **res.to_row(),
        **acc.to_row(),
    }

    # Sweeps
    sweep_rows: list[dict] = []
    sweep_rows.extend(
        row.to_row()
        for row in run_qse_sweep(
            loaded,
            krylov_dims=config.krylov_dims,
            num_trotter_steps=config.num_trotter_steps,
            trotter_order=config.trotter_order,
            threshold=config.threshold_qse,
            transpile_level=config.transpile_level,
            cache_dir=config.output_dir / "synthesis_cache",
            verbose=config.verbose,
        )
    )
    sweep_rows.extend(
        row.to_row()
        for row in run_sqd_sweep(
            loaded,
            krylov_dims=config.krylov_dims,
            num_trotter_steps=config.num_trotter_steps,
            trotter_order=config.trotter_order,
            num_samples=config.num_samples,
            threshold=config.threshold_sqd,
            transpile_level=config.transpile_level,
            cache_dir=config.output_dir / "synthesis_cache",
            verbose=config.verbose,
        )
    )

    for row in sweep_rows:
        row["method"] = run.method
        row["config"] = run.config

    return runs_row, sweep_rows


# =====================================================================
# Helpers
# =====================================================================

def _filter_runs(
    runs: list[OrbitalRun], config: BenchmarkConfig,
) -> list[OrbitalRun]:
    """Apply method and (later) other CLI filters."""
    if config.methods:
        wanted = {m.upper() for m in config.methods}
        runs = [r for r in runs if r.method.upper() in wanted]
    return runs


def _concat(
    existing: pd.DataFrame, new_rows: list[dict],
) -> pd.DataFrame:
    """Concatenate an existing DataFrame with a list of new row dicts."""
    if not new_rows:
        return existing
    new_df = pd.DataFrame(new_rows)
    if existing.empty:
        return new_df
    return pd.concat([existing, new_df], ignore_index=True)
