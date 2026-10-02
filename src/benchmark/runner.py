"""
Orchestrator for the orbital-method benchmarking study.

Guarantees
----------
- Synthesis cache is written before parallel workers start, so no two
  workers ever write the same .qpy file.
- Parquet checkpoints are deduplicated on (run_key, algorithm, krylov_dim).
- --force clears exact/ and sweeps_raw/ but keeps synthesis_cache/.
"""

from __future__ import annotations

import logging
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import EXACT_DIR, SWEEPS_RAW_DIR, save_exact, save_sweep
from .discovery import OrbitalRun, discover_runs
from .loader import load_run
from .metrics import measure_accuracy
from .resources import measure_resources
from .sweeps import (
    build_exact_artifact,
    run_qse_sweep,
    run_sqd_sweep,
    synthesis_cache_path,
    synthesis_key,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Config
# =====================================================================

@dataclass
class BenchmarkConfig:
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
    # Points 11: prefer strongly-correlated systems
    priority_systems: tuple[str, ...] = ()
    priority_only: bool = False


# =====================================================================
# Entry point
# =====================================================================

def run_benchmark(config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = config.output_dir / "runs.parquet"
    sweeps_path = config.output_dir / "sweeps.parquet"

    # Point 6: --force clears the artifact directories
    if config.force:
        for sub in (EXACT_DIR, SWEEPS_RAW_DIR):
            target = config.output_dir / sub
            if target.exists():
                logger.info("--force: removing %s", target)
                shutil.rmtree(target)
        if runs_path.exists():
            runs_path.unlink()
        if sweeps_path.exists():
            sweeps_path.unlink()

    # 1. Discover runs
    all_runs = _filter_runs(discover_runs(config.data_dir), config)
    logger.info("Discovered %d JSON files under %s",
                len(all_runs), config.data_dir)
    if not all_runs:
        raise SystemExit(
            f"No JSON files found under {config.data_dir}. "
            f"Expected <molecule>/<basis>/<method>/output/**/*.json"
        )

    # 2. Load checkpoint
    existing_runs, existing_sweeps, done_keys = _load_checkpoint(
        runs_path, sweeps_path,
    )
    logger.info("Resuming: %d runs already complete", len(done_keys))

    pending = [r for r in all_runs if r.run_key not in done_keys]
    logger.info("Pending: %d runs", len(pending))

    # 3. Point 2: serial pre-synthesis pass
    if pending:
        _pre_synthesize(pending, config)

    # 4. Parallel sweep
    if pending:
        if config.workers > 1:
            new_runs, new_sweeps = _run_parallel(pending, config)
        else:
            new_runs, new_sweeps = _run_serial(pending, config)
    else:
        new_runs, new_sweeps = [], []

    # 5. Concat + dedup + save
    runs_df = _checkpoint_concat(existing_runs, new_runs, ["run_key"])
    sweeps_df = _checkpoint_concat(
        existing_sweeps, new_sweeps,
        ["run_key", "algorithm", "krylov_dim"],
    )
    runs_df.to_parquet(runs_path, index=False)
    sweeps_df.to_parquet(sweeps_path, index=False)
    logger.info("Saved %d run rows, %d sweep rows",
                len(runs_df), len(sweeps_df))

    return runs_df, sweeps_df


# =====================================================================
# Pre-synthesis
# =====================================================================

def _pre_synthesize(runs: list[OrbitalRun], config: BenchmarkConfig) -> None:
    """
    Build every Trotter circuit serially before parallel workers start.

    This is the point-2 fix: workers can now assume cache hits, so they
    never write to the same .qpy concurrently.
    """
    cache_dir = config.output_dir / "synthesis_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Pre-synthesizing %d runs × 2 algorithms", len(runs))
    t0 = time.time()
    for i, run in enumerate(runs, 1):
        try:
            loaded = load_run(run, strict=config.strict_sanity)
        except Exception:
            logger.error(
                "Pre-synthesis: failed to load %s:\n%s",
                run.run_key, traceback.format_exc(),
            )
            continue
        for tag, builder in (
            ("qse", _build_qse_circuit),
            ("sqd", _build_sqd_circuit),
        ):
            try:
                circuit = builder(loaded, config)
            except Exception:
                logger.error(
                    "Pre-synthesis: failed to build %s circuit for %s:\n%s",
                    tag, run.run_key, traceback.format_exc(),
                )
                continue
            key = synthesis_key(
                loaded, tag, config.num_trotter_steps,
                config.trotter_order, config.transpile_level,
            )
            cache_file = synthesis_cache_path(cache_dir, key)
            if cache_file.exists():
                continue
            _write_synthesis(cache_file, circuit, config.transpile_level)
        if config.verbose or i % 5 == 0:
            logger.info("  [%d/%d] %s  (%.0fs)",
                        i, len(runs), run.run_key, time.time() - t0)


def _build_qse_circuit(loaded, config) -> "QuantumCircuit":
    from ..qse_baseline import QSESolver
    solver = QSESolver(
        hamiltonian=loaded.hamiltonian,
        reference_circuit=loaded.reference_circuit,
        reference_bitstring=loaded.ref_bitstring,
        krylov_dim=max(config.krylov_dims),
        num_trotter_steps=config.num_trotter_steps,
        trotter_order=config.trotter_order,
        use_shifting=False,
        threshold=config.threshold_qse,
    )
    return solver._build_circuit()


def _build_sqd_circuit(loaded, config) -> "QuantumCircuit":
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from ..circuits import build_trotter_circuit

    evolution = build_trotter_circuit(
        loaded.hamiltonian, time=0.0,
        num_trotter_steps=config.num_trotter_steps,
        order=config.trotter_order, parameterized=True,
    )
    t_sqd = Parameter("t_sqd")
    evolution = evolution.assign_parameters(
        {list(evolution.parameters)[0]: t_sqd}
    )
    qc = QuantumCircuit(loaded.num_qubits)
    qc.compose(loaded.reference_circuit, inplace=True)
    qc.compose(evolution, inplace=True)
    return qc


def _write_synthesis(cache_file: Path, circuit, transpile_level: int) -> None:
    """Transpile a circuit and write it atomically to the cache."""
    from qiskit import qpy, transpile
    from .sweeps import BASIS_GATES

    logger.debug("  synthesizing %s", cache_file.name)
    qc_synth = transpile(circuit, basis_gates=BASIS_GATES,
                         optimization_level=transpile_level)
    tmp = cache_file.with_suffix(".qpy.tmp")
    with open(tmp, "wb") as f:
        qpy.dump(qc_synth, f)
    tmp.replace(cache_file)


# =====================================================================
# Parallel / serial execution
# =====================================================================

def _run_serial(runs, config):
    run_rows, sweep_rows = [], []
    for i, run in enumerate(runs, 1):
        logger.info("[%d/%d] %s", i, len(runs), run.run_key)
        try:
            rr, sr = _process_one_run(run, config)
            run_rows.append(rr)
            sweep_rows.extend(sr)
        except Exception:
            logger.error("Failed on %s:\n%s",
                         run.run_key, traceback.format_exc())
    return run_rows, sweep_rows


def _run_parallel(runs, config):
    run_rows, sweep_rows = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(_process_one_run, r, config): r for r in runs}
        for i, fut in enumerate(as_completed(futures), 1):
            run = futures[fut]
            try:
                rr, sr = fut.result()
                run_rows.append(rr)
                sweep_rows.extend(sr)
                logger.info("[%d/%d] %s  (%.0fs elapsed)",
                            i, len(runs), run.run_key, time.time() - t0)
            except Exception:
                logger.error("Failed on %s:\n%s",
                             run.run_key, traceback.format_exc())
    return run_rows, sweep_rows


# =====================================================================
# Worker
# =====================================================================

def _process_one_run(run: OrbitalRun, config: BenchmarkConfig):
    loaded = load_run(run, strict=config.strict_sanity)

    save_exact(config.output_dir, run.run_key, build_exact_artifact(loaded))

    res = measure_resources(
        loaded,
        num_trotter_steps=config.num_trotter_steps,
        trotter_order=config.trotter_order,
        transpile_level=config.transpile_level,
    )
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

    qse_rows, qse_art = run_qse_sweep(
        loaded,
        krylov_dims=config.krylov_dims,
        num_trotter_steps=config.num_trotter_steps,
        trotter_order=config.trotter_order,
        threshold=config.threshold_qse,
        transpile_level=config.transpile_level,
        cache_dir=config.output_dir / "synthesis_cache",
        verbose=config.verbose,
    )
    save_sweep(config.output_dir, run.run_key, qse_art)

    sqd_rows, sqd_art = run_sqd_sweep(
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
    save_sweep(config.output_dir, run.run_key, sqd_art)

    sweep_rows = []
    for row in (*qse_rows, *sqd_rows):
        d = row.to_row()
        d["method"] = run.method
        d["config"] = run.config
        sweep_rows.append(d)

    return runs_row, sweep_rows


# =====================================================================
# Checkpoint helpers
# =====================================================================

def _load_checkpoint(runs_path: Path, sweeps_path: Path):
    if runs_path.exists() and sweeps_path.exists():
        runs = pd.read_parquet(runs_path)
        sweeps = pd.read_parquet(sweeps_path)
        return runs, sweeps, set(runs["run_key"])
    return pd.DataFrame(), pd.DataFrame(), set()


def _checkpoint_concat(
    existing: pd.DataFrame, new_rows: list[dict], key_cols: list[str],
) -> pd.DataFrame:
    """
    Concatenate and deduplicate on `key_cols`, keeping the newest row.

    Point 5: prevents duplicate rows when a rerun overlaps with a
    previous checkpoint.
    """
    if not new_rows:
        return existing
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True) \
        if not existing.empty else new_df
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    return combined.reset_index(drop=True)


# =====================================================================
# Filters
# =====================================================================

def _filter_runs(runs: list[OrbitalRun], config: BenchmarkConfig) -> list[OrbitalRun]:
    if config.methods:
        wanted = {m.upper() for m in config.methods}
        runs = [r for r in runs if r.method.upper() in wanted]
    if config.priority_systems:
        priority = {s.lower() for s in config.priority_systems}
        pri = [r for r in runs if r.molecule.lower() in priority]
        if config.priority_only:
            return pri
        rest = [r for r in runs if r.molecule.lower() not in priority]
        return pri + rest
    return runs