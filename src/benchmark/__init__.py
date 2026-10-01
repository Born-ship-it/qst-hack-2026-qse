"""
Benchmarking subpackage for QSD methods across orbital-selection strategies.

This subpackage provides:

- discovery:    walk a data directory and identify JSON runs
- loader:       load and sanity-check each JSON into a LoadedRun
- resources:    quantum-circuit and Pauli-term resource metrics
- metrics:      accuracy, overlap, and spectral-gap metrics
- sweeps:       run QSE / SQD / noise sweeps per LoadedRun
- runner:       orchestrate the full study with parallelism + parquet cache
- plots:        one function per figure
- style:        shared plotting palette and helpers
"""

from .discovery import OrbitalRun, discover_runs
from .loader import LoadedRun, load_run
from .resources import ResourceMetrics, measure_resources
from .metrics import AccuracyMetrics, measure_accuracy
from .sweeps import QSERow, SQDRow, run_qse_sweep, run_sqd_sweep
from .runner import run_benchmark

__all__ = [
    "OrbitalRun",
    "discover_runs",
    "LoadedRun",
    "load_run",
    "ResourceMetrics",
    "measure_resources",
    "AccuracyMetrics",
    "measure_accuracy",
    "QSERow",
    "SQDRow",
    "run_qse_sweep",
    "run_sqd_sweep",
    "run_benchmark",
]
