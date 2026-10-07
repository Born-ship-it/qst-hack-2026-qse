#!/usr/bin/env python3
"""
CLI entry point for the orbital-method benchmarking study.

Example
-------
    python scripts/benchmark_orbital_methods.py \\
        --data-dir data/h2/cc-pvdz \\
        --output-dir results/h2_ccpvdz \\
        --methods OVOS,COVO,RHF,AVAS,NO-MP2 \\
        --krylov-max 12 --workers 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.plots import generate_all
from src.benchmark.runner import BenchmarkConfig, run_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark QSD methods across orbital-selection strategies.",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Root directory containing <molecule>/<basis>/<method>/output/")
    p.add_argument("--output-dir", type=Path, default=Path("results/benchmark"),
                   help="Where to write parquet files and figures")
    p.add_argument("--methods", type=str, default=None,
                   help="Comma-separated list of methods to include")
    p.add_argument("--krylov-min", type=int, default=2)
    p.add_argument("--krylov-max", type=int, default=12)
    p.add_argument("--krylov-step", type=int, default=2)
    p.add_argument("--num-trotter-steps", type=int, default=2)
    p.add_argument("--trotter-order", type=int, default=2, choices=[1, 2, 4])
    p.add_argument("--num-samples", type=int, default=20_000)
    p.add_argument("--transpile-level", type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--strict", action="store_true",
                   help="Fail on loader sanity check")
    p.add_argument("--force", action="store_true",
                   help="Ignore existing parquet checkpoint")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip figure generation")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--priority", type=str, default="",
                   help="Comma-separated list of molecule names to prioritize")
    p.add_argument("--priority-only", action="store_true",
                   help="Run only priority molecules, ignore the rest")
    p.add_argument(
        "--algorithms", type=str,
        default="QSE,SQD,SKQD",
        help="Comma-separated list: QSE, SQD, SKQD",
    )
    p.add_argument(
        "--layout", type=str, default="auto",
        choices=["auto", "molecule_first", "method_first"],
        help="Directory ordering under --data-dir",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    algorithms = tuple(
        a.strip().upper() for a in args.algorithms.split(",") if a.strip()
    )

    KNOWN_ALGORITHMS = {"QSE", "SQD", "SKQD"}
    unknown = set(algorithms) - KNOWN_ALGORITHMS
    if unknown:
        raise SystemExit(
            f"Unknown algorithms: {unknown}. "
            f"Valid choices: {sorted(KNOWN_ALGORITHMS)}"
        )

    krylov_dims = tuple(
        range(args.krylov_min, args.krylov_max + 1, args.krylov_step)
    )
    methods = (
        [m.strip() for m in args.methods.split(",")]
        if args.methods else None
    )
    priority = tuple(
        s.strip() for s in args.priority.split(",") if s.strip()
    )
    config = BenchmarkConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        methods=methods,
        krylov_dims=krylov_dims,
        num_trotter_steps=args.num_trotter_steps,
        trotter_order=args.trotter_order,
        num_samples=args.num_samples,
        transpile_level=args.transpile_level,
        strict_sanity=args.strict,
        workers=args.workers,
        force=args.force,
        verbose=args.verbose,
        priority_systems=priority,
        priority_only=args.priority_only,
        algorithms=algorithms,  
        layout=args.layout,
    )

    runs_df, sweeps_df = run_benchmark(config)

    if not args.no_plots:
        print(f"Generating figures in {args.output_dir / 'figures'}")
        generate_all(runs_df, sweeps_df, args.output_dir / "figures")

    print(f"\nDone. Runs: {len(runs_df)}, sweeps: {len(sweeps_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
