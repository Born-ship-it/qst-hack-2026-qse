#!/usr/bin/env python3
"""
Single-bit-flip noise sweep for SQD subspaces.

Reads sweeps_raw/<key>__SQD.npz from a completed benchmark, corrupts the
sampled counts at several noise levels, rebuilds the projected Hamiltonian,
and solves the GEVP. Writes noise.parquet alongside the benchmark outputs.

Example
-------
    python scripts/noise_sweep.py \\
        --artifacts results/h2_ccpvdz \\
        --noise-levels 0,0.01,0.02,0.05,0.1,0.2 \\
        --workers 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.artifacts import load_sweep  # noqa: E402
from src.benchmark.discovery import OrbitalRun  # noqa: E402
from src.benchmark.loader import load_run  # noqa: E402
from src.utils import solve_thresholded_gevp, subspace_matrix_elements  # noqa: E402


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, required=True,
                   help="Benchmark output directory")
    p.add_argument("--noise-levels", type=str, default="0,0.01,0.02,0.05,0.1,0.2")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    noise_levels = tuple(float(x) for x in args.noise_levels.split(","))
    runs_df = pd.read_parquet(args.artifacts / "runs.parquet")

    jobs = list(runs_df.to_dict("records"))
    logger.info("Noise sweep: %d runs × %d levels", len(jobs), len(noise_levels))

    rows: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(_noise_one, job, args.artifacts, noise_levels, args.seed):
                    job for job in jobs
            }
            for fut in as_completed(futures):
                try:
                    rows.extend(fut.result())
                except Exception as exc:
                    logger.error("Failed on %s: %s",
                                 futures[fut]["run_key"], exc)
    else:
        for job in jobs:
            rows.extend(_noise_one(job, args.artifacts, noise_levels, args.seed))

    df = pd.DataFrame(rows)
    out_path = args.artifacts / "noise.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df))
    return 0


# ---------------------------------------------------------------------

def _noise_one(job: dict, artifacts_dir: Path, noise_levels, seed: int) -> list[dict]:
    run_key = job["run_key"]
    json_path = Path(job["json_path"])
    loaded = load_run(
        OrbitalRun(
            json_path=json_path,
            molecule=job["molecule"], basis=job["basis"],
            method=job["method"], config=job["config"],
        ),
        strict=False,
    )

    artifact = load_sweep(artifacts_dir, run_key, "SQD")

    # Use the last krylov dim's sampled counts as the baseline
    k_idx = len(artifact.krylov_dims) - 1
    counts = artifact.sampled_counts[k_idx]         # (n_union,)
    union_bs = [str(x) for x in artifact.union_bitstrings]

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for noise in noise_levels:
        corrupted = _corrupt_counts(counts, union_bs, noise, rng)
        noisy_bitstrings = sorted(
            bs for bs, c in corrupted.items() if c > 0
        )
        h_mat, s_mat = subspace_matrix_elements(
            loaded.hamiltonian, noisy_bitstrings,
        )
        try:
            e_elec, _, retained = solve_thresholded_gevp(h_mat, s_mat, 1e-8)
            e_total = e_elec + loaded.nuclear_repulsion
            err_mha = abs(e_total - loaded.casci_total) * 1000.0
        except ValueError:
            e_total = float("nan")
            err_mha = float("nan")
            retained = 0

        rows.append({
            "run_key": run_key,
            "molecule": job["molecule"],
            "basis": job["basis"],
            "method": job["method"],
            "noise": noise,
            "energy_total": e_total,
            "min_err_mha": err_mha,
            "subspace_dim": len(noisy_bitstrings),
            "retained_dim": retained,
        })
    return rows


def _corrupt_counts(counts, union_bs, noise_level, rng):
    """
    Apply single-bit-flip corruption to aggregated counts.

    Each shot has probability `noise_level` of one bit being flipped.
    """
    from collections import Counter

    if noise_level <= 0:
        return {bs: int(c) for bs, c in zip(union_bs, counts) if c > 0}

    n_qubits = len(union_bs[0])
    out = Counter()
    for bs, c in zip(union_bs, counts):
        c = int(c)
        if c == 0:
            continue
        n_corrupt = rng.binomial(c, noise_level)
        out[bs] += c - n_corrupt
        if n_corrupt > 0:
            positions = rng.integers(0, n_qubits, size=n_corrupt)
            for pos in positions:
                b = list(bs)
                b[pos] = "1" if b[pos] == "0" else "0"
                out["".join(b)] += 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())