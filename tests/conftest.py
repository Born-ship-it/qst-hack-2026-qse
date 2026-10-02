"""
Shared pytest fixtures for the benchmark test suite.

The `fake_h2_json` fixture writes a small, self-consistent OVOS-like JSON
into a tmp_path laid out as `<molecule>/<basis>/<method>/output/...`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _make_h2_payload() -> dict:
    """Return a minimal but structurally valid OVOS payload for H2/STO-3G."""
    # h1 in a spatial-orbital basis (2 orbitals)
    h1 = np.array([
        [-1.25, 0.0],
        [ 0.0, -0.48],
    ])

    # h2 in chemist notation (2,2,2,2)
    h2 = np.zeros((2, 2, 2, 2))
    h2[0, 0, 0, 0] = 0.65
    h2[0, 0, 1, 1] = 0.18
    h2[1, 1, 0, 0] = 0.18
    h2[1, 1, 1, 1] = 0.70
    h2[0, 1, 0, 1] = 0.18
    h2[1, 0, 1, 0] = 0.18
    h2[0, 1, 1, 0] = 0.18
    h2[1, 0, 0, 1] = 0.18

    e_nuc = 0.7
    e_hf_active = -1.80
    e_mp2_corr = -0.02

    return {
        "task_title": "h2_ovos_sto-3g_nvirt1",
        "method": "OVOS",
        "basis_set": "sto-3g",
        "geometry": [0.0, 0.0, 0.0, 0.0, 0.0, 0.74],
        "symbols": ["H", "H"],
        "n_active_orbitals": 2,
        "n_active_electrons": 2,
        "n_active_occ": 1,
        "n_active_vir": 1,
        "nuclear_repulsion_energy": e_nuc,
        "active_hf_energy": e_hf_active,
        "active_mp2_correlation_energy": e_mp2_corr,
        "casci_energy": e_hf_active + e_mp2_corr + e_nuc,
        "full_fci_energy": e_hf_active + e_mp2_corr + e_nuc - 0.01,
        "one_electron_integrals": h1.tolist(),
        "two_electron_integrals": h2.tolist(),
    }


@pytest.fixture
def fake_h2_json(tmp_path: Path) -> Path:
    """Write a single H2 JSON into the expected layout."""
    run_dir = tmp_path / "h2" / "sto-3g" / "OVOS" / "output" / "initfoh_virtual"
    run_dir.mkdir(parents=True)
    out_file = run_dir / "output.json"
    out_file.write_text(json.dumps(_make_h2_payload()))
    return tmp_path


@pytest.fixture
def fake_h2_two_methods(fake_h2_json: Path) -> Path:
    """Add a second method (RHF) at the same molecule/basis."""
    payload = _make_h2_payload()
    payload["method"] = "RHF"
    run_dir = fake_h2_json / "h2" / "sto-3g" / "RHF" / "output"
    run_dir.mkdir(parents=True)
    (run_dir / "output.json").write_text(json.dumps(payload))
    return fake_h2_json