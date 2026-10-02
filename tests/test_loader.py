"""Tests for src.benchmark.loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmark.discovery import OrbitalRun
from src.benchmark.loader import load_run


def _run_from_path(path: Path) -> OrbitalRun:
    return OrbitalRun(
        json_path=path,
        molecule="h2", basis="sto-3g", method="OVOS",
        config="initfoh_virtual",
    )


def test_load_run_smoke(fake_h2_json: Path):
    json_path = next(fake_h2_json.rglob("*.json"))
    loaded = load_run(_run_from_path(json_path))

    # 2 spatial orbitals -> 4 spin orbitals -> 4 qubits
    assert loaded.num_qubits == 4
    assert loaded.num_electrons == 2
    assert loaded.hf_total == pytest.approx(-1.10, abs=1e-6)
    assert loaded.casci_total == pytest.approx(-1.12, abs=1e-6)
    assert loaded.sanity_diff_ha < 1e-9


def test_reference_bitstring_is_sane(fake_h2_json: Path):
    json_path = next(fake_h2_json.rglob("*.json"))
    loaded = load_run(_run_from_path(json_path))
    # Bitstring should have exactly num_electrons ones
    assert loaded.ref_bitstring.count("1") == loaded.num_electrons
    assert len(loaded.ref_bitstring) == loaded.num_qubits


def test_sanity_check_raises_when_strict(tmp_path: Path, fake_h2_json: Path):
    """A JSON whose CASCI is off by >0.1 Ha triggers strict mode."""
    src = next(fake_h2_json.rglob("*.json"))
    payload = json.loads(src.read_text())
    payload["casci_energy"] += 0.5   # break the sanity relation
    broken = tmp_path / "h2" / "sto-3g" / "OVOS" / "output"
    broken.mkdir(parents=True)
    broken_file = broken / "broken.json"
    broken_file.write_text(json.dumps(payload))

    run = _run_from_path(broken_file)
    with pytest.raises(ValueError, match="deviates"):
        load_run(run, strict=True)

    # Non-strict should succeed with a warning
    loaded = load_run(run, strict=False)
    assert loaded.sanity_diff_ha > 0.1