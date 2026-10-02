"""Tests for src.benchmark.discovery."""

from __future__ import annotations

from pathlib import Path

from src.benchmark.discovery import discover_runs


def test_discovers_single_run(fake_h2_json: Path):
    runs = discover_runs(fake_h2_json)
    assert len(runs) == 1
    r = runs[0]
    assert r.molecule == "h2"
    assert r.basis == "sto-3g"
    assert r.method == "OVOS"
    assert r.config == "initfoh_virtual"
    assert r.run_key == "h2/sto-3g/OVOS/initfoh_virtual"


def test_discovers_multiple_methods(fake_h2_two_methods: Path):
    runs = discover_runs(fake_h2_two_methods)
    assert len(runs) == 2
    methods = {r.method for r in runs}
    assert methods == {"OVOS", "RHF"}


def test_ignores_json_outside_output(tmp_path: Path):
    # JSON not under an output/ directory should be ignored
    (tmp_path / "h2" / "sto-3g" / "OVOS").mkdir(parents=True)
    (tmp_path / "h2" / "sto-3g" / "OVOS" / "stray.json").write_text("{}")
    assert discover_runs(tmp_path) == []


def test_empty_directory(tmp_path: Path):
    assert discover_runs(tmp_path) == []