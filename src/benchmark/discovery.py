"""
Discover OVOS/COVO/AVAS/... JSON files in a hierarchical data directory.

Two layouts are supported:

    molecule-first:  <data>/<molecule>/<basis>/<method>/output/...
    method-first:    <data>/<method>/<molecule>/<basis>/output/...

Auto-detection inspects the first path component and matches it against
a set of known method names.  If it matches, method-first is assumed;
otherwise molecule-first.

Files are matched by the ``.json`` suffix, which also catches the
``*.eval.json`` naming convention used by the evaluator outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Directory that anchors the split between (molecule, basis, method) and
# the per-configuration subdirectories.
_OUTPUT_MARKER = "output"

# Recognised method-directory names for layout auto-detection.
_KNOWN_METHODS: frozenset[str] = frozenset({
    "ovos", "avas", "covo", "covo_orb", "baseline",
    "no_mp2", "no-mp2", "rhf", "mp2", "ccsd", "ccsd_t",
})

# Only files matching this glob are considered.  The evaluator writes
# `*.eval.json` (metrics + integrals) alongside a raw `*.json` file; we
# want the evaluated one only.
_FILE_GLOB = "*.eval.json"

# =====================================================================
# OrbitalRun
# =====================================================================

@dataclass(frozen=True)
class OrbitalRun:
    """A single JSON file to benchmark."""

    json_path: Path
    molecule: str
    basis: str
    method: str
    config: str  # "default" if the file sits directly under output/

    @property
    def json_stem(self) -> str:
        """
        Filename without the trailing ``.json``.

        Handles both ``foo.json`` and ``foo.eval.json``; the latter
        keeps the ``.eval`` part because it is part of the evaluator
        naming convention and helps distinguish files.
        """
        name = self.json_path.name
        if name.endswith(".json"):
            name = name[:-5]
        return name

    @property
    def run_key(self) -> str:
        """
        Unique identifier for this run.

        Combines the three identity fields, the config subpath (if any),
        and the filename stem.  The stem is what disambiguates files that
        sit in the same output/ directory with different parameter values
        (ncas04 vs ncas05, nvirt1 vs nvirt2, etc.).
        """
        parts = [self.molecule, self.basis, self.method]
        if self.config != "default":
            parts.append(self.config)
        parts.append(self.json_stem)
        return "::".join(parts)

    @property
    def display_key(self) -> str:
        """
        Short label for plots and legends.

        Drops the redundant molecule/basis/method prefix from the stem,
        keeping only the parameter that varies within a method.
        """
        prefix = f"{self.molecule}_{self.method}_{self.basis}_"
        stem = self.json_stem
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
        # Strip a trailing ".eval" if present
        if stem.endswith(".eval"):
            stem = stem[:-5]
        if self.config != "default":
            return f"{self.method}/{self.config}/{stem}"
        return f"{self.method}/{stem}"

    @property
    def short_config(self) -> str:
        """Compact config label (unchanged from before)."""
        if self.config == "default":
            return ""
        parts = self.config.replace("::", "/").split("/")
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1]


# =====================================================================
# Discovery
# =====================================================================

def discover_runs(
    data_dir: Path,
    layout: str = "auto",
) -> list[OrbitalRun]:
    """
    Find every JSON file matching the expected directory layout.

    Parameters
    ----------
    data_dir : Path
        Root directory.  Must contain ``<molecule>/<basis>/<method>/output/``
        or ``<method>/<molecule>/<basis>/output/``.
    layout : {"auto", "molecule_first", "method_first"}
        Directory ordering.  ``"auto"`` (default) inspects the first
        component of each path and matches against known method names.

    Returns
    -------
    list[OrbitalRun]
        Discovered runs, sorted by (molecule, basis, method, config).
    """
    if layout not in ("auto", "molecule_first", "method_first"):
        raise ValueError(
            f"layout must be 'auto', 'molecule_first', or 'method_first'; "
            f"got {layout!r}"
        )

    runs: list[OrbitalRun] = []
    for json_path in data_dir.rglob(_FILE_GLOB):
        run = _parse_path(json_path, data_dir, layout)
        if run is not None:
            runs.append(run)

    runs.sort(key=lambda r: (r.molecule, r.basis, r.method, r.config))
    return runs


# =====================================================================
# Path parsing
# =====================================================================

def _parse_path(
    json_path: Path, data_dir: Path, layout: str,
) -> OrbitalRun | None:
    """Return an OrbitalRun if the path matches the expected layout, else None."""
    try:
        rel = json_path.relative_to(data_dir)
    except ValueError:
        return None

    parts = rel.parts
    if _OUTPUT_MARKER not in parts:
        return None
    idx = parts.index(_OUTPUT_MARKER)

    pre = parts[:idx]                 # three identity fields (+ optional extras)
    post = parts[idx + 1 : -1]        # config subdirectories

    if len(pre) < 3:
        # Not enough context.  Happens when --data-dir is set to a
        # subdirectory like data/<method>/<molecule>/<basis>.  We could
        # guess, but it's better to require the full tree.
        return None

    a, b, c = pre[0], pre[1], pre[2]
    extra_pre = list(pre[3:])
    config_parts = extra_pre + list(post)

    use_method_first = (
        layout == "method_first"
        or (layout == "auto" and a.lower() in _KNOWN_METHODS)
    )

    if use_method_first:
        method, molecule, basis = a, b, c
    else:
        molecule, basis, method = a, b, c

    config = "::".join(config_parts) if config_parts else "default"

    return OrbitalRun(
        json_path=json_path,
        molecule=molecule,
        basis=basis,
        method=method,
        config=config,
    )