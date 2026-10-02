"""
Discover OVOS/COVO/... JSON files in a hierarchical data directory.

Expected layout::

    <data-dir>/<molecule>/<basis>/<method>/output/[<config>/...]/file.json

where ``<config>`` is an optional sequence of subdirectory names that
distinguish different runs of the same method (e.g. ``initfock_virtual``,
``aux1/reopt1``).  The config is encoded with ``::`` separators in the
:class:`OrbitalRun` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# A directory literally named "output" is the anchor that separates the
# method name from the per-configuration subdirectories.
_OUTPUT_MARKER = "output"


@dataclass(frozen=True)
class OrbitalRun:
    """A single JSON file to benchmark."""

    json_path: Path
    molecule: str
    basis: str
    method: str
    config: str  # "default" if no subdirectories

    @property
    def run_key(self) -> str:
        """Human-readable unique identifier used as the parquet join key."""
        parts = [self.molecule, self.basis, self.method]
        if self.config != "default":
            parts.append(self.config)
        return "/".join(parts)


def discover_runs(data_dir: Path) -> list[OrbitalRun]:
    """
    Find every JSON file matching the expected directory layout.

    Parameters
    ----------
    data_dir : Path
        Root directory. Must contain ``<molecule>/<basis>/<method>/output/``.

    Returns
    -------
    list[OrbitalRun]
        Discovered runs, sorted by ``(molecule, basis, method, config)``.
    """
    runs: list[OrbitalRun] = []
    for json_path in data_dir.rglob("*.json"):
        run = _parse_path(json_path, data_dir)
        if run is not None:
            runs.append(run)
    runs.sort(key=lambda r: (r.molecule, r.basis, r.method, r.config))
    return runs


def _parse_path(json_path: Path, data_dir: Path) -> OrbitalRun | None:
    """Return an OrbitalRun if the path matches the expected layout, else None."""
    try:
        rel = json_path.relative_to(data_dir)
    except ValueError:
        return None

    parts = rel.parts
    if _OUTPUT_MARKER not in parts:
        return None
    idx = parts.index(_OUTPUT_MARKER)

    pre = parts[:idx]                    # molecule / basis / method
    post = parts[idx + 1 : -1]           # optional config subdirs

    if len(pre) < 3:
        return None

    molecule, basis, method = pre[0], pre[1], pre[2]
    config_parts = list(pre[3:]) + list(post)
    config = "::".join(config_parts) if config_parts else "default"

    return OrbitalRun(
        json_path=json_path,
        molecule=molecule,
        basis=basis,
        method=method,
        config=config,
    )
