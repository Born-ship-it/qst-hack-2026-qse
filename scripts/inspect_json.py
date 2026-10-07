#!/usr/bin/env python3
"""
Inspect OVOS/AVAS/COVO/... JSON files for schema compliance.

Usage:
    python scripts/inspect_json.py data/avas/ch2_singlet/6-31g/output
    python scripts/inspect_json.py data/ --all
    python scripts/inspect_json.py data/ovos/ --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# Keys the loader requires (see src/benchmark/loader.py and ovos_bridge.py).
REQUIRED_KEYS = {
    "one_electron_integrals",
    "two_electron_integrals",
    "n_active_orbitals",
    "n_active_electrons",
    "nuclear_repulsion_energy",
    "active_hf_energy",
    "casci_energy",
    "full_fci_energy",
}

# Keys that are nice to have; the loader degrades gracefully if absent.
OPTIONAL_KEYS = {
    "active_mp2_correlation_energy",
    "n_active_occ",
    "n_active_vir",
    "basis_set",
    "method",
    "casscf_energy",
    "full_fci_n_orbitals",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "path", type=Path,
        help="A JSON file, a directory to scan recursively, or a directory "
             "whose immediate subdirectories contain JSONs.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Scan all JSONs under the path (default: first one only).",
    )
    p.add_argument(
        "--summary", action="store_true",
        help="Print only the missing-keys summary (no per-file detail).",
    )
    return p.parse_args()


def find_jsons(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.json"))
    return []


def inspect_one(json_path: Path) -> tuple[set[str], dict]:
    """Return (keys, payload). Raises on invalid JSON."""
    with open(json_path, "r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{json_path}: top-level JSON is not an object")
    return set(payload.keys()), payload


def main() -> int:
    args = parse_args()

    paths = find_jsons(args.path)
    if not paths:
        print(f"No JSON files found under {args.path}", file=sys.stderr)
        return 1

    if not args.all:
        paths = paths[:1]

    missing_counter: Counter[tuple[str, ...]] = Counter()
    first_ok = None

    for json_path in paths:
        try:
            keys, _ = inspect_one(json_path)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[ERROR] {json_path}: {exc}")
            continue

        missing = REQUIRED_KEYS - keys
        extra = keys - REQUIRED_KEYS - OPTIONAL_KEYS

        missing_counter[tuple(sorted(missing))] += 1
        if not missing and first_ok is None:
            first_ok = json_path

        if not args.summary:
            print(f"\n{json_path}")
            print(f"  total keys:      {len(keys)}")
            print(f"  required found:  {len(REQUIRED_KEYS - missing)}/{len(REQUIRED_KEYS)}")
            if missing:
                print(f"  MISSING:         {sorted(missing)}")
            if extra:
                print(f"  extra keys:      {sorted(extra)}")

    # Summary
    print("\n" + "=" * 72)
    print(f"Summary over {len(paths)} file(s)")
    print("=" * 72)
    if not missing_counter or tuple() in missing_counter:
        n_ok = missing_counter.get(tuple(), 0)
        print(f"  {n_ok} file(s) have all required keys")
    for missing_tuple, count in missing_counter.most_common():
        if not missing_tuple:
            continue
        print(f"  {count} file(s) missing: {list(missing_tuple)}")

    if first_ok is not None:
        print(f"\nExample valid file: {first_ok}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())