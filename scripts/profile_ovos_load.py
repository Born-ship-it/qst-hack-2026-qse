import cProfile
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark.discovery import discover_runs
from src.benchmark.loader import load_run

target = "ch2_singlet_ovos_6-31g_nvirt3.eval"
runs = [r for r in discover_runs(Path("data"), layout="auto")
        if r.run_key.endswith(target)]
assert len(runs) == 1, runs

pr = cProfile.Profile()
pr.enable()
load_run(runs[0], strict=True)
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(25)