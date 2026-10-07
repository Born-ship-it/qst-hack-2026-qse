import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.benchmark.discovery import discover_runs
from src.benchmark.loader import load_run
from src.utils import subspace_matrix_elements, solve_thresholded_gevp

run = [r for r in discover_runs(Path("data"), layout="auto")
       if r.run_key.endswith("baseline_6-31g_ncas06.eval")][0]
loaded = load_run(run)

# Get the 12q bitstrings (small enough for dense)
from itertools import combinations
n = loaded.hamiltonian.num_qubits
bs = []
for occ in combinations(range(n), 8):
    bits = ["0"] * n
    for q in occ:
        bits[n - 1 - q] = "1"
    bs.append("".join(bits))

# Force both paths
h_dense, s_dense = subspace_matrix_elements(loaded.hamiltonian, bs, sparse=False)
h_sparse, s_sparse = subspace_matrix_elements(loaded.hamiltonian, bs, sparse=True)

# Compare against the sparse→dense conversion
h_from_sparse = h_sparse.toarray()
print("h agrees:", np.allclose(h_dense, h_from_sparse, atol=1e-12))

# GEVP on both paths
e_d, c_d, r_d = solve_thresholded_gevp(h_dense, s_dense, threshold=1e-10)
e_s, c_s, r_s = solve_thresholded_gevp(h_sparse, s_sparse, threshold=1e-10)
print(f"dense  e0 = {e_d:.12f}")
print(f"sparse e0 = {e_s:.12f}")
print(f"|diff|    = {abs(e_d - e_s):.2e}")
assert abs(e_d - e_s) < 1e-10