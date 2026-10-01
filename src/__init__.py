"""
Quantum Subspace Expansion / Krylov Quantum Diagonalization
============================================================

Modular implementation of QSE and SQD for molecular ground-state
estimation on near-term quantum hardware.

Public API
----------
Hamiltionians:  create_heisenberg_hamiltonian, create_molecular_hamiltonian,
                create_single_excitation_reference, get_reference_state_circuit
Circuits:       build_trotter_circuit, build_extended_swap_test,
                build_control_free_swap_test, build_hadamard_test_circuit
OVOS bridge:    load_ovos_data, ovos_to_qubit_problem, find_hf_bitstring
Solvers:        QSESolver, run_baseline_qse, SQDSolver, run_sqd
Utilities:      solve_thresholded_gevp, compute_exact_ground_state,
                compute_exact_ground_state_subspace, plot_energy_convergence,
                plot_comparison, compute_energy_error
"""

from .hamiltonian import (
    create_heisenberg_hamiltonian,
    create_molecular_hamiltonian,
    create_single_excitation_reference,
    get_reference_state_circuit,
)
from .circuits import (
    build_trotter_circuit,
    build_extended_swap_test,
    build_control_free_swap_test,
    build_hadamard_test_circuit,
)
from .ovos_bridge import (
    load_ovos_data,
    ovos_to_qubit_problem,
    find_hf_bitstring,
)
from .qse_baseline import QSESolver, run_baseline_qse
from .sqd_enhanced import SQDSolver, run_sqd
from .utils import (
    solve_thresholded_gevp,
    compute_exact_ground_state,
    compute_exact_ground_state_subspace,
    plot_energy_convergence,
    plot_comparison,
    compute_energy_error,
)

__version__ = "0.2.0"

__all__ = [
    # Hamiltonian
    "create_heisenberg_hamiltonian",
    "create_molecular_hamiltonian",
    "create_single_excitation_reference",
    "get_reference_state_circuit",
    # Circuits
    "build_trotter_circuit",
    "build_extended_swap_test",
    "build_control_free_swap_test",
    "build_hadamard_test_circuit",
    # OVOS bridge
    "load_ovos_data",
    "ovos_to_qubit_problem",
    "find_hf_bitstring",
    # Solvers
    "QSESolver",
    "run_baseline_qse",
    "SQDSolver",
    "run_sqd",
    # Utils
    "solve_thresholded_gevp",
    "compute_exact_ground_state",
    "compute_exact_ground_state_subspace",
    "plot_energy_convergence",
    "plot_comparison",
    "compute_energy_error",
]
