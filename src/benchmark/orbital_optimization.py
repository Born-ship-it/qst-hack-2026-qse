def qsd_orbital_loop(
    hamiltonian_factory,   # callable: (rotation) → (H_qubit, ref_bitstring)
    initial_rotation,      # (n, n) unitary from the initial basis
    qsd_config,
    num_iterations: int = 4,
    convergence_tol: float = 1e-5,
) -> OrbitalOptimizationResult:
    """
    Self-consistent QSD + natural orbital re-optimization.

    At each iteration:
        1. Build H in the current orbital basis
        2. Run QSD  →  E, coeffs
        3. Compute Γ, diagonalize → NOs
        4. Update the orbital rotation
    """
    rotation = initial_rotation
    energies = []
    for it in range(num_iterations):
        H, ref_bs = hamiltonian_factory(rotation)
        # ... run QSD, get coeffs and bitstrings ...
        gamma = one_body_rdm_from_coeffs(coeffs, bitstrings, n)
        occupations, no_rotation = natural_orbitals_from_rdm(gamma)
        rotation = rotation @ no_rotation
        energies.append(E_total)
        if it > 0 and abs(energies[-1] - energies[-2]) < convergence_tol:
            break
    return OrbitalOptimizationResult(energies, rotation, ...)