def one_body_rdm_from_coeffs(
    coeffs: np.ndarray,
    bitstrings: list[str],
    num_qubits: int,
) -> np.ndarray:
    """
    Compute Γ_pq = <Ψ| a_p† a_q |Ψ> from a QSD wavefunction.

    Ψ = Σ c_i |b_i⟩ over the given bitstrings.

    Parameters
    ----------
    coeffs : (d,) complex
        QSD eigenvector.
    bitstrings : (d,) list of str
    num_qubits : int

    Returns
    -------
    gamma : (n_spin_orbitals, n_spin_orbitals) complex
        Spin-orbital 1-RDM. Block by alpha/beta as needed.
    """
    n = num_qubits
    gamma = np.zeros((n, n), dtype=complex)
    # ... apply Jordan–Wigner creation/annihilation operators
    return gamma


def natural_orbitals_from_rdm(
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize the 1-RDM.

    Returns
    -------
    occupations : (n,) float
        Natural orbital occupations (eigenvalues of Γ).
    rotation : (n, n) complex
        Columns are the natural orbitals in the original basis.
    """
    evals, evecs = np.linalg.eigh(0.5 * (gamma + gamma.conj().T))
    # sort descending by occupation
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]