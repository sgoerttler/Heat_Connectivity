import numpy as np


def simulate_pos_def_correlation_matrix(N_s, min_corr=0.3, max_corr=0.9, seed=None):
    """
    Generates a random positive definite correlation matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    A = np.random.uniform(min_corr, max_corr, size=(N_s, N_s))
    np.fill_diagonal(A, 1.0)  # Set diagonal to 1 (correlation with itself)
    A = 0.5 * (A + A.T)  # Symmetrize matrix

    # Ensure positive definiteness by adjusting eigenvalues
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, 0.01, None)  # Force all eigenvalues to be non-negative
    A_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T  # Reconstruct matrix

    # Normalize to make it a correlation matrix
    D_inv = 1 / np.sqrt(np.diag(A_pd))
    A_corr = np.diag(D_inv) @ A_pd @ np.diag(D_inv)
    return A_corr


def simulate_directed_graph(N_s, min_corr=0.3, max_corr=0.9, seed=None):
    """
    Generates a random directed graph represented by an adjacency matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    # Fill the adjacency matrix with uniform random values
    A_dir = np.random.uniform(min_corr, max_corr, size=(N_s, N_s))
    np.fill_diagonal(A_dir, 1.0)  # Set diagonal to 1 (correlation with itself)
    return A_dir
