import numpy as np
from connectivity.utils_graph import calculate_laplacian, calculate_adjacency_matrix


def impose_noise_sq_constraints(A, avg=False):
    # Impose constraints on noise matrix square before optional average
    iden = np.eye(A.shape[0])
    diag = iden == 1
    A[~diag] = 0
    A[diag] = np.clip(A[diag], 0, None)
    if avg:
        return np.mean(A[diag]) * iden
    else:
        return A


def impose_laplacian_constraints(L=None, A=None, D=None, return_matrix=None):
    # Impose symmetry and diagonal elements constraints on Laplacian matrix
    if L is not None:
        return_matrix = return_matrix or 'Laplacian'
        A, D = calculate_adjacency_matrix(L, return_D=True)
    elif A is not None and D is not None:
        return_matrix = return_matrix or 'adjacency'
    else:
        raise ValueError("Either L or both A and D must be provided.")

    A_sym = (A + A.T) / 2
    A_sym[A_sym < 0] = 1e-12
    D1_diag = D[np.eye(L.shape[0]) == 1]
    D2_diag = A_sym @ np.ones(L.shape[1])
    D_sqrt = np.diag(np.sqrt(np.clip(D1_diag + D2_diag, a_min=1e-12, a_max=None) / (2 * D2_diag)))
    A_sym = D_sqrt @ A_sym @ D_sqrt

    if return_matrix == 'Laplacian':
        return calculate_laplacian(A_sym)
    elif return_matrix == 'adjacency':
        return A_sym


def matrix_sq(A):
    # Matrix square computation
    return A @ A.T


def matrix_sq_diag(A, avg=False):
    # Matrix square computation, limited to diagonal elements, which allows pointwise computations
    if avg:
        return np.mean(np.sum(A ** 2, axis=1)) * np.eye(A.shape[0])
    else:
        return np.diag(np.sum(A ** 2, axis=1))

