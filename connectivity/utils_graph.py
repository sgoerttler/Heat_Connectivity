import numpy as np


def calculate_laplacian(A):
    return calculate_diagonal_matrix(A) - A


def calculate_diagonal_matrix(A):
    # In-degree matrix for directed graphs
    return np.diag(np.sum(A, axis=0))


def calculate_adjacency_matrix(L, return_D=False):
    if np.iscomplexobj(L):
        L = L.real
    D = np.zeros(L.shape)
    A = np.zeros(L.shape)
    D[np.eye(L.shape[0]) == 1] = L[np.eye(L.shape[0]) == 1]
    A[np.eye(L.shape[0]) == 0] = -L[np.eye(L.shape[0]) == 0]

    if return_D:
        return A, D
    else:
        return A


def normalize_graph_weights(L=None, A=None):
    # Normalize graph weights to have zero mean and unit standard deviation, excluding self-connections
    if L is not None:
        return_matrix = 'Laplacian'
        A, D = calculate_adjacency_matrix(L, return_D=True)
    else:
        return_matrix = 'adjacency'

    weights = A[np.eye(A.shape[0]) == 0]

    if return_matrix == 'Laplacian':
        return (L - np.nanmean(weights)) / np.nanstd(weights)
    else:
        return (A - np.nanmean(weights)) / np.nanstd(weights)
