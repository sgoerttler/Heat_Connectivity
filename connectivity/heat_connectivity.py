import numpy as np
import scipy.linalg as la

from connectivity.utils_graph import calculate_adjacency_matrix
from connectivity.utils_heat import matrix_sq, matrix_sq_diag, impose_noise_sq_constraints, impose_laplacian_constraints


def retrieve_heat_graph(X_data, estimation_type='2nd_order', regularisation=True, include_noise_constraints=True,
                        include_L_constraints=True, return_matrix='adjacency', input_shape='spatial_first'):
    """
    Estimate a heat-diffusion graph (adjacency or Laplacian matrix) from multivariate data [1].

    Parameters
    ----------
    X_data : np.ndarray
        Input data array. If `input_shape='spatial_first'`, data is assumed to be
        shaped as (n_nodes, n_timepoints). If `input_shape='temporal_first'`,
        data is interpreted as (n_timepoints, n_nodes).

    estimation_type : {'equal_variance', '2nd_order'}, default='2nd_order'
        Specifies the estimation type used to approximate unknown noise sources.

    regularisation : bool, default=True
        Whether to apply regularisation by adding a scaled identity matrix to reduce the condition number
        and improve numerical stability. While this enhances the structural reliability of the estimated graph,
        it reduces the accuracy of the absolute scale, namely the graph thermal diffusivity.

    include_noise_constraints : bool, default=True
        If True, includes noise-related constraints or priors in the optimisation.

    include_L_constraints : bool, default=True
        If True, enforces structural Laplacian constraints such as symmetry and zero row sums.

    return_matrix : {'adjacency', 'laplacian'}, default='adjacency'
        Determines which matrix representation of the learned graph is returned.
        - 'adjacency' : Return the estimated adjacency matrix.
        - 'laplacian' : Return the corresponding graph Laplacian.

    input_shape : {'spatial_first', 'temporal_first'}, default='spatial_first'
        Specifies the shape of the input data array.

    Returns
    -------
    np.ndarray
        The estimated graph matrix (adjacency or Laplacian), shaped (n_nodes, n_nodes).
    References
    ----------
    [1] Goerttler, S., He, F., & Wu, M. (2024, July). Stochastic Graph Heat Modelling for Diffusion-based Connectivity
    Retrieval. In 2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society
    (EMBC) (pp. 1-4). IEEE.
    """

    # Transpose data if required
    if input_shape == 'temporal_first':
        X_data = X_data.T

    # Shorthand notation for simplification
    X_0 = X_data[:, :-1]
    X_1 = X_data[:, 1:]

    # Compute M_0_sq based on estimation type
    if estimation_type == 'equal_variance':
        E_sq = 1 / 3 * ((X_1 - X_0) @ (X_1 - X_0).T)
        if include_noise_constraints:
            E_sq = impose_noise_sq_constraints(E_sq, avg=True)

        M_0_sq = X_0 @ X_0.T + 2 * E_sq

    elif estimation_type == '2nd_order':
        # Shorthand notation for simplification
        E_1 = X_data[:, 1:] - X_data[:, :-1]
        E_2 = X_data[:, 2:] - X_data[:, :-2]

        if include_noise_constraints:
            # Compute only diagonal elements of matrix square which follows constraint that noise is uncorrelated
            # between nodes
            E_int_sq_plus_E_ext_sq = matrix_sq_diag(E_2) / 2
            E_int_sq = matrix_sq_diag(E_2) - matrix_sq_diag(E_1)
            # Clip to ensure E_int_sq is non-negative and less than E_int_sq + E_ext_sq
            E_int_sq = np.clip(E_int_sq, 0, E_int_sq_plus_E_ext_sq)
        else:
            E_int_sq_plus_E_ext_sq = matrix_sq(E_2) / 2
            E_int_sq = matrix_sq(E_2) - matrix_sq(E_1)

        M_0_sq = X_0 @ X_0.T + E_int_sq_plus_E_ext_sq

    else:
        raise ValueError('Variable estimation_type has to be equal_variance or 2nd_order!')

    # Compute inverse of M_0_sq
    if regularisation:
        evalues, evectors = np.linalg.eig(M_0_sq)
        gamma_factor = np.linalg.cond(M_0_sq) ** 0.5
        gamma = -(gamma_factor * min(evalues) - max(evalues)) / (gamma_factor - 1)
        correction_factor = (gamma + np.mean(evalues)) / gamma
        iden = np.eye(X_data.shape[0])
        M_0_sq_inv = np.linalg.inv(M_0_sq + gamma * iden) * correction_factor
    else:
        M_0_sq_inv = np.linalg.inv(M_0_sq)

    # Final computation of heat Laplacian
    if estimation_type == 'equal_variance':
        L_heat = -1 * la.logm((X_1 @ X_0.T + E_sq) @ M_0_sq_inv)
    else:
        # 2nd order
        L_heat = -1 * la.logm((X_1 @ X_0.T + E_int_sq) @ M_0_sq_inv)

    # Impose constraints on Laplacian if required and return requested matrix
    if include_L_constraints:
        return impose_laplacian_constraints(L=L_heat, return_matrix=return_matrix)

    # Return requested matrix
    if return_matrix == 'Laplacian':
        return L_heat
    elif return_matrix == 'adjacency':
        return calculate_adjacency_matrix(L_heat)
