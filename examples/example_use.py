from simulate_graphs import simulate_pos_def_correlation_matrix, simulate_directed_graph
from simulate_data import simulate_heat_signal
from plot_results import plot_matrices

from connectivity.heat_connectivity import retrieve_heat_graph

"""
Example usage of the heat connectivity retrieval function with simulated data.
Several estimation configurations and regularisation settings are compared for both undirected and directed graphs.
"""

if __name__ == '__main__':
    # ========== UNDIRECTED GRAPH EXAMPLE ========== #
    # Simulate data
    A_sim = simulate_pos_def_correlation_matrix(N_s=10, min_corr=0.3, max_corr=0.9, seed=42)
    X_data = simulate_heat_signal(A_sim, N_t=20000, alpha=0.05, sig_internal=1, sig_measurement=1, burn_in=100)

    # Compute heat connectivity for different configurations of estimation methods and regularisation
    A_EV = retrieve_heat_graph(X_data, estimation_type='equal_variance', regularisation=False)
    A_2 = retrieve_heat_graph(X_data, estimation_type='2nd_order', regularisation=False)
    A_2R = retrieve_heat_graph(X_data, estimation_type='2nd_order', regularisation=True)

    # Plot results
    A_conns = [A_sim, A_EV, A_2, A_2R]
    A_conn_names = ['Simulated', 'Equal Variance', '2nd Order', '2nd Order Regularised']
    plot_matrices('Heat Connectivity Retrieval $-$ Undirected Graph', A_conns, A_conn_names)

    # ========== DIRECTED GRAPH EXAMPLE ========== #
    # Simulate data
    A_sim = simulate_directed_graph(N_s=10, min_corr=0.3, max_corr=0.9, seed=43)
    X_data = simulate_heat_signal(A_sim, N_t=20000, alpha=0.05, sig_internal=1, sig_measurement=1, burn_in=100)
    
    # Compute connectivity of directed graph by not imposing Laplacian constraints
    A_EV = retrieve_heat_graph(X_data, estimation_type='equal_variance', regularisation=False, include_L_constraints=False)
    A_2 = retrieve_heat_graph(X_data, estimation_type='2nd_order', regularisation=False, include_L_constraints=False)
    A_2R = retrieve_heat_graph(X_data, estimation_type='2nd_order', regularisation=True, include_L_constraints=False)

    # Plot results
    A_conns = [A_sim, A_EV, A_2, A_2R]
    A_conn_names = ['Simulated', 'Equal Variance', '2nd Order', '2nd Order Regularised']
    plot_matrices('Heat Connectivity Retrieval $-$ Directed Graph', A_conns, A_conn_names)
