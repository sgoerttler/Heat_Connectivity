import numpy as np
import scipy

from connectivity.utils_graph import calculate_laplacian


def simulate_heat_signal(A_sim, N_t=500, alpha=0.01, sig_internal=1, sig_measurement=1, burn_in=100):
    """
    Generate synthetic time series data based on the heat diffusion model over a graph.
    """

    # Initialize parameters and storage for simulated data
    N_s = A_sim.shape[0]
    X_sim = np.zeros((N_s, N_t + burn_in))
    eps_1 = np.random.normal(0, sig_internal, size=(N_s, N_t + burn_in))
    eps_2 = np.random.normal(0, sig_measurement, size=(N_s, N_t + burn_in))
    L_sim = calculate_laplacian(A_sim)

    # Simulate heat diffusion process step by step over time
    for idx_t in np.arange(N_t + burn_in):
        if idx_t == 0:
            x = np.random.normal(0, 1, size=N_s)
        x = 100 * np.tanh(scipy.linalg.expm(-alpha * L_sim) @ (x + eps_1[:, idx_t]) / 100)
        X_sim[:, idx_t] = x + eps_2[:, idx_t]

    # Discard burn-in period to ensure stationarity
    return X_sim[:, burn_in:]