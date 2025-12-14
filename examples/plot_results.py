import numpy as np
import matplotlib.pyplot as plt

from connectivity.utils_graph import normalize_graph_weights


def plot_matrices(figure_title, A_conns, A_conn_names):
    # Prepare matrices for visualization
    for A_conn in A_conns:
        A_conn[np.eye(A_conn.shape[0]) == 1] = np.nan  # Exclude diagonal for visualization
    A_sim_norm = normalize_graph_weights(A=A_conns[0])

    images = []
    titles = []
    for mode in ['norm', 'distance']:
        for idx, (A_conn, name) in enumerate(zip(A_conns, A_conn_names)):
            A_conn_norm = normalize_graph_weights(A=A_conn)
            if mode == 'norm':
                images.append(A_conn_norm)
                titles.append(f'${["A_{sim}", "A_{heat}"][idx > 0]}$ ({name})')
            elif mode == 'distance':
                images.append(np.absolute(A_sim_norm - A_conn_norm))
                titles.append(f'$|A_{{sim}} - {["A_{sim}", "A_{heat}"][idx > 0]}$| ({name})')

    # Plotting
    fig, axes = plt.subplots(2, len(A_conns), figsize=(15, 6))
    axes = axes.flatten()

    cmap = [plt.get_cmap('RdYlBu').copy()]
    cmap[0].set_bad(color='black')
    cmap.append(plt.get_cmap('YlGnBu').copy())
    cmap[1].set_bad(color='black')

    for idx, (ax, img, title) in enumerate(zip(axes, images, titles)):
        im = ax.imshow(img, cmap=cmap[idx >= len(A_conns)], vmin=[-4, 0][idx >= len(A_conns)], vmax=[4, 2][idx >= len(A_conns)])
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.suptitle(figure_title)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()