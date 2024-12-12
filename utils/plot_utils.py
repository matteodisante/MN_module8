import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def plot_histograms(series1, series2, bins):
    """Disegna istogrammi sovrapposti per due serie temporali."""
    plt.figure(figsize=(10, 5))
    plt.hist(series1, bins=bins, alpha=0.5, label='Serie 1', color='blue', density=True)
    plt.hist(series2, bins=bins, alpha=0.5, label='Serie 2', color='green', density=True)
    plt.title('Istogrammi Sovrapposti')
    plt.xlabel('Valore')
    plt.ylabel('Densità')
    plt.legend()
    plt.show()

def plot_density(series1, series2, bins, theoretical_pdf):
    """Disegna le curve di densità sovrapposte con la funzione teorica."""
    fig, ax = plt.subplots(figsize=(10, 5))
    counts1, bin_edges1 = np.histogram(series1, bins=bins, density=True)
    counts2, bin_edges2 = np.histogram(series2, bins=bins, density=True)
    bin_centers1 = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])
    bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])
    
    ax.plot(bin_centers1, counts1, label='Densità Serie 1', color='blue')
    ax.plot(bin_centers2, counts2, label='Densità Serie 2', color='green')
    x = np.linspace(min(np.min(series1), np.min(series2)), max(np.max(series1), np.max(series2)), 3000)
    y = theoretical_pdf(x)
    ax.plot(x, y, 'r--', label='Funzione Teorica')
    
    ax.set_xlabel('Valore')
    ax.set_ylabel('Densità')
    ax.legend()
    plt.title('Curve di Densità con Funzione Teorica')
    plt.show()
    
    
    

def plot_3d_histogram(samples, title, output_path, bins=30):
    """Plotta un istogramma 3D con densità."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, density=True)
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    ax.bar3d(xpos.ravel(), ypos.ravel(), 0, 0.1, 0.1, hist.ravel(), shade=True)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


def plot_marginals(samples, output_path):
    """Plotta le marginali standard e in scala log-log."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(samples[:, 0], bins=50, density=True)
    axes[0, 0].set_title("Marginale X")

    axes[0, 1].hist(samples[:, 1], bins=50, density=True)
    axes[0, 1].set_title("Marginale Y")

    axes[1, 0].hist(samples[:, 0], bins=50, density=True)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("Marginale X (semilogy)")

    axes[1, 1].hist(samples[:, 1], bins=50, density=True)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title("Marginale Y (log-log)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()