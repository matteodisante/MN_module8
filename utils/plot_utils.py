import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scipy.stats as stats




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


def plot_density(series1, series2=None, bins=50):
    """
    Generate a density plot for one or two series of data using KDE.

    Parameters:
        series1 (array-like): First data series.
        series2 (array-like, optional): Second data series (default: None).
        bins (int): Number of bins for the histogram (used for reference).
    """
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KDE for series1
    sns.kdeplot(series1, fill=True, label="Density - Series 1", color="blue", ax=ax)
    
    # Optionally plot KDE for series2
    if series2 is not None:
        sns.kdeplot(series2, fill=True, label="Density - Series 2", color="green", ax=ax)
    
    # Overlay histograms (optional, can be removed)
    ax.hist(series1, bins=bins, density=True, alpha=0.3, color="blue", label="Histogram - Series 1")
    if series2 is not None:
        ax.hist(series2, bins=bins, density=True, alpha=0.3, color="green", label="Histogram - Series 2")
    
    # Plot labels and legend
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Density Plot")
    ax.legend()
    
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
    

    
    