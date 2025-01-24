import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scipy.stats as stats

from io_utils import ensure_directory_and_handle_file_conflicts, load_data_csv


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
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))

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
    
    
    
    


def generate_plot(
    file_paths, x_col, y_col, yerr_col, title, xlabel, ylabel, output_dir, 
    combine=False, theoretical_value=0, k_or_bins=None, x_transform=None
):
    """
    Generates plots for the provided datasets based on the configuration.

    Parameters:
        file_paths (list of str): Paths to the CSV files.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        yerr_col (str): Column name for error on y-axis.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        output_dir (str): Directory to save the plots.
        combine (bool): Whether to combine multiple curves in one plot.
        theoretical_value (float): Theoretical value to subtract from y_col.
        k_or_bins (list of int): Selected k or bins_number values.
        x_transform (callable): Optional transformation for x-axis values.
    """
    plt.style.use('seaborn-v0_8-darkgrid')

    # Group files by distribution and parameters
    grouped_files = {}
    for file in file_paths:
        distribution_key = '_'.join(os.path.basename(file).split('_')[2:-1])
        grouped_files.setdefault(distribution_key, []).append(file)

    for k_or_bin in k_or_bins:
        plt.figure(figsize=(10, 6))

        for group, files in grouped_files.items():
            curve_x, curve_y, curve_yerr = [], [], []

            for file in files:
                data = load_data_csv(file)
                if not all(col in data.columns for col in [x_col, y_col, yerr_col]):
                    print(f"[WARNING] Skipping file {file}: required columns missing.")
                    continue

                filtered_data = data[data[x_col] == k_or_bin]
                if filtered_data.empty:
                    print(f"[WARNING] Skipping {file}: k_or_bin {k_or_bin} not found in {x_col}.")
                    continue

                size = int(os.path.basename(file).split('_')[-1].replace('.csv', '').replace('size', ''))
                x_value = size if x_transform is None else x_transform(size)
                curve_x.append(x_value)
                curve_y.append(filtered_data[y_col].values[0] - theoretical_value)
                curve_yerr.append(filtered_data[yerr_col].values[0])

            sorted_indices = np.argsort(curve_x)
            curve_x, curve_y, curve_yerr = np.array(curve_x)[sorted_indices], np.array(curve_y)[sorted_indices], np.array(curve_yerr)[sorted_indices]

            plt.errorbar(curve_x, curve_y, yerr=curve_yerr, fmt="o", capsize=4, label=f"{group}")

        plt.legend(loc='upper right', fontsize=9, frameon=True, edgecolor="black")
        plt.title(f"{title} (k = {k_or_bin})", fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        combined_dir = os.path.join(output_dir, "combined", title.replace(' ', '_'), f"k_{k_or_bin}")
        ensure_directory_and_handle_file_conflicts(combined_dir)
        plt.savefig(os.path.join(combined_dir, f"figure_{title.replace(' ', '_')}_k{k_or_bin}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    

    
    