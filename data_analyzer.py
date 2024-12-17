import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm
from mpl_toolkits.mplot3d import Axes3D

# Import utility modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))
from plot_utils import plot_histograms, plot_density, plot_3d_histogram, plot_marginals 
from math_utils import log_likelihood, calculate_statistics
from io_utils import ensure_directory, save_data, load_data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))
# Utility functions for file management and configuration loading
from pre_processing_utils import sample_data



def analyze_univariate_data(series1, series2):
    """
    Analyze and plot univariate data: statistics, correlation, histograms, and log-likelihood.

    Parameters:
        series1 (np.ndarray): First univariate series.
        series2 (np.ndarray): Second univariate series.
    """
    bins = 100
    
    # Statistics and correlation
    stats1 = calculate_statistics(series1)
    stats2 = calculate_statistics(series2)
    correlation = pearsonr(series1, series2)[0]
    
    print(f"Statistics for Series 1: {stats1}")
    print(f"Statistics for Series 2: {stats2}")
    print(f"Correlation between series: {correlation:.7f}")
    
        
    # Save empirical statistics and correlation
    #save_data(np.array([[correlation]]), os.path.join(save_dir, "correlation.csv"))
    
    
    # Sample the data
    sample_size = 300000  # Adjust this number as needed
    series1_sample, series2_sample = sample_data(series1, series2, sample_size)
    
    # Plotting using sampled data
    plot_histograms(series1_sample, series2_sample, bins)
    plot_density(series1_sample, series2_sample, bins)
    
    
def analyze_multivariate_data(data, target_function=None):
    """
    Analyze and plot multivariate data: correlation, 2D histogram, scatter plot, and marginals.

    Parameters:
        data (np.ndarray): Input data with two columns (X, Y).
        target_function (callable, optional): Target PDF for theoretical comparisons.
    """
    series1, series2 = data[:, 0], data[:, 1]
    bins = 100
    
    # Compute correlation
    correlation = np.corrcoef(series1, series2)[0, 1]
    print(f"Correlation between series: {correlation:.7f}")
    
    # 2D Histogram
    hist, xedges, yedges = np.histogram2d(series1, series2, bins=bins, density=True)
    
    # Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(series1, series2, s=1, alpha=0.5)
    plt.title("Scatter Plot of Multivariate Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show(block=True)
    
    # 3D Histogram
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = np.ones_like(zpos) * (xedges[1] - xedges[0])
    dz = hist.ravel()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
    ax.set_title("3D Histogram of Multivariate Data")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Density")
    plt.show(block=True)
    
    # Marginals
    marginal_x = hist.sum(axis=0) * (yedges[1] - yedges[0])
    marginal_y = hist.sum(axis=1) * (xedges[1] - xedges[0])
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(xedges[:-1], marginal_x, label="Marginal X")
    plt.title("Marginal Distribution of X")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(yedges[:-1], marginal_y, label="Marginal Y")
    plt.title("Marginal Distribution of Y")
    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show(block=True)
    
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(xedges[:-1], marginal_x, label="Marginal X")
    plt.title("Marginal Distribution of X")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(yedges[:-1], marginal_y, label="Marginal Y")
    plt.title("Marginal Distribution of Y")
    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.yscale('log')
    plt.grid(True)
    plt.show(block=True)
    
    
    
    
    # Theoretical distribution comparison
    if target_function:
        x = np.linspace(xedges.min(), xedges.max(), 300)
        y = np.linspace(yedges.min(), yedges.max(), 300)
        xgrid, ygrid = np.meshgrid(x, y)
        density = target_function(xgrid, ygrid)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xgrid, ygrid, density, cmap="viridis", edgecolor="none", alpha=0.8)
        ax.set_title("Theoretical PDF Surface")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Density")
        plt.show(block=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze univariate or multivariate data from a file.")
    parser.add_argument("file_path", type=str, help="Path to the input data file (CSV format).")
    parser.add_argument("--target_function", type=str, help="Name of the theoretical target function (optional).")
    args = parser.parse_args()

    # Load data
    data = load_data(args.file_path)

    # Check for valid data format
    if data.shape[1] != 2:
        raise ValueError("The dataset must have exactly 2 columns for analysis.")

    # Ask user for the type of analysis
    while True:
        print("\nIs this dataset bivariate or correlated univariate?")
        print("1. Bivariate")
        print("2. Correlated Univariate")
        user_input = input("Enter 1 or 2: ").strip()
        if user_input == "1":
            print("Running bivariate analysis...")
            analyze_multivariate_data(data)
            break
        elif user_input == "2":
            print("Running correlated univariate analysis...")
            series1, series2 = data[:, 0], data[:, 1]
            analyze_univariate_data(series1, series2)
            break
        else:
            print("Invalid input. Please enter 1 or 2.")