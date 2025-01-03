import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/')))
from mutual_information_utils import process_directory, compute_mi_theoretical_folder, compute_mi_theoretical_file


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



# Mutual information plots:

def plot_scaling_with_kN(k_values, N_values, mi_estimate_function, mutual_info_values_errors, exact_value):
    """
    Plot the scaling of mutual information estimates with k/N or (k-0.5)/N based on the estimation function.

    :param k_values: List of k values.
    :param N_values: List of N values.
    :param mi_estimate_function: Function used to estimate MI.
    :param mutual_info_values_errors: 3D array of mutual information estimates and errors [2, len(k_values), len(N_values)].
                                    mutual_info_values_errors[0] contains the MI estimates,
                                    mutual_info_values_errors[1] contains the associated errors.
    :param exact_value: Exact mutual information value.
    """
    mi_values = mutual_info_values_errors[0]  # Mutual information estimates
    mi_errors = mutual_info_values_errors[1]  # Associated errors

    plt.figure(figsize=(8, 6))

    for i, N in enumerate(N_values):
        if mi_estimate_function.__name__ == "mutual_information_2":
            x_values = (k_values - 0.5) / N
        else:
            x_values = k_values / N

        plt.errorbar(x_values, mi_values[:, i], yerr=mi_errors[:, i], label=f"N = {N}", fmt='o', capsize=3)

    method_suffix = mi_estimate_function.__name__.split("_")[-1]  # Extract the last part of the function name
    plt.axhline(y=exact_value, color="r", linestyle="--", label="Exact Value")
    plt.xlabel("$(k-0.5)/N$" if mi_estimate_function.__name__ == "mutual_information_2" else "$k/N$")
    plt.ylabel(f"Mutual Information Estimate $I_{{{method_suffix}}}$")
    plt.title("Scaling of Mutual Information with $k/N$")
    plt.legend()
    plt.grid()
    plt.show()


def plot_standard_deviations(k_values, N_values, mi_estimate_function, std_devs):
    """
    Plot standard deviations of the mutual information estimates.

    :param k_values: List of k values.
    :param N_values: List of N values.
    :param mi_estimate_function: Function or tuple of functions used to estimate MI.
    :param std_devs: 3D array of standard deviations [len(functions), len(k_values), len(N_values)]
                      if multiple functions are provided, otherwise 2D [len(k_values), len(N_values)].
    """
    plt.figure(figsize=(8, 6))

    # Handle single or multiple MI estimation functions
    if not isinstance(mi_estimate_function, tuple):
        mi_estimate_function = (mi_estimate_function,)

    for func_idx, func in enumerate(mi_estimate_function):
        method_suffix = func.__name__.split("_")[-1]  # Extract the last part of the function name

        for i, N in enumerate(N_values):
            if func.__name__ == "mutual_information_2":
                x_values = (k_values - 0.5) / N
            else:
                x_values = k_values / N

            plt.plot(x_values, std_devs[func_idx, :, i] * np.sqrt(N),
                     label=f"N = {N}, $I_{{{method_suffix}}}$")

    plt.xlabel("$(k-0.5)/N$" if "mutual_information_2" in [f.__name__ for f in mi_estimate_function] else "$k/N$")
    plt.ylabel("Standard Deviation $\times \sqrt{N}$")
    plt.title("Standard Deviations of Mutual Information Estimates")
    plt.legend()
    plt.grid()
    plt.show()


def plot_systematic_errors(N_values, mi_estimate_function, mutual_info_values_errors, exact_value):
    """
    Plot systematic errors of the mutual information estimates with error bars.

    :param N_values: List of N values.
    :param mi_estimate_function: Function used to estimate MI.
    :param mutual_info_values_errors: 3D array of mutual information estimates and errors [2, len(k_values), len(N_values)].
                                      mutual_info_values_errors[0] contains the MI estimates,
                                      mutual_info_values_errors[1] contains the associated errors.
    :param exact_value: Exact mutual information value.
    """
    plt.figure(figsize=(8, 6))

    # Extract mutual information estimates and errors
    mi_values = mutual_info_values_errors[0]  # MI estimates (shape: [len(k_values), len(N_values)])
    mi_errors = mutual_info_values_errors[1]  # MI errors (shape: [len(k_values), len(N_values)])

    # Calculate systematic errors (mean MI estimated - MI exact)
    systematic_errors = np.mean(mi_values, axis=0) - exact_value
    systematic_error_bars = np.sqrt(np.sum(mi_errors**2, axis=0)) / mi_values.shape[0]  # Combine errors

    # Plot the systematic errors with error bars
    plt.errorbar(N_values, systematic_errors, yerr=systematic_error_bars, fmt="o", label="Systematic Errors", capsize=5)

    # Plot the reference lines proportional to N^(-0.5) and N^(-0.85)
    N_values = np.array(N_values)
    reference_curve_1 = systematic_errors[0] * (N_values / N_values[0])**-0.5
    reference_curve_2 = systematic_errors[0] * (N_values / N_values[0])**-0.85

    plt.loglog(N_values, reference_curve_1, linestyle="--", label="$N^{-0.5}$", color="r")
    plt.loglog(N_values, reference_curve_2, linestyle="--", label="$N^{-0.85}$", color="g")

    # Add labels, title, and legend
    method_suffix = mi_estimate_function.__name__.split("_")[-1]  # Extract the last part of the function name
    plt.xlabel("N")
    plt.ylabel(f"Systematic Error ($I_{{{method_suffix}}}$)")
    plt.title("Systematic Errors vs N")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


def plot_ratio_estimated_to_exact(N_values, mi_estimate_function, mutual_info_values_errors, Iexact_values, r_values):
    """
    Plot ratios of estimated MI to exact MI for different correlation values as a function of 1/N.
    
    :param N_values: List of N values.
    :param mi_estimate_function: Function used to calculate the estimated MI for different N and r values.
    :param mutual_info_values_errors: 3D array of mutual information estimates and errors [2, len(k_values), len(N_values)].
                                      mutual_info_values_errors[0] contains the MI estimates,
                                      mutual_info_values_errors[1] contains the associated errors.
    :param Iexact_values: 1D array of exact MI values corresponding to each r in r_values.
    :param r_values: List of correlation values.
    """
    # Extract the last part of the mi_estimate_function's name (e.g., 'mutual_information_2' -> '2')
    function_name = mi_estimate_function.__name__
    pedice = function_name.split('_')[-1]
    
    # Calculate 1/N values
    one_over_N_values = 1 / np.array(N_values, dtype=float)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    
    for i, r in enumerate(r_values):
        # Estimating MI for current r_value over all N_values
        mi_estimates = mutual_info_values_errors[0][i]  # Extract MI estimates for current r_value
        mi_errors = mutual_info_values_errors[1][i]  # Extract errors for current r_value
        
        # Get the exact value for the current r from Iexact_values
        Iexact = Iexact_values[i]
        
        # Calculate the ratio of estimated MI to exact MI
        ratios = mi_estimates / Iexact
        
        # Plot the ratios with error bars
        plt.errorbar(one_over_N_values, ratios, yerr=mi_errors / Iexact, label=f"r = {r}", capsize=5)
    
    # Use the extracted 'pedice' to label the y-axis
    plt.xlabel("$1/N$")
    plt.ylabel(f"$I_{pedice}$ / Exact MI")
    plt.title(f"Ratios of $I_{pedice}$ to Exact MI vs $1/N$")
    plt.legend()
    plt.grid()
    plt.show()