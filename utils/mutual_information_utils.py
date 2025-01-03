import sys
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from scipy.special import digamma
from scipy.integrate import quad

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/')))
from decorators import time_it
from io_utils import load_data, save_results
from config_utils import load_config


def find_k_nearest_neighbors(matrix, k):
    """
    Finds the k-nearest neighbors for each point in a dataset based on the max metric.

    Parameters:
        matrix (2D array-like): Input data where each row is a point and each column is a coordinate.
        k (int): Number of nearest neighbors to find for each point.

    Returns:
        indices (2D array): Indices of the k-nearest neighbors for each point.
        distances (2D array): Distances to the k-nearest neighbors for each point.
    """
    # Use sklearn's NearestNeighbors with the Chebyshev (max) metric
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(matrix)
    distances, indices = nbrs.kneighbors(matrix)
    return indices[:, 1:], distances[:, 1:]  # Remove self-neighbor



@time_it
def compute_marginal_counts(matrix, epsilon):
    """
    Computes the marginal counts for a single variable (1D) using NearestNeighbors for efficiency.
    For each sample, counts how many points are within a specified distance threshold (epsilon/2).

    Parameters:
        matrix (2D array-like): Input data for a single variable, reshaped as a column vector of shape (n_samples, 1).
        epsilon (1D array-like): Distance thresholds for each sample, provided as a vector.

    Returns:
        np.ndarray: Array of shape (n_samples,) containing the marginal counts for each sample.
    """

    # Ensure the input matrix is in 2D format
    matrix = matrix.reshape(-1, 1)
    n_samples = matrix.shape[0]
    marginal_counts = np.zeros(n_samples)

    # Initialize NearestNeighbors with a fixed radius (max epsilon)
    max_radius = np.max(epsilon) / 2
    nbrs = NearestNeighbors(radius=max_radius, metric='euclidean', algorithm='ball_tree').fit(matrix)

    # Query neighbors within the maximum radius for all points at once
    distances, _ = nbrs.radius_neighbors(matrix, radius=max_radius)  # MODIFIED: query in batch
    # Compute marginal counts using the provided epsilon for each point
    for i in range(n_samples):
        marginal_counts[i] = np.sum(distances[i] <= epsilon[i] / 2) - 1  # Exclude the point itself
        # MODIFIED: compare with epsilon[i] / 2

    return marginal_counts


def compute_std_corr_matrix(data):
    """
    Compute a matrix with standard deviations on the diagonal and correlations off-diagonal.

    :param data: 2D NumPy array where rows are samples and columns are variables.
    :return: 2D NumPy array with standard deviations on the diagonal and correlations off-diagonal.
    """
    std_devs = np.std(data, axis=0, ddof=1)  # Compute standard deviations
    corr_matrix = np.corrcoef(data, rowvar=False)  # Compute correlation matrix
    std_corr_matrix = corr_matrix.copy()

    # Replace diagonal elements with standard deviations
    np.fill_diagonal(std_corr_matrix, std_devs)

    return std_corr_matrix


# Functions that compute the mi estimate for a single file and for a directory. The only thing that
# changes for the single distribution is the value of the size N (config.json)

def process_file(file_path, k, mi_estimate):
    """
    Process a single file to compute mutual information and the standard deviation-correlation matrix.

    :param file_path: Path to the file.
    :param k: Number of nearest neighbors for mutual information calculation.
    :param mi_estimate: A function to estimate mutual information, should accept (data, k) as arguments.
    :return: Tuple containing the filename, std-corr matrix (as a list), and mutual information.
    """
    try:
        # Load data from the file
        data = load_data(file_path)

        # Compute the standard deviation-correlation matrix
        std_corr_matrix = compute_std_corr_matrix(data).tolist()  # Convert to list for JSON-friendly storage

        # Compute mutual information
        mi = mi_estimate(data, k)

        return os.path.basename(file_path), std_corr_matrix, mi
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return os.path.basename(file_path), None, None


@time_it
def process_directory(input_dir, output_file, k, mi_estimate):
    """
    Process all files in the given directory in parallel, compute mutual information and std-corr matrix for each,
    and save the results to an output file.

    :param input_dir: Directory containing input data files.
    :param output_file: File to save the results.
    :param k: Number of nearest neighbors for mutual information calculation.
    :param mi_estimate: A function to estimate mutual information, should accept (data, k) as arguments.
    """
    if not os.path.exists(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.")
        return

    results = []

    with ProcessPoolExecutor() as executor:
        file_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        process_fn = partial(process_file, k=k, mi_estimate=mi_estimate)  # Partial function to pass `k` and `mi_estimate` to `process_file`
        for result in executor.map(process_fn, file_paths):
            results.append(result)

    # Save results in the format: filename, std-corr matrix, mutual information
    save_results(results, output_file)


def circular_mi_theoretical(a, b, c):
    """
    Calculate the theoretical mutual information for a circular distribution.

    :param a: Inner radius of the circular distribution.
    :param b: Outer radius of the circular distribution.
    :param c: Middle radius where the triangular distribution switches.
    :return: Theoretical mutual information.
    """
    def marginal_density(x):
        if abs(x) < a:
            return 0
        elif a <= abs(x) <= c:
            return (2 / (np.pi * (b - a) * (c - a))) * (
                np.sqrt(c**2 - x**2) - np.sqrt(a**2 - x**2) -
                a * np.log((np.sqrt(c**2 - x**2) + c) / (np.sqrt(a**2 - x**2) + a))
            )
        elif c < abs(x) <= b:
            return (2 / (np.pi * (b - a) * (b - c))) * (
                b * np.log((np.sqrt(b**2 - x**2) + b) / (np.sqrt(c**2 - x**2) + c)) -
                np.sqrt(b**2 - x**2) + np.sqrt(c**2 - x**2)
            )
        else:
            return 0

    def marginal_entropy():
        result, _ = quad(lambda x: -marginal_density(x) * np.log(marginal_density(x) + 1e-12), -b, b)
        return result

    h_x = marginal_entropy()

    h_xy = (
        0.5 + np.log(np.pi * (b - a))
        - c**2 / ((c - a) * (b - c)) * (np.log(c) - 1.5)
        + a**2 / ((b - a) * (c - a)) * (np.log(a) - 1.5)
        + b**2 / ((b - a) * (b - c)) * (np.log(b) - 1.5)
    )

    return 2 * h_x - h_xy


def weinman_exponential_mi_theoretical(u):
    """
    Calculate the exact mutual information for the Weinman exponential distribution.

    :param u: Parameter of the distribution (0 < u < 1 for the valid range).
    :return: Mutual information (I_exact).
    """
    if u < 0.5:
        mi_exact = (
            np.log((2 * u) / (1 - 2 * u)) +
            digamma(1 / (1 - 2 * u)) -
            digamma(1)
        )
    elif u > 0.5:
        mi_exact = (
            np.log((2 * u - 1) / u) +
            digamma(2 * u / (2 * u - 1)) -
            digamma(1)
        )
    else:
        mi_exact = - digamma(1)
    return mi_exact


def gamma_exponential_mi_theoretical(u):
    """
    Calculate the exact mutual information for the Gamma exponential distribution.

    :param u: Shape parameter of the Gamma distribution (u > 0).
    :return: Mutual information (I_exact).
    """
    if u <= 0:
        raise ValueError("The parameter u must be greater than 0.")
    mi_exact = digamma(u + 1) - np.log(u)
    return mi_exact


def compute_mi_theoretical_file(config_file, distribution, output_path, mutual_output_file):
    """
    Compute theoretical mutual information for a single distribution and save the results.

    :param distribution: The distribution for which to calculate theoretical MI.
    :param output_path: Path to the output file.
    :param mutual_output_file: Results from the `process_file` function.
    """
    results = []

    config = load_config(config_file)

    # get the parameters of the selected distribution, if it is in the configuration file
    for dist in config['distributions']:
        dist_name = dist['name']
        if distribution is not dist_name:
            continue #Avoiding non-selected distributions  
        params = dist['params']


    for record in mutual_output_file:
        filename, std_corr_matrix, _ = record
        if std_corr_matrix is None:
            continue

        correlation = np.array(std_corr_matrix).trace()  # Compute theoretical correlation

        # Idependent variables
        if distribution == "uniform":
            mi_theoretical = 0  # Checking only if they are independent
        if distribution == "exponential":
            mi_theoretical = 0  # Checking only if they are independent

        # Two correlated variables
        if distribution == "weinman_exponential":
            mi_theoretical = weinman_exponential_mi_theoretical(params)
        elif distribution == "gamma_exponential":
            mi_theoretical = gamma_exponential_mi_theoretical(params)
        elif distribution == "bivariate_gaussian":
            mi_theoretical = -0.5 * np.log(1 - correlation**2)
        elif distribution == "circular":
            mi_theoretical = circular_mi_theoretical(params)
        else:
            mi_theoretical = "Unsupported distribution"

        results.append({
            "filename": filename,
            "mi_theoretical": mi_theoretical
        })

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Theoretical results saved to: {output_path}")


def compute_mi_theoretical_folder(input_dir, output_path, k, mi_estimate, distribution):
    """
    Compute theoretical MI for all files in a folder in parallel.

    :param input_dir: Directory containing input files.
    :param output_path: Path to the output file to save the results.
    :param k: Number of neighbors for MI calculation.
    :param mi_estimate: Function to estimate MI.
    :param distribution: Distribution for which to calculate theoretical MI.
    """
    file_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    with ProcessPoolExecutor() as executor:
        mutual_output_file = list(executor.map(lambda fp: process_file(fp, k, mi_estimate), file_paths))

    compute_mi_theoretical_file(distribution, output_path, mutual_output_file)


def compute_mi_estimate_error_from_results(processed_results):
    """
    Compute mutual information errors based on the results from `process_directory`.

    :param processed_results: List of results from `process_directory`.
    :return: A list of dictionaries containing filenames, estimated MI, and errors.
    """
    def compute_error(mi_values):
        """
        Compute the error for a list of MI estimates.

        :param mi_values: List of mutual information estimates.
        :return: Standard error of the mean (SEM).
        """
        return np.std(mi_values) / np.sqrt(len(mi_values))

    results = []

    def process_result(result):
        filename = result["filename"]
        mi = result["mutual_information"]
        std_corr_matrix = result["std_corr_matrix"]
        if mi is not None:
            error = compute_error([mi])
            return {"filename": filename, "mi_estimate": mi, "error": error}
        return {"filename": filename, "mi_estimate": None, "error": None}

    # Process results in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_result, processed_results))

    return results


def compare_mi_estimate_with_theoretical(estimates, theoretical_results):
    """
    Compare estimated MI ± error with theoretical MI for each file.

    :param estimates: List of results containing filenames, estimated MI, and errors.
    :param theoretical_results: List of theoretical MI values for each file.
    :return: List of comparisons for each file.
    """
    comparisons = []

    # Compare estimated MI ± error with theoretical MI
    for estimate in estimates:
        filename = estimate["filename"]
        mi_estimate = estimate["mi_estimate"]
        error = estimate["error"]
        theoretical_entry = next((item for item in theoretical_results if item["filename"] == filename), None)

        if theoretical_entry:
            mi_theoretical = theoretical_entry["mi_theoretical"]
            comparison = {
                "filename": filename,
                "mi_estimate": mi_estimate,
                "error": error,
                "mi_theoretical": mi_theoretical,
                "within_error": abs(mi_estimate - mi_theoretical) <= error
            }
            comparisons.append(comparison)

    return comparisons