import os
import sys
import numpy as np
from joblib import Parallel, delayed
from scipy.special import digamma



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from decorators import time_it

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from mutual_information_utils import * 


@time_it
def mutual_information_2(dataset, k):
    """
    Estimates the mutual information between two variables using the second algorithm
    by Kraskov, Stögbauer, and Grassberger with modified marginal distance calculation.

    Parameters:
        dataset (np.ndarray): 2D array with samples (rows) and variables (columns).
        k (int): Number of nearest neighbors to consider for the estimation.

    Returns:
        float: The estimated mutual information using the second algorithm.
    """
    dataset = np.asarray(dataset)
    n_samples, n_variables = dataset.shape

    # Step 1: Check if dataset has at least 2 variables
    if n_variables < 2:
        raise ValueError("This algorithm is designed for datasets with at least two variables.")

    # Step 2: Find the k-nearest neighbors in the joint space using the \infty norm (Chebyshev)
    _, distances = find_k_nearest_neighbors(dataset, k)
    epsilon = 2 * distances[:, k-1]  # 2*Distance to the k-th nearest neighbor for each point
    print("Epsilon/2 values (joint space):", epsilon/2)

    # Step 3: Compute marginal distances with the modified approach:
    # - for each point identify points within the square of side epsilon centered at the current point
    # - compute max distances for the marginal variables
    marginal_epsilons = []

    for dim in range(n_variables):
        marginal_data = dataset[:, dim] # takes only the samples of that variable
        marginal_epsilon = np.zeros(n_samples)

        for i in range(n_samples):
            # Identify points within the square of side epsilon centered at the current point
            in_square = np.all(np.abs(dataset - dataset[i]) <= epsilon[i], axis=1)

            # Compute max distances for the marginal variable
            max_distance = np.max(np.abs(marginal_data[in_square] - marginal_data[i]))
            marginal_epsilon[i] = max_distance

        marginal_epsilons.append(marginal_epsilon)

    marginal_epsilons = np.array(marginal_epsilons)

    # Step 4: Compute mutual information using Kraskov's formula (algorithm 2)
    mi = digamma(k) - (1 / k)*(n_variables -1) + (n_variables -1)*digamma(n_samples)

    # Sum of marginal entropies for each variable
    for dim in range(n_variables):
        mi -= np.mean(digamma(np.sum(marginal_epsilons[dim] <= epsilon[:, None], axis=1)))

    # Subtract the joint entropy
    mi += digamma(n_samples)

    return mi