import os
import sys
import numpy as np
from scipy.special import digamma


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from core_utils import find_k_nearest_neighbors, kth_nearest_distance_1d




def mutual_information_1_entropies_sum(dataset, k):
    """
    Computes the mutual information among multiple 1D variables based on Grassberger's method,
    using a fully vectorized approach for the marginal (1D) spaces.
    
    The procedure is as follows:
      1. In the joint space (multidimensional) we compute the k-th nearest neighbor distances
         using the function find_k_nearest_neighbors.
      2. For each variable (1D), we compute the k-th nearest neighbor distances using a 
         vectorized approach (via kth_nearest_distance_1d).
      3. For each variable, the marginal entropy is estimated as the mean of the logarithm of 
         2 * (k-th nearest neighbor distance), with a floor to avoid log(0).
      4. Finally, the mutual information is computed using Grassberger's formula.
    
    Parameters:
        dataset (2D array-like): Data matrix where each row is a sample and each column a variable.
        k (int): Number of nearest neighbors to consider.
    
    Returns:
        float: The estimated mutual information.
    """
    dataset = np.asarray(dataset)
    n_samples, n_variables = dataset.shape

    # Step 1: Compute the joint-space k-th nearest neighbor distances
    distances_joint = find_k_nearest_neighbors(dataset, k)
    if distances_joint is None:
        raise ValueError("Error computing joint k-nearest neighbors.")
    epsilon_joint = 2 * distances_joint  # 2 * distance for each sample in the joint space

    # Step 2: For each variable (1D), compute the marginal k-th nearest neighbor distances 
    # using the vectorized approach
    entropy_marginal_means = np.empty(n_variables)
    for var_idx in range(n_variables):
        marginal_data = dataset[:, var_idx]
        kth_dists = kth_nearest_distance_1d(marginal_data, k)
        epsilon_marginal = 2 * kth_dists
        entropy_marginal_means[var_idx] = np.mean(np.log(np.maximum(epsilon_marginal, 1e-10)))
    
    # Step 3: Compute the mutual information using Grassberger's formula
    mi = ((n_variables - 1) * (digamma(n_samples) - digamma(k))
          + np.sum(entropy_marginal_means)
          - n_variables * np.mean(np.log(np.maximum(epsilon_joint, 1e-10))))
    
    return mi
