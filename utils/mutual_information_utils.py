import sys
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from decorators import time_it


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
    nbrs = NearestNeighbors(radius=np.max(epsilon) / 2, metric='euclidean', algorithm='ball_tree').fit(matrix)

    # Query the neighbors within the radius for all points
    for i in range(n_samples):
        distances, _ = nbrs.radius_neighbors(matrix[i].reshape(1, -1), radius=epsilon[i] / 2)
        marginal_counts[i] = len(distances[0]) - 1  # Exclude the point itself

    return marginal_counts    