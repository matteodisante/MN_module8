import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

from utils.decorators import time_it


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
    try:
        logging.info(f"Finding {k} nearest neighbors using Chebyshev metric.")

        # Use sklearn's NearestNeighbors with the Chebyshev (max) metric
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(matrix)
        distances, indices = nbrs.kneighbors(matrix)

        # Exclude self-neighbor and return
        logging.info(f"Found nearest neighbors for {matrix.shape[0]} samples.")
        return indices[:, 1:], distances[:, 1:]  # Remove self-neighbor
    except Exception as e:
        logging.error(f"Error finding k-nearest neighbors with k={k}: {e}")
        return None, None




def compute_marginal_counts(matrix, epsilon):
    """
    Computes the marginal counts for a single variable (1D) using NearestNeighbors for efficiency.
    For each sample, counts how many points are within a specified distance threshold (epsilon/2).

    Parameters:
        matrix (1D array-like): Input data for a single variable, reshaped as a column vector of shape (n_samples, 1).
        epsilon (1D array-like): Distance thresholds for each sample, provided as a vector.

    Returns:
        np.ndarray: Array of shape (n_samples,) containing the marginal counts for each sample.
    """
    try:
        logging.info(f"Computing marginal counts with epsilon values: {epsilon}")

        # Ensure the input matrix is in (n_samples, 1) format
        matrix = matrix.reshape(-1, 1)
        n_samples = matrix.shape[0]
        marginal_counts = np.zeros(n_samples)

        # Initialize NearestNeighbors 
        nbrs = NearestNeighbors(metric='euclidean', algorithm='ball_tree').fit(matrix)

        # Query the neighbors within the radius for all points 
        # nbrs.radius_neighbors doens't count the central point
        # nbrs.radius_neighbors actually counts the point on the edge but substracting 1e-12 to radius able us to ignore the counting of the border point
        for i in range(n_samples):
            distances, _ = nbrs.radius_neighbors(matrix[i].reshape(1, -1), radius = (epsilon[i]/2 - 1e-12) ) 
            marginal_counts[i] = len(distances[0]) 

        logging.info(f"Marginal counts computed for {n_samples} samples.")
        return marginal_counts
    except Exception as e:
        logging.error(f"Error computing marginal counts: {e}")
        return None
