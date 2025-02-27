import os
import sys
import numpy as np
from scipy.spatial import cKDTree
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_k_nearest_neighbors(matrix, k):
    """
    Finds the k-th nearest neighbor for each point in a dataset using the Chebyshev (max) metric,
    optimized for 2D using scipy.spatial.cKDTree.

    Parameters:
        matrix (2D array-like): Input data where each row is a point.
        k (int): Number of nearest neighbors to consider (excluding the point itself).

    Returns:
        tuple:
            np.ndarray: 1D array containing the distance to the k-th nearest neighbor for each point.
            np.ndarray: 1D array containing the index of the k-th nearest neighbor for each point.
    """
    try:
        logger.info(f"Finding {k} nearest neighbors using Chebyshev metric with cKDTree.")
        # Build the tree from the data
        tree = cKDTree(matrix)
        # p=np.inf indicates Chebyshev distance
        distances, indices = tree.query(matrix, k=k+1, p=np.inf)
        logger.info(f"Found nearest neighbors for {matrix.shape[0]} samples.")
        # Return the distance and index of the k-th neighbor (excluding the self-neighbor)
        return distances[:, k], indices[:, k]
    except Exception as e:
        logger.error(f"Error finding k-nearest neighbors with k={k}: {e}")
        return None, None

def main():
    # Test dataset 1: Simple 2D points
    dataset1 = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [10, 10]
    ])
    
    print("Test dataset 1:")
    print(dataset1)
    
    # Test with different values of k
    for k in [1, 2]:
        kth_dists, kth_indices = find_k_nearest_neighbors(dataset1, k)
        print(f"\nFor k = {k}:")
        print("Distances to k-th nearest neighbor:")
        print(kth_dists)
        print("Indices of k-th nearest neighbor:")
        print(kth_indices)
    
    # Test dataset 2: Random 2D points
    np.random.seed(42)
    dataset2 = np.random.rand(10, 2) * 100
    print("\nTest dataset 2 (random points):")
    print(dataset2)
    
    for k in [1, 3]:
        kth_dists, kth_indices = find_k_nearest_neighbors(dataset2, k)
        print(f"\nFor k = {k}:")
        print("Distances to k-th nearest neighbor:")
        print(kth_dists)
        print("Indices of k-th nearest neighbor:")
        print(kth_indices)

if __name__ == '__main__':
    main()