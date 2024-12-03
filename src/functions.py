import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma


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


def compute_mutual_information(dataset, k):

	"""
	Computes the mutual information among multiple 1D variables based on Grassberger's method.
	
	Parameters:
	    dataset (2D array-like): Data matrix where each row is a sample and each column is a variable.
	    k (int): Number of nearest neighbors to consider for the estimation.
	
    Returns:
        float: The estimated mutual information.
        np.ndarray: Marginal counts for each variable and sample (step 2 intermediate result).
    """
	dataset = np.asarray(dataset)
	n_samples, n_variables = dataset.shape
	
	# Step 1: Find k-nearest neighbors in the joint space
	index_s, distances = find_k_nearest_neighbors(dataset, k)
	epsilon = 2 * distances[:, k-1]  # 2*Distance to the k-th nearest neighbor for each point
	print("Epsilon/2 values (joint space):", epsilon/2)
	
	# Step 2: Initialize the counts for each variable
	marginal_counts = np.zeros((n_variables, n_samples))
	
	
	for var_idx in range(n_variables):
		# Extract the current variable as a 1D array
		marginal_data = dataset[:, var_idx]
		
		# Compute pairwise differences (Euclidean distance in 1D)
		distances_marginal = np.abs(marginal_data[:, None] - marginal_data)
		print(f"\nDistances Marginal for Variable {var_idx + 1}:\n", distances_marginal)
		
		# Count points within epsilon/2 for each point
		marginal_counts[var_idx] = np.maximum(
		    0, np.sum(distances_marginal < epsilon[:, None] / 2, axis=1) - 1
		)
		print(f"\nMarginal Counts for Variable {var_idx + 1}:\n", marginal_counts[var_idx])

		
	# Step 3: Compute the mutual information using Grassberger's formula
	mi = (
	digamma(k)
	- 1 / k
	+ (n_variables - 1) * digamma(n_samples)
	- np.mean(np.sum(digamma(marginal_counts + 1), axis=0))
	)
	return mi, marginal_counts, epsilon


def debug_k_nearest_neighbors(dataset, k):
	from sklearn.metrics import pairwise_distances

	# Calcolo delle distanze pairwise con la metrica Chebyshev
	distances_joint = pairwise_distances(dataset, metric='chebyshev')
	print("Pairwise distances (joint space, Chebyshev):\n", distances_joint)
	
	# Calcolo delle distanze del k-esimo vicino
	epsilon = []
	for i in range(len(dataset)):
		sorted_distances = np.sort(distances_joint[i])
		print(f"Point {i}: sorted distances = {sorted_distances}")
		kth_neighbor_distance = sorted_distances[k]  # k-th neighbor distance
		print(f"Point {i}: k-th neighbor distance = {kth_neighbor_distance}")
		epsilon.append(2 * kth_neighbor_distance)
	
	print("\nEpsilon values (joint space):", epsilon)
	return np.array(epsilon)



if __name__ == "__main__":
	# Define a simple dataset with clear separations
	dataset = np.array([
		[1.0, 2.0, 3.0],
		[2.0, 3.0, 4.0],
		[3.0, 4.0, 5.0],
		[5.0, 6.0, 7.0],
		[1.5, 2.5, 3.5]
		])
		
	# Number of nearest neighbors
	k = 2
	
	
	
	
	# Print the dataset for clarity
	print("Dataset:")
	print(dataset)
	
	# Print epsilon/2 expected values:	
	debug_k_nearest_neighbors(dataset, k)
	
	
	# Compute mutual information and marginal counts
	mi, marginal_counts, epsilon = compute_mutual_information(dataset, k)
	
	#print epsilon/2
	print('\nepsilosn/2 computed')
	print(epsilon/2)
	
	# Print marginal counts from Step 2
	print("\nMarginal Counts (Step 2):")
	print(marginal_counts)
	
	# Print mutual information
	print("\nMutual Information (Grassberger):", mi)
	



	# Verifica con valori attesi
	print("\nExpected Counts (manual check):")
	expected_counts = np.array([
		[1, 1, 1, 2, 2],  # For first variable
		[1, 1, 2, 2, 2],  # For second variable
		[1, 1, 3, 2, 2]   # For third variable
		])
	print(expected_counts)
	
	print("\nComparison (Calculated == Expected):")
	print(np.allclose(marginal_counts, expected_counts))



	
	
	
	
	
	
	
