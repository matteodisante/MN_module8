import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma

def add_noise(matrix,  magnitude = 1e-14, seed = None):
	# Crea un generatore di numeri casuali
	rng = np.random.default_rng(seed = seed)
	
	# Genera un array di rumore uniforme
	noise_matrix = rng.uniform(low=1*magnitude, high=5*magnitude, size=matrix.shape)
	
	noisy_matrix = matrix + noise_matrix
	return noisy_matrix



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


def mutual_information_1(dataset, k):

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
	+ (n_variables - 1) * digamma(n_samples)
	- np.mean(np.sum(digamma(marginal_counts + 1), axis=0))
	)

	
	return mi




def mutual_information_1_wrong(dataset, k):

	"""
	Computes the mutual information among multiple 1D variables based on Grassberger's method.
	
	Parameters:
	    dataset (2D array-like): Data matrix where each row is a sample and each column is a variable.
	    k (int): Number of nearest neighbors to consider for the estimation.
	
    Returns:
        float: The estimated mutual information.
    """
	dataset = np.asarray(dataset)
	n_samples, n_variables = dataset.shape
	
	# Step 1: Given k find the distance from each point to its k-NN in the joint space
	index_s_joint, distances_joint = find_k_nearest_neighbors(dataset, k)
	epsilon_joint = 2 * distances_joint[:, k-1]  # 2*Distance in the joint space to the k-th nearest neighbor for each point
	print("Epsilon/2 values (joint space):", epsilon_joint/2)
	
	
	epsilon_marginal_v = np.zeros(dataset.shape)
	entropy_marginal_means = np.zeros(n_variables)
	
	for var_idx in range(n_variables):
		# Extract the current variable as a 1D array
		marginal_data = dataset[:, var_idx]
		_, distances_marginal = find_k_nearest_neighbors(marginal_data.reshape(-1, 1), k)
		print(f"\nDistances Marginal for Variable {var_idx + 1}:\n", distances_marginal)
		epsilon_marginal_v[:, var_idx] = 2 * distances_marginal[:, k-1]
		entropy_marginal_means[var_idx] = np.mean(np.log(epsilon_marginal_v[:, var_idx]))
	
	print(f"\n epsilon_marginal:\n", epsilon_marginal_v)
	
	mi = ( 
	(n_variables - 1) * (digamma(n_samples) - digamma(k)) 
	+ np.sum(entropy_marginal_means) 
	- n_variables*np.mean(np.log(epsilon_joint)) 
	)	
	
	return mi


def data_generator(distribution, size, params, correlation=None, seed=None):
    """
    Generate data from a specified distribution, with optional correlation for univariate distributions.

    Parameters:
        distribution (str): The name of the distribution ('gaussian', 'exponential', 'gamma_exponential', 
                            'weinman_exponential', 'circular', 'uniform').
        size (int): The number of samples to generate.
        params (dict): Dictionary of parameters for the specific distribution.
        correlation (float, optional): Desired correlation coefficient for univariate distributions (-1 <= r <= 1).
        seed (int, optional): Seed for reproducibility.

    Returns:
        np.ndarray: Generated data. For univariate distributions, returns a 2D array with two correlated series.
                    For bivariate distributions, returns a 2D array with two dimensions.
    """
    rng = np.random.default_rng(seed)

    # Handle univariate distributions
    if distribution in ['gaussian', 'exponential', 'uniform']:
        # Generate two independent series based on the specified distribution
        if distribution == 'gaussian':
            mu = params.get('mu', 0)
            sigma = params.get('sigma', 1)
            series_1 = rng.normal(mu, sigma, size)
            series_2 = rng.normal(mu, sigma, size)
        elif distribution == 'exponential':
            lam = params.get('lambda', 1)
            series_1 = rng.exponential(1 / lam, size)
            series_2 = rng.exponential(1 / lam, size)
        elif distribution == 'uniform':
            low = params.get('low', 0)
            high = params.get('high', 1)
            series_1 = rng.uniform(low, high, size)
            series_2 = rng.uniform(low, high, size)
        
        # Enforce correlation if specified
        if correlation is not None:
            correlation_matrix = np.array([[1, correlation], [correlation, 1]])
            cholesky_decomposition = np.linalg.cholesky(correlation_matrix)
            independent_series = np.stack((series_1, series_2), axis=0)
            correlated_series = np.dot(cholesky_decomposition, independent_series)
            return correlated_series.T  # Return as (size, 2)

        return np.column_stack((series_1, series_2))  # Return as independent series

    # Handle bivariate Gamma-Exponential distribution
    elif distribution == 'gamma_exponential':
        alpha = params.get('alpha', 1)
        beta = params.get('beta', 1)
        x = rng.gamma(alpha, 1 / beta, size)
        y = rng.exponential(1 / beta, size)
        return np.column_stack((x, y))

    # Handle bivariate Ordered Weinman Exponential distribution
    elif distribution == 'weinman_exponential':
        mu = params.get('mu', 1)
        x = rng.exponential(mu, size)
        y = rng.exponential(mu, size)
        x, y = np.sort(x), np.sort(y)  # Ensure ordering
        return np.column_stack((x, y))

    # Handle circular distribution
    elif distribution == 'circular':
    	l = params.get('l', 1)
    	m = params.get('m', 1)
    	r = params.get('r', 1)
    	radius = rng.triangular(l, m, r, size)
    	angles = rng.uniform(0, 2 * np.pi, size)
    	x = radius * np.cos(angles)
    	y = radius * np.sin(angles)
    	return np.column_stack((x, y))

    # Raise an error for unsupported distributions
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

	





if __name__ == '__main__':
# Define a simple dataset with clear separations
	np.set_printoptions(precision=20, suppress=True)

	dataset = np.array([
		[1.0, 2.0, 3.0],
		[2.0, 3.0, 4.0],
		[3.0, 4.0, 5.0],
		[5.0, 6.0, 7.0],
		[1.5, 2.5, 3.5]
		])
	
	dataset = dataset*1.34564238e6
		
	#dataset = add_noise(dataset)
		
	# Number of nearest neighbors
	k = 2
	
	# Print the dataset for clarity
	print("Dataset:")
	print(dataset)
	
	#
	mi_right = mutual_information_1(dataset, k)
	# Print mutual information
	print("\nMutual Information right (Grassberger):", mi_right)	
	# Compute wrong mutual information 
	mi_wrong = mutual_information_1_wrong(dataset, k)
	# Print mutual information
	print("\nMutual Information wrong (Grassberger):", mi_wrong)
	







	
	
	
	
	
	
	
