import os
import sys
import numpy as np
from scipy.special import digamma


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from utils.core_utils import find_k_nearest_neighbors



def mutual_information_1_entropies_sum(dataset, k):

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
	
	epsilon_marginal_v = np.zeros(dataset.shape)
	entropy_marginal_means = np.zeros(n_variables)
	
	for var_idx in range(n_variables):
		# Extract the current variable as a 1D array
		marginal_data = dataset[:, var_idx]
		_, distances_marginal = find_k_nearest_neighbors(marginal_data.reshape(-1, 1), k)
		epsilon_marginal_v[:, var_idx] = 2 * distances_marginal[:, k-1]
		entropy_marginal_means[var_idx] = np.mean(np.log(np.maximum(epsilon_marginal_v[:, var_idx], 1e-10)))
		
	mi = ( 
	(n_variables - 1) * (digamma(n_samples) - digamma(k)) 
	+ np.sum(entropy_marginal_means) 
	- n_variables*np.mean(np.log(np.maximum(epsilon_joint, 1e-10))) 
	)	
	
	return mi
