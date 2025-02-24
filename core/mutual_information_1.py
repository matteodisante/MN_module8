import os
import sys
import numpy as np
from joblib import Parallel, delayed
from scipy.special import digamma

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/')))
from utils.decorators import time_it
from utils.core_utils import find_k_nearest_neighbors, compute_marginal_counts



def mutual_information_1(dataset, k, n_jobs = 2):

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
	distances = find_k_nearest_neighbors(dataset, k)
	epsilon = 2 * distances  # 2*Distance to the k-th nearest neighbor for each point

	
	# Step 2: Parallel computation of marginal counts
	def compute_counts_for_variable(var_idx):
		marginal_data = dataset[:, var_idx].reshape(-1, 1)
		return np.maximum(0, compute_marginal_counts(marginal_data, epsilon))
 
	results = Parallel(n_jobs=n_jobs)(delayed(compute_counts_for_variable)(var_idx) for var_idx in range(n_variables))
	marginal_counts = np.array(results)


    # Step 3: Compute the mutual information using Grassberger's formula
	mi = (
	digamma(k)
	+ (n_variables - 1) * digamma(n_samples)
	- np.mean(np.sum(digamma(marginal_counts + 1), axis=0))
	)

	return mi
