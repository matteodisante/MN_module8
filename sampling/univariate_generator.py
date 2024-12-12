import sys
import os
import numpy as np
from scipy.stats import norm, expon, uniform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from decorators import time_it

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from math_utils import correlate_data



@time_it
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
	
	# Handle admitted univariate distributions
	if distribution in ['gaussian']:  	
		# Generate two independent series based on the specified distribution
	    mu = params.get('mu', 0)
	    sigma = params.get('sigma', 1)
	    series_1 = rng.normal(mu, sigma, size)
	    series_2 = rng.normal(mu, sigma, size)
	    return np.column_stack((series_1, series_2))
			
	    if correlation is not None:
	    	corr_series = correlate_data(np.column_stack((series_1, series_2)), correlation)
	    	series_1 = corr_series[:,0]
	    	series_2 = corr_series[:,1]
	    	return np.column_stack((series_1, series_2))
	    	
	
	# Generate two independent gaussian series with mean 0 and std.dev 1        
	
	if distribution in ['exponential', 'uniform']:
		series_1 = rng.normal(0, 1, size)
		series_2 = rng.normal(0, 1, size)
		#Apply Cholesky decompostion to create a third series correlated with series_1 with a given correlation 
		corr_series = correlate_data(np.column_stack((series_1, series_2)), correlation)
		# The following corr_series_1 and corr_series_2 are again gaussian and are corrrelated
		corr_series_1 = corr_series[:,0]
		corr_series_2 = corr_series[:,1]
		emp_corr = np.corrcoef(corr_series_1, corr_series_2)
		print("Given correlation is: {correlation}")
		print("Gotten correlation is {emp_corr}")
		#Transform the two correlated gaussian series in two correlated uniform series through the comulative function of a standard normal
		uniform_1 = norm.cdf(corr_series_1, loc=0, scale=1)
		uniform_2 = norm.cdf(corr_series_2, loc=0, scale=1)
		
		if distribution == 'uniform':
			low = params.get('low', 0)
			high = params.get('high', 1)
			# Apply ppf to shift the uniform distribution from [0,1] to [loc, loc + scale].
			series_1 = uniform.ppf(uniform_1, loc = low, scale = high - low) 
			series_2 = uniform.ppf(uniform_2, loc = low, scale = high - low)
			return np.column_stack((series_1, series_2))

		if distribution == 'exponential':
			lam = params.get('lambda', 1)
			series_1 = expon.ppf(uniform_1, loc=0, scale=1/lam)
			series_2 = expon.ppf(uniform_2, loc=0, scale=1/lam)
			return np.column_stack((series_1, series_2))
	            
	# Raise an error for unsupported distributions
	if distribution not in ['gaussian', 'exponential', 'uniform']:
		raise ValueError(f"Unsupported distribution: {distribution}")
