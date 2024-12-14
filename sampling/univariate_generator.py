import sys
import os
import numpy as np
from scipy.stats import norm, expon, uniform, pearsonr
from scipy.optimize import brentq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from decorators import time_it

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from math_utils import correlate_data





def adjust_correlation(target_corr, size, distribution, params, rng, a_tol=1e-5, max_iter=500):
	"""
	Adjust the correlation using Brent's method to achieve the target correlation for non-Gaussian distributions.

	Parameters:
		target_corr (float): Desired correlation (-1 <= target_corr <= 1).
		size (int): Number of samples to generate.
		distribution (str): The target distribution ('uniform' or 'exponential').
		params (dict): Parameters for the target distribution.
		rng: Random number generator instance.
		a_tol (float): Tolerance for the achieved correlation.
		max_iter (int): Maximum number of iterations.

	Returns:
		np.ndarray: A 2D array with two correlated series.
	"""

	def empirical_correlation_error(current_corr):
		"""
		Calculate the error between the achieved and target correlation.
		"""
		# Generate correlated Gaussian variables
		mean = [0, 0]
		cov = [[1, current_corr], [current_corr, 1]]
		gaussian_series = rng.multivariate_normal(mean, cov, size=size)
		corr_series_1, corr_series_2 = gaussian_series[:, 0], gaussian_series[:, 1]

		# Transform to target distribution
		uniform_1 = norm.cdf(corr_series_1)
		uniform_2 = norm.cdf(corr_series_2)

		if distribution == 'correlated_uniform':
			low = params.get('low', 0)
			high = params.get('high', 1)
			series_1 = uniform.ppf(uniform_1, loc=low, scale=high - low)
			series_2 = uniform.ppf(uniform_2, loc=low, scale=high - low)
		elif distribution == 'correlated_exponential':
			lam = params.get('lambda', 1)
			series_1 = expon.ppf(uniform_1, scale=1 / lam)
			series_2 = expon.ppf(uniform_2, scale=1 / lam)
		else:
			raise ValueError(f"Unsupported distribution: {distribution}")

		# Compute the empirical correlation and return the error
		empirical_corr, _ = pearsonr(series_1, series_2)
		return empirical_corr - target_corr

	# Find the optimal correlation guess using Brent's method
	print("Optimizing correlation using Brent's method...")
	try:
		optimal_corr = brentq(empirical_correlation_error, -1, 1, xtol=a_tol, maxiter=max_iter)
	except ValueError as e:
		raise RuntimeError(f"Failed to find optimal correlation: {e}")

	# Generate final series using the optimal correlation
	mean = [0, 0]
	cov = [[1, optimal_corr], [optimal_corr, 1]]
	gaussian_series = rng.multivariate_normal(mean, cov, size=size)
	corr_series_1, corr_series_2 = gaussian_series[:, 0], gaussian_series[:, 1]

	uniform_1 = norm.cdf(corr_series_1)
	uniform_2 = norm.cdf(corr_series_2)

	if distribution == 'correlated_uniform':
		low = params.get('low', 0)
		high = params.get('high', 1)
		series_1 = uniform.ppf(uniform_1, loc=low, scale=high - low)
		series_2 = uniform.ppf(uniform_2, loc=low, scale=high - low)
	elif distribution == 'correlated_exponential':
		lam = params.get('lambda', 1)
		series_1 = expon.ppf(uniform_1, scale=1 / lam)
		series_2 = expon.ppf(uniform_2, scale=1 / lam)

	# Print the final achieved correlation
	final_corr, _ = pearsonr(series_1, series_2)
	print(f"Achieved correlation: {final_corr:.6f}")

	return np.column_stack((series_1, series_2))



@time_it
def generate_univariate_data(distribution, size, params, correlation=None, a_tol=1e-3, max_iter=4000, seed=None):
	"""
	Generate data from a specified distribution, with optional correlation for univariate distributions.
	"""
	rng = np.random.default_rng(seed)

	# Handle Gaussian
	if distribution in ['correlated_gaussian']:
		mu = params.get('mu', 0)
		sigma = params.get('sigma', 1)
		series_1 = rng.normal(mu, sigma, size)
		series_2 = rng.normal(mu, sigma, size)

		if correlation is not None:
			corr_series = correlate_data(np.column_stack((series_1, series_2)), correlation)
			series_1, series_2 = corr_series[:, 0], corr_series[:, 1]

		return np.column_stack((series_1, series_2))

	# Handle Uniform and Exponential with iterative correlation adjustment
	if distribution in ['correlated_uniform', 'correlated_exponential']:
		if correlation is not None:
			# Use the adjust_correlation function to fine-tune correlation
			return adjust_correlation(correlation, size, distribution, params, rng, a_tol, max_iter)
		else:
			raise ValueError("Correlation must be provided for uniform or exponential distributions.")

	# Raise an error for unsupported distributions
	raise ValueError(f"Unsupported distribution: {distribution}")







