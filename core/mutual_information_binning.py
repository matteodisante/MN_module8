import os
import sys
import numpy as np
import logging
from sklearn.preprocessing import KBinsDiscretizer


def mutual_information_binning_adaptive(data, num_bins):
    """
    Estimate mutual information using adaptive binning and first-order correction.

    :param data: 2D NumPy array (n_samples, 2), where each row is a sample (x, y).
    :param num_bins: Number of bins for discretizing the data.
    :return: Estimated mutual information (MI).
    """
    
    # Validate input
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Input 'data' must be a 2D NumPy array with shape (n_samples, 2).")
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("'num_bins' must be a positive integer.")
    
    # Ensure we have enough unique values for binning
    unique_x = len(np.unique(data[:, 0]))
    unique_y = len(np.unique(data[:, 1]))
    
    max_bins = min(num_bins, unique_x, unique_y, data.shape[0] - 1)  # Ensure no more bins than samples
    
    if max_bins < num_bins:
        logging.warning(f"Reducing bins_number from {num_bins} to {max_bins} due to insufficient unique values.")

    if max_bins < 2:
        logging.warning("Not enough unique values for binning. Returning MI=0.")
        return 0  # If not enough data, return MI=0

    # Adaptive binning using quantile strategy
    discretizer = KBinsDiscretizer(n_bins=max_bins, encode='ordinal', strategy='quantile')
    
    try:
        binned_data = discretizer.fit_transform(data)
    except Exception as e:
        logging.error(f"Error in KBinsDiscretizer: {e}")
        return 0  # If binning fails, return MI=0

    # Compute joint histogram
    joint_hist, x_edges, y_edges = np.histogram2d(binned_data[:, 0], binned_data[:, 1], bins=[max_bins, max_bins])

    # Ensure the histogram shape matches expected dimensions
    if joint_hist.shape != (max_bins, max_bins):
        logging.error(f"Shape mismatch in joint histogram: expected ({max_bins}, {max_bins}), got {joint_hist.shape}")
        return 0
    
    # Normalize joint histogram to get probabilities
    joint_prob = joint_hist / np.sum(joint_hist)
    
    # Compute marginal probabilities
    p_x = np.sum(joint_prob, axis=1, keepdims=True)  # Shape (max_bins, 1)
    p_y = np.sum(joint_prob, axis=0, keepdims=True)  # Shape (1, max_bins)
    
    # Avoid division by zero
    p_x[p_x == 0] = 1e-10
    p_y[p_y == 0] = 1e-10
    joint_prob[joint_prob == 0] = 1e-10  # Avoid log(0)

    # Compute mutual information
    try:
        mi = np.sum(joint_prob * np.log(joint_prob / (p_x @ p_y)))
    except Exception as e:
        logging.error(f"Error computing mutual information: {e}")
        return 0  # If computation fails, return MI=0
    
    return mi




