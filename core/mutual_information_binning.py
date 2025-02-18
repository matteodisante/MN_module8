import os
import sys
import numpy as np
import logging
from sklearn.preprocessing import KBinsDiscretizer




def mutual_information_binning_adaptive(data, num_bins):
    """
    Estimate mutual information using adaptive binning and first-order correction.

    :param data: 2D NumPy array (n_samples, 2), where each row is a sample (x, y).
    :param num_bins: Number of bins alog each axis for discretizing the data. Total number of bins = num_bins^2
    :return: Estimated mutual information (MI).
    """

    # Adaptive binning using quantile strategy
    discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='quantile')
    
    try:
        binned_data = discretizer.fit_transform(data)
    except Exception as e:
        logging.error(f"Error in KBinsDiscretizer: {e}")
        return None, None # If binning fails

    # Compute joint histogram. It's ok just for (X,Y) in R2
    joint_hist, _, _ = np.histogram2d(binned_data[:, 0], binned_data[:, 1], bins=[num_bins, num_bins])

    # Compute how many cells contain at least one point
    non_empty_cells = np.count_nonzero(joint_hist)

    # Compute total number of cells
    total_cells =  (len(discretizer.bin_edges_[0]) - 1) * (len(discretizer.bin_edges_[1]) - 1)  # Subtract 1 from each to get the number of intervals
    
    
    # Normalize joint histogram to get probabilities
    joint_prob = joint_hist / np.sum(joint_hist)
    
    # Compute marginal probabilities
    p_x = np.sum(joint_prob, axis=1, keepdims=True)  # Shape (max_bins, 1)
    p_y = np.sum(joint_prob, axis=0, keepdims=True)  # Shape (1, max_bins)
    

    # Compute mutual information
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            # Usa il broadcasting per ottenere il prodotto esterno
            outer_prob = p_x * p_y  
            valid_mask = (joint_prob > 0) & (outer_prob > 0)
            mi = np.sum(np.where(valid_mask, joint_prob * np.log(joint_prob / outer_prob), 0))
    except Exception as e:
        logging.error(f"Error computing mutual information: {e}")
        return None, None
    
    return mi, non_empty_cells









