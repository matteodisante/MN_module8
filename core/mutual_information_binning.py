import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/')))
from utils.decorators import time_it



def mutual_information_binning(data, num_bins):
    """
    Estimate mutual information using binning.

    :param data: 2D array of shape (n_samples, 2), where each row is a sample (x, y).
    :param num_bins: Number of bins for discretizing the data.
    :return: Estimated mutual information.
    """
    # Validazione degli input
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Input 'data' must be a 2D numpy array with shape (n_samples, 2).")
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("'num_bins' must be a positive integer.")

    # Rimuovere campioni con valori mancanti o infiniti
    data = data[~np.any(np.isnan(data) | np.isinf(data), axis=1)]
    if data.shape[0] == 0:
        raise ValueError("No valid data points remain after removing NaN or Inf values.")

    # Estrazione delle variabili
    x, y = data[:, 0], data[:, 1]

    # Creazione dei bin
    x_bins = np.linspace(np.min(x), np.max(x), num_bins + 1)
    y_bins = np.linspace(np.min(y), np.max(y), num_bins + 1)

    # Calcolo della distribuzione congiunta
    joint_hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    joint_prob = joint_hist / np.sum(joint_hist)  # Normalizzazione

    # Distribuzioni marginali
    p_x = np.sum(joint_prob, axis=1, keepdims=True)  # Marginale di X
    p_y = np.sum(joint_prob, axis=0, keepdims=True)  # Marginale di Y

    # Evitare divisioni per zero
    valid_bins = joint_prob > 0  # Considera solo i bin con probabilità > 0

    # Calcolo della mutua informazione (vettorializzata)
    mi = np.sum(
        joint_prob[valid_bins] * 
        np.log(joint_prob[valid_bins] / (p_x[valid_bins.sum(axis=1), :] * p_y[:, valid_bins.sum(axis=0)]))
    )

    return mi



def mutual_information_binning_adaptive(data, num_bins):
    """
    Estimate mutual information using adaptive binning and first-order correction.

    :param data: 2D array of shape (n_samples, 2), where each row is a sample (x, y).
    :param num_bins: Target number of bins for each variable.
    :return: Estimated mutual information with adaptive binning and corrections.
    """
    from sklearn.preprocessing import KBinsDiscretizer

    # Validate input
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Input 'data' must be a 2D numpy array with shape (n_samples, 2).")
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("'num_bins' must be a positive integer.")
    
    # Adaptive binning using quantile strategy
    discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='quantile')
    binned_data = discretizer.fit_transform(data)

    # Joint and marginal frequencies
    joint_hist, _, _ = np.histogram2d(binned_data[:, 0], binned_data[:, 1], bins=[num_bins, num_bins])
    joint_prob = joint_hist / np.sum(joint_hist)  # Normalize to joint probabilities

    p_x = np.sum(joint_prob, axis=1, keepdims=True)  # Marginal distribution of X
    p_y = np.sum(joint_prob, axis=0, keepdims=True)  # Marginal distribution of Y

    # Apply first-order correction for finite sample size
    N = data.shape[0]  # Number of samples
    joint_prob_corrected = joint_prob - 1 / (2 * N)
    joint_prob_corrected[joint_prob_corrected < 0] = 0  # Avoid negative probabilities

    # Mutual information calculation (vectorized)
    valid_bins = joint_prob > 0  # Consider only bins with probability > 0
    mi = np.sum(
        joint_prob_corrected[valid_bins] *
        np.log(joint_prob_corrected[valid_bins] / (p_x[valid_bins.sum(axis=1), :] * p_y[:, valid_bins.sum(axis=0)]))
    )

    return mi