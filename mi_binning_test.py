import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import KBinsDiscretizer

# Ensure the utilities directory is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './utils/')))
from utils.interface_utils import navigate_directories


def mutual_information_binning_adaptive(data, num_bins):
    """
    Estimate mutual information using adaptive binning and first-order correction.
    :param data: 2D NumPy array (n_samples, 2), where each row is a sample (x, y).
    :param num_bins: Number of bins along each axis for discretizing the data.
    Total number of bins = num_bins^2
    :return: Estimated mutual information (MI).
    """
    # Use quantile strategy for adaptive binning
    discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='quantile')
    
    try:
        binned_data = discretizer.fit_transform(data)
        x_bin_edges = discretizer.bin_edges_[0]
        y_bin_edges = discretizer.bin_edges_[1]
    except Exception as e:
        logging.error(f"Error in KBinsDiscretizer: {e}")
        return np.nan, None, None, None, None, None, None, None

    # Compute joint histogram
    joint_hist, x_edges, y_edges = np.histogram2d(binned_data[:, 0], binned_data[:, 1], bins=[num_bins, num_bins])

    # Compute total and non-empty cells
    total_cells = (len(x_bin_edges) - 1) * (len(y_bin_edges) - 1)
    non_empty_cells = np.count_nonzero(joint_hist)
    
    # Normalize and compute probabilities
    joint_prob = joint_hist / np.sum(joint_hist)
    p_x = np.sum(joint_prob, axis=1, keepdims=True)
    p_y = np.sum(joint_prob, axis=0, keepdims=True)

    # Mutual information computation
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            outer_prob = p_x * p_y  
            valid_mask = (joint_prob > 0) & (outer_prob > 0)
            mi = np.sum(np.where(valid_mask, joint_prob * np.log(joint_prob / outer_prob), 0))
    except Exception as e:
        logging.error(f"Error computing mutual information: {e}")
        return np.nan, None, None, None, None, None, None, None
    
    return mi, total_cells, non_empty_cells, joint_hist, x_edges, y_edges, x_bin_edges, y_bin_edges


def extract_params_from_path(file_path):
    """ Extracts parameters from the file path to use in plot title and filename. """
    parts = file_path.split('/')
    dataset_type = parts[-4]
    params = parts[-3]
    dataset_size = parts[-2]

    return f"{dataset_type} - {params} - {dataset_size}"

if __name__ == '__main__':
    selected_files = navigate_directories(start_path=".", file_extension=".txt", multi_select=False)
    
    if not selected_files:
        print("No file selected, exiting.")
        sys.exit(1)

    filename = selected_files[0]
    try:
        data = np.loadtxt(filename)
    except Exception as e:
        logging.error(f"Error loading the file {filename}: {e}")
        sys.exit(1)
    
    if data.ndim != 2 or data.shape[1] != 2:
        logging.error(f"The file must contain two columns, but {data.shape[1]} were found")
        sys.exit(1)

    print(f"Loaded {data.shape[0]} points from the file {filename}")

    num_bins = int(input("Enter the number of bins per axis (e.g., 10, 100, 1000): "))

    params = extract_params_from_path(filename)
    
    mi, total_cells, non_empty_cells, joint_hist, x_edges, y_edges, real_x_edges, real_y_edges = mutual_information_binning_adaptive(data, num_bins)

    short_title = f"{data.shape[0]} Points - {num_bins} Bins"
    
    print(f"\nMutual Information: {mi:.4f}")
    print(f"Total number of cells: {total_cells}")
    print(f"Non-empty cells: {non_empty_cells}")
    
    # Plotting
    plt.figure(figsize=(10, 7))
    plt.pcolormesh(real_x_edges, real_y_edges, joint_hist.T, cmap='viridis')
    plt.colorbar(label='Number of points per cell')
    plt.xlabel('X edge')
    plt.ylabel('Y edge')
    title_string = f'{params} | {short_title} - Non-Empty: {non_empty_cells}'
    plt.title(title_string)
    file_name = f'MI_{params.replace(" ", "_")}_{short_title.replace(" ", "_")}_{non_empty_cells}nonempty.png'
    plt.savefig(file_name, dpi=300)
    plt.show()