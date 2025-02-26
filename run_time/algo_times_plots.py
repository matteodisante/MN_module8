import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

# Add the path for the utility modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/')))
from interface_utils import navigate_directories, setup_logger
from io_utils import extract_file_details, ensure_directory

if __name__ == '__main__':
    setup_logger(stdout_printing=True)
    
    logging.info("Select file containing central values of computation times for a given mi estimate")
    central_value_file_path = navigate_directories(
        start_path="../computation_times/",
        multi_select=False,
        file_extension=".txt"
    )[0]
    
    logging.info("Select file containing uncertainties for the same mi estimate")
    error_value_file_path = navigate_directories(
        start_path="../computation_times/",
        multi_select=False,
        file_extension=".txt"
    )[0]
    
    # Extract details from the file path (distribution name, params, size, file index)
    details = extract_file_details(central_value_file_path)
    
    # Read the files with times and uncertainties
    # (Assuming the first column contains N and the subsequent columns represent different k values)
    run_times_df = pd.read_csv(central_value_file_path, sep='\t', header=0)
    run_errors_df = pd.read_csv(error_value_file_path, sep='\t', header=0)
    
    # Extract the N values from the first column (each row corresponds to a fixed N)
    N_values = run_times_df.iloc[:, 0].values
    # Extract the k values from the headers (excluding the first column) and convert them to float
    k_headers = run_times_df.columns[1:].astype(float)
    
    # Extract the matrices of times and uncertainties (excluding the first column)
    times_matrix = run_times_df.iloc[:, 1:].values
    errors_matrix = run_errors_df.iloc[:, 1:].values
    
    # Set up the directory to save the plots
    plots_dir = os.path.join("..", "plots", "time_estimates")
    if not ensure_directory(plots_dir):
        sys.exit("Operation cancelled by the user.")
    
    ###############################################################
    # First plot: Curves for fixed N (x = k/N)
    ###############################################################
    plt.figure(figsize=(9, 6))
    plt.xscale('log')
    plt.yscale('log')
    
    # Loop through each row (each fixed N)
    for i, N in enumerate(N_values):
        row_times = times_matrix[i, :]
        row_errors = errors_matrix[i, :]
        
        # Filter valid points (time > 0)
        valid_mask = row_times > 0
        if not np.any(valid_mask):
            continue
        
        # Compute x = k/N for the valid points
        x_vals = k_headers[valid_mask] / N
        y_vals = row_times[valid_mask]
        y_errs = row_errors[valid_mask]
        
        # Estimate the slope using linear regression on the log-transformed data if there are at least two points
        if len(x_vals) >= 2:
            log_x = np.log(x_vals)
            log_y = np.log(y_vals)
            slope, intercept = np.polyfit(log_x, log_y, 1)
        else:
            slope = np.nan
        
        plt.errorbar(
            x_vals, y_vals, yerr=y_errs,
            fmt='o-', capsize=5, label=f'N={int(N)} (slope={slope:.2f})'
        )
    
    plt.xlabel('k / N (log scale)')
    plt.ylabel('Time (log scale)')
    plt.title('Log-Log Plot of Time vs k/N for each fixed N')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot with a filename that includes the extracted details
    plot_filename_N_fixed = f"{details['distribution_name']}_{details['params']}_{details['size']}_{details['file_index']}_N_fixed.png"
    plt.savefig(os.path.join(plots_dir, plot_filename_N_fixed))
    plt.show()
    
    ###############################################################
    # Second plot: Curves for fixed k (x = k/N, with N varying)
    ###############################################################
    plt.figure(figsize=(9, 6))
    plt.xscale('log')
    plt.yscale('log')
    
    # Loop through each column (each fixed k)
    for j, k in enumerate(k_headers):
        col_times = times_matrix[:, j]      # Times for this fixed k for every N
        col_errors = errors_matrix[:, j]      # Corresponding uncertainties
        # Compute x = k/N for each N
        x_vals = k / N_values
        valid_mask = col_times > 0
        if not np.any(valid_mask):
            continue
        
        x_vals_valid = x_vals[valid_mask]
        y_vals_valid = col_times[valid_mask]
        y_errs_valid = col_errors[valid_mask]
        
        # Estimate the slope if there are at least two valid points
        if len(x_vals_valid) >= 2:
            log_x = np.log(x_vals_valid)
            log_y = np.log(y_vals_valid)
            slope, intercept = np.polyfit(log_x, log_y, 1)
        else:
            slope = np.nan
        
        plt.errorbar(
            x_vals_valid, y_vals_valid, yerr=y_errs_valid,
            fmt='o-', capsize=5, label=f'k={k} (slope={slope:.2f})'
        )
    
    plt.xlabel('k / N (log scale)')
    plt.ylabel('Time (log scale)')
    plt.title('Log-Log Plot of Time vs k/N for each fixed k')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Save the second plot with a filename that includes the extracted details
    plot_filename_k_fixed = f"{details['distribution_name']}_{details['params']}_{details['size']}_{details['file_index']}_k_fixed.png"
    plt.savefig(os.path.join(plots_dir, plot_filename_k_fixed))
    plt.show()