import sys
import os
import numpy as np
import shutil
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/')))
from io_utils import load_data
from interface_utils import navigate_directories

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/')))
from mutual_information_1 import *
from mutual_information_1_entropies_sum import *
from mutual_information_binning import *


def compute_std_corr_matrix(data):
    """
    Compute a matrix with standard deviations on the diagonal and correlations off-diagonal.

    :param data: 2D NumPy array where rows are samples and columns are variables.
    :return: 2D NumPy array with standard deviations on the diagonal and correlations off-diagonal.
    """
    std_devs = np.std(data, axis=0, ddof=1)  # Compute standard deviations
    corr_matrix = np.corrcoef(data, rowvar=False)  # Compute correlation matrix
    std_corr_matrix = corr_matrix.copy()

    # Replace diagonal elements with standard deviations
    np.fill_diagonal(std_corr_matrix, std_devs)

    return std_corr_matrix



# Functions that compute the mi estimate for a single file and for a directory.

def process_file(file_path, k, mi_estimate):
    """
    Process a single file to compute mutual information.

    :param file_path: Path to the file.
    :param k: Number of nearest neighbors for mutual information calculation.
    :param mi_estimate: A function to estimate mutual information, should accept (data, k) as arguments.
    :return: The mutual information value.
    """
    try:
        # Load data from the file
        data = load_data(file_path)

        # Compute mutual information
        mi = mi_estimate(data, k)

        return mi
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None



def extract_file_details(file_path):
    """
    Extracts distribution name, size, parameters, and file index from a given file path.

    :param file_path: Path to the file.
    :return: A dictionary with distribution_name, size, params, and file_index.
    """
    base_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)

    # Extract file index from the file name
    file_index = base_name.split('.')[0]

    # Extract size from the directory "size_*"
    size_dir = [part for part in dir_name.split(os.sep) if part.startswith("size_")]
    size = size_dir[0].split('_')[1] if size_dir else "unknown"

    # Extract parameters from directories above "size_*"
    dir_parts = dir_name.split(os.sep)
    size_index = dir_parts.index(size_dir[0]) if size_dir else len(dir_parts)
    params = '_'.join(dir_parts[size_index - 2:size_index]) if size_index > 1 else "default"

    # Extract distribution name (the directory before params)
    distribution_name = dir_parts[size_index - 3] if size_index > 2 else "unknown"

    return {
        "distribution_name": distribution_name,
        "size": size,
        "params": params,
        "file_index": file_index
    }

def process_and_save_mi_table(file_path, num_bins=10):
    """
    Process a dataset file and save mutual information calculations to a CSV file.

    :param file_path: Path to the input dataset file.
    :param num_bins: Number of bins for the adaptive binning MI calculation.
    :return: Path to the generated CSV file.
    """
    try:
        # Extract details from the file path
        details = extract_file_details(file_path)
        distribution_name = details["distribution_name"]
        size = details["size"]
        params = details["params"]
        file_index = details["file_index"]

        # Prepare output CSV file name
        output_csv = f"mi_{distribution_name}_size_{size}_params_{params}_file_{file_index}.csv"

        # Load the dataset
        data = load_data(file_path)

        # Prepare rows for the CSV
        rows = [["k", "mi_1", "mi_sum", "mi_binning"]]
        print('before computing mi')
        for k in range(1, 3):
            # Compute MI using the three methods
            mi_1 = mutual_information_1(data, k)
            mi_sum = mutual_information_1_entropies_sum(data, k)
            mi_binning = mutual_information_1(data, k)

            # Append the results
            rows.append([k, mi_1, mi_sum, mi_binning])
            
        print('After for cycle in computng mi')

        # Write the results to the CSV file
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        print(f"Mutual information results saved to: {output_csv}")
        return output_csv

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def create_summary_csv_for_folder(folder_path):
    """
    Create a summary CSV file for all CSVs in a given folder. Computes the mean and standard deviation
    of the mutual information values for each k and saves the summary in the same folder.

    :param folder_path: Path to the folder containing the CSV files.
    """
    try:
        # Find all CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not csv_files:
            print(f"[INFO] No CSV files found in folder: {folder_path}")
            return

        # Initialize storage for data
        mi_1_values = []
        mi_sum_values = []
        mi_binning_values = []

        # Loop through each CSV file and extract data
        for csv_file in csv_files:
            try:
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(csv_file)

                # Append values for each k
                mi_1_values.append(df["mi_1"].to_numpy())
                mi_sum_values.append(df["mi_sum"].to_numpy())
                mi_binning_values.append(df["mi_binning"].to_numpy())
            except Exception as e:
                print(f"[ERROR] Failed to process {csv_file}: {e}")
                continue

        # Convert the collected values into numpy arrays
        mi_1_values = np.array(mi_1_values)
        mi_sum_values = np.array(mi_sum_values)
        mi_binning_values = np.array(mi_binning_values)

        # Calculate means and standard deviations for each k
        k_values = np.arange(1, 31)
        mean_mi_1 = np.mean(mi_1_values, axis=0)
        sigma_mi_1 = np.std(mi_1_values, axis=0)
        mean_mi_sum = np.mean(mi_sum_values, axis=0)
        sigma_mi_sum = np.std(mi_sum_values, axis=0)
        mean_mi_binning = np.mean(mi_binning_values, axis=0)
        sigma_mi_binning = np.std(mi_binning_values, axis=0)

        # Prepare the summary DataFrame
        summary_df = pd.DataFrame({
            "k": k_values,
            "mean_mi_1": mean_mi_1,
            "sigma_mi_1": sigma_mi_1,
            "mean_mi_sum": mean_mi_sum,
            "sigma_mi_sum": sigma_mi_sum,
            "mean_mi_binning": mean_mi_binning,
            "sigma_mi_binning": sigma_mi_binning,
        })

        # Extract distribution, size, and params info from folder path
        folder_name = os.path.basename(folder_path)
        parent_folder = os.path.basename(os.path.dirname(folder_path))
        grandparent_folder = os.path.basename(os.path.dirname(os.path.dirname(folder_path)))

        # Build the output file name
        summary_csv_name = f"mi_{grandparent_folder}_{parent_folder}_{folder_name}_mean_error.csv"
        summary_csv_path = os.path.join(folder_path, summary_csv_name)

        # Save the summary to a CSV file
        summary_df.to_csv(summary_csv_path, index=False)

        print(f"[INFO] Summary CSV saved: {summary_csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed to create summary CSV for folder {folder_path}: {e}")


def analyze_and_save_mi_values(input_dir, output_dir, num_bins=10):
    """
    Analyze .txt files in the input directory, compute mutual information, and save results
    in a structured output directory.

    :param input_dir: Path to the input directory containing .txt files.
    :param output_dir: Path to the output directory where results will be saved.
    :param num_bins: Number of bins for the adaptive binning MI calculation.
    """
    # Step 1: Navigate and select files
    selected_files = navigate_directories(start_path=input_dir, multi_select=True, file_extension=".txt")

    if not selected_files:
        print("[INFO] No files selected. Exiting.")
        return

    # Step 2: Create the output directory structure
    mi_values_dir = os.path.join(output_dir, "mi_values")
    os.makedirs(mi_values_dir, exist_ok=True)

    # Step 3: Process each file and save the results
    for file_path in selected_files:
        try:
            # Extract details from the file path
            details = extract_file_details(file_path)
            distribution_name = details["distribution_name"]
            size = details["size"]
            params = details["params"]

            # Build the output subdirectory structure
            subfolder_path = os.path.join(
                mi_values_dir,
                distribution_name,
                params,
                f"size_{size}"
            )
            os.makedirs(subfolder_path, exist_ok=True)

            # Process the file and save the CSV in the corresponding subfolder
            output_csv = process_and_save_mi_table(file_path, num_bins=num_bins)

            if output_csv:
                # Move the generated CSV to the corresponding subfolder
                output_csv_name = os.path.basename(output_csv)
                shutil.move(output_csv, os.path.join(subfolder_path, output_csv_name))

        except Exception as e:
            print(f"[ERROR] Failed to process file {file_path}: {e}")

    # Step 4: Create summary CSVs for each folder
    for root, dirs, files in os.walk(mi_values_dir):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            create_summary_csv_for_folder(folder_path)

    print(f"[INFO] Analysis completed. Results saved in: {mi_values_dir}")