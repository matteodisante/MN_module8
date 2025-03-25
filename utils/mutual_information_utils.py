import sys
import os
import numpy as np
import csv
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/utils/')))
from utils.io_utils import load_data
from utils.interface_utils import navigate_directories

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/')))
from core.mutual_information_1 import mutual_information_1
from core.mutual_information_1_entropies_sum import mutual_information_1_entropies_sum
from core.mutual_information_binning import mutual_information_binning_adaptive



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



def process_file(file_path, k, mi_estimate):
    """
    Process a single file to compute mutual information.

    :param file_path: Path to the file.
    :param k: Number of nearest neighbors for mutual information calculation.
    :param mi_estimate: A function to estimate mutual information, should accept (data, k) as arguments.
    :return: The mutual information value.
    """
    try:
        logging.info(f"Starting to process file: {file_path} with k={k}")

        # Load data from the file
        data = load_data(file_path)

        if data is None:
            logging.error(f"Failed to load data from {file_path}. Skipping processing.")
            return None

        # Compute mutual information
        mi = mi_estimate(data, k)

        logging.info(f"Computed MI for file: {file_path} with k={k} is {mi}")

        return mi
    except Exception as e:
        logging.error(f"Error processing {file_path} with k={k}: {e}")
        return None







def process_and_save_mi_table(file_path, output_dir, k_values, num_bins=10, mi_estimate_function=None):
    """
    Process a dataset file and save mutual information calculations to a CSV file.
    """
    try:
        if mi_estimate_function is None:
            raise ValueError("mi_estimate_function cannot be None. Provide a valid MI estimation function.")

        logging.info(f"Started processing file: {file_path}")

        # Map function to column name
        mi_estimate_map = {
            mutual_information_1: "mi_1",
            mutual_information_1_entropies_sum: "mi_sum",
            mutual_information_binning_adaptive: "mi_binning"
        }

        if mi_estimate_function not in mi_estimate_map:
            raise ValueError("Invalid mi_estimate_function. Choose from mutual_information_1, mutual_information_1_entropies_sum, or mutual_information_binning_adaptive.")

        mi_estimate = mi_estimate_map[mi_estimate_function]

        # Extract details from the file path
        details = extract_file_details(file_path)
        distribution_name = details.get("distribution_name", None)
        size = details.get("size", None)
        params = details.get("params", None)
        file_index = details.get("file_index", None)

        # Build the relative path based on input file structure
        relative_output_path_parts = []

        if distribution_name is not None:
            relative_output_path_parts.append(distribution_name)
        if size is not None:
            relative_output_path_parts.append(f"size_{size}")
        if params is not None:
            relative_output_path_parts.append(f"params_{params}")

        # Join parts to create the relative output path
        relative_output_path = os.path.join(*relative_output_path_parts)

        # Build the output CSV file name
        if distribution_name or size or params:
            output_csv_name = f"{mi_estimate}_{distribution_name if distribution_name else ''}{f'_N{size}' if size else ''}{f'_{params}' if params else ''}_{file_index}.csv"
        else:
            output_csv_name = f"{mi_estimate}_{file_index}.csv"

        # Full path to the output file
        full_output_path = os.path.join(output_dir, relative_output_path)
        output_csv = os.path.join(full_output_path, output_csv_name)

        # Create necessary directories if they don't exist
        os.makedirs(full_output_path, exist_ok=True)

        # Check for existing files
        existing_files = navigate_directories(
            start_path=output_dir,
            multi_select=True,
            file_extension=".csv"
        )
        matched_files = [file for file in existing_files if file.endswith(output_csv_name)]

        if matched_files:
            logging.warning(f"Found existing file(s) with the name '{output_csv_name}':")
            for file in matched_files:
                logging.warning(f"  - {file}")

            overwrite = input(f"Do you want to overwrite the existing file(s)? [y/n]: ").strip().lower()
            if overwrite != "y":
                logging.info(f"File not overwritten, returning existing file: {matched_files[0]}")
                return matched_files[0]

        # Load the dataset
        data = load_data(file_path)

        # Prepare rows for the CSV
        rows = [["k", mi_estimate]]
        for k in k_values:
            # Compute MI using the selected function
            if mi_estimate_function == mutual_information_binning_adaptive:
                mi_value = mi_estimate_function(data, num_bins)
            else:
                mi_value = mi_estimate_function(data, k)

            rows.append([k, mi_value])

        # Write the results to the CSV file
        with open(output_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        logging.info(f"Mutual information results saved to: {output_csv}")
        return output_csv

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None



@time_it
def aggregate_mi_results(base_dir, output_dir, mi_estimate_functions, k_values):
    """
    Aggregates mutual information results from detailed CSV files and updates summary aggregation CSVs.

    :param base_dir: Path to the directory containing the processed MI CSV files.
    :param output_dir: Path to the directory where the summary CSVs will be saved.
    :param mi_estimate_functions: List of mutual information estimation functions to consider.
    :param k_values: List or range of k values to include in the aggregation.
    """
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        logging.info("Started aggregating mutual information results.")

        # Map MI functions to column names
        mi_estimate_map = {
            mutual_information_1: "mi_1",
            mutual_information_1_entropies_sum: "mi_sum",
            mutual_information_binning: "mi_binning"
        }

        # Filter only selected MI estimation functions
        selected_columns = [mi_estimate_map[func] for func in mi_estimate_functions if func in mi_estimate_map]
        logging.info(f"Selected MI functions: {selected_columns}")

        # Get all CSV files in the base directory
        csv_files = navigate_directories(
            start_path=base_dir,
            multi_select=True,
            file_extension=".csv"
        )
        logging.info(f"Found {len(csv_files)} CSV files in the base directory.")

        # Group files by distribution, size, params, and MI estimate
        grouped_files = {}
        for csv_file in csv_files:
            if "_file_" not in os.path.basename(csv_file):
                continue

            details = extract_file_details(csv_file)
            distribution = details["distribution_name"] or "unknown"
            size = details["size"] or "unknown"
            params = details["params"] or "default"
            mi_estimate = None

            for estimate, column_name in mi_estimate_map.items():
                if column_name in csv_file and column_name in selected_columns:
                    mi_estimate = column_name
                    break

            if mi_estimate:
                key = (distribution, size, params, mi_estimate)
                if key not in grouped_files:
                    grouped_files[key] = []
                grouped_files[key].append(csv_file)

        logging.info(f"Grouped files by distribution, size, params, and MI estimate.")

        # Process each group
        for (distribution, size, params, mi_estimate), files in grouped_files.items():
            # Build the relative output path for each group
            if distribution == "unknown" and size == "unknown" and params == "default":
                relative_output_path = ""
                output_csv_name = f"{mi_estimate}.csv"  # Simplified file name if all are unknown
            else:
                relative_output_path = os.path.join(
                    distribution if distribution != "unknown" else "",
                    f"size_{size}" if size != "unknown" else "",
                    f"params_{params}" if params != "default" else ""
                )
                # Exclude 'unknown' or 'default' from file name if not present
                output_csv_name = f"{mi_estimate}_{distribution}_N{size}_{params}.csv"
                output_csv_name = output_csv_name.replace("_unknown", "").replace("_default", "").replace("Nunknown", "").replace("params_default", "")

            full_output_path = os.path.join(output_dir, relative_output_path)
            os.makedirs(full_output_path, exist_ok=True)
            output_file = os.path.join(full_output_path, output_csv_name)

            logging.info(f"Processing group: {distribution}, {size}, {params}, {mi_estimate}")

            aggregated_data = {}
            if os.path.exists(output_file):
                with open(output_file, mode="r") as existing_csv:
                    reader = csv.reader(existing_csv)
                    next(reader)
                    for row in reader:
                        k = int(row[0])
                        mean_value = float(row[1])
                        sigma_value = float(row[2])
                        aggregated_data[k] = {"mean": mean_value, "sigma": sigma_value, "values": []}

            # Aggregating data by k
            data_by_k = {}
            for file in files:
                with open(file, mode="r") as csv_file:
                    reader = csv.reader(csv_file)
                    next(reader)
                    for row in reader:
                        k = int(row[0])
                        value = float(row[1])
                        if k not in k_values:  # Skip values not in specified k_values
                            continue
                        if k not in data_by_k:
                            data_by_k[k] = []
                        data_by_k[k].append(value)

            # Update aggregated data with new values
            for k, values in data_by_k.items():
                if k not in aggregated_data:
                    aggregated_data[k] = {"mean": 0, "sigma": 0, "values": []}
                aggregated_data[k]["values"].extend(values)

            # Prepare final aggregated data
            final_data = [["k", f"mean_{mi_estimate}", f"sigma_{mi_estimate}"]]
            for k, stats in sorted(aggregated_data.items()):
                all_values = stats["values"]
                if all_values:  # Only recompute for k in specified range or updated data
                    mean_value = sum(all_values) / len(all_values)
                    sigma_value = (sum((x - mean_value) ** 2 for x in all_values) / len(all_values)) ** 0.5
                else:  # Keep existing values for k outside the range
                    mean_value = stats["mean"]
                    sigma_value = stats["sigma"]
                final_data.append([k, mean_value, sigma_value])

            # Write the final aggregated results
            with open(output_file, mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(final_data)

            logging.info(f"Updated summary results saved to: {output_file}")

    except Exception as e:
        logging.error(f"Error during aggregation or update: {e}")