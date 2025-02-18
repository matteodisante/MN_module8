import os
import sys
import numpy as np
import logging
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './utils/')))
from mutual_information_utils import process_and_save_mi_table
from interface_utils import navigate_directories, setup_logger
from io_utils import save_transformed_file


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './core/')))
from mutual_information_1 import mutual_information_1
from mutual_information_1_entropies_sum import mutual_information_1_entropies_sum
from mutual_information_binning import mutual_information_binning_adaptive 






def parse_k_values(k_input):
    """
    Parses a string of k values into a list of integers.

    :param k_input: String representing k values (e.g., "1-15,17,30-35").
    :return: A list of integers representing the parsed k values.
    """
    k_values = set()  # Use a set to avoid duplicates

    # Split the input by commas to process ranges and individual values
    for part in k_input.split(","):
        part = part.strip()
        if "-" in part:
            try:
                # Handle ranges (e.g., "1-15")
                start, end = map(int, part.split("-"))
                k_values.update(range(start, end + 1))  # Add all values in the range
            except ValueError:
                print(f"Invalid range: {part}. Skipping.")
        else:
            try:
                # Handle individual values
                k_values.add(int(part))
            except ValueError:
                print(f"Invalid value: {part}. Skipping.")

    return sorted(k_values)  # Return a sorted list of k values






def read_existing_values(file_path, key_label):
    """
    Reads existing values from a file and returns them as a dictionary.
    
    For mi_1 and mi_sum, it expects a header in the format "<key_label> mi...".  
    For mi_binning (when key_label == "bins_asked_per_axis"), it expects the header:
    "bins_asked_per_axis bins_x bins_y total_cells non_empty_cells mi_binning".  
    In this case, it reads the first column (bins_asked_per_axis) from each line.
    
    :param file_path: Path to the file to read.
    :param key_label: The key label to use ("k" or "bins_asked_per_axis").
    :return: A dictionary of existing keys.
    """
    existing_values = {}
    if not os.path.exists(file_path):
        logging.info(f"File not found: {file_path}. No values exist.")
        return existing_values

    with open(file_path, 'r') as file:
        header = file.readline().strip()
        if key_label == "bins_asked_per_axis":
            if not header.startswith("bins_asked_per_axis"):
                logging.warning(f"Unexpected header format in {file_path}: {header}")
                return {}
            # Read the first column from each line (the already calculated bins_asked_per_axis)
            for line in file:
                try:
                    parts = line.strip().split()
                    key = int(parts[0])
                    existing_values[key] = True  # The value is not used, only the key matters
                except Exception as e:
                    logging.error(f"Malformed line in {file_path}: {line.strip()}")
            return existing_values
        else:
            if not header.startswith(f"{key_label} mi"):
                logging.warning(f"Unexpected header format in {file_path}: {header}")
                return {}
            for line in file:
                try:
                    key, mi = line.strip().split()[:2]
                    existing_values[int(key)] = float(mi)
                except Exception as e:
                    logging.error(f"Malformed line in {file_path}: {line.strip()}")
            return existing_values



def save_values_to_file(file_path, keys, mi_results, key_label, mi_label):
    """
    Saves the values for mi_1 and mi_sum in a file, with two columns (key and mi).

    :param file_path: Path to the output file.
    :param keys: List of keys (k or bins_number) to save.
    :param mi_results: List of mutual information values corresponding to the keys.
    :param key_label: The label for the key column in the file (e.g., "k" or "bin_size").
    :param mi_label: The label for the MI column in the file (e.g., "mi_1", "mi_sum", "mi_binning").
    """
    existing_data = read_existing_values(file_path, key_label)
    for key, mi in zip(keys, mi_results):
        existing_data[key] = mi 

    sorted_data = sorted(existing_data.items())

    with open(file_path, 'w') as file:
        file.write(f"{key_label} {mi_label}\n")
        for key, mi in sorted_data:
            file.write(f"{key} {mi}\n")
    print(f"Created/updated file saved to: {file_path}")




def save_mi_binning_values_to_file(file_path, results_list):
    """
    Save the results of mutual_information_binning_adaptive to files with the following column structure:
    bins_asked_per_axis, bins_x, bins_y, total_cells, non_empty_cells, mi_binning.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        # The header is setted accordingly to the wanted output datafile stucture
        f.write("bins_asked_per_axis bins_x bins_y total_cells non_empty_cells mi_binning\n")
        for row in results_list:
            # Each row is a tuple: (bins_asked, bins_x, bins_y, total_cells, non_empty_cells, mi)
            f.write(" ".join(map(str, row)) + "\n")
    print(f"Created/updated file saved to: {file_path}")




def get_output_file_path(original_file_path, mi_function):
    """
    Generates the corresponding output file path for a given input file and MI function.

    :param original_file_path: Path to the input file.
    :param mi_function: The mutual information function being used.
    :return: Path to the corresponding output file.
    """
    base_output_dir = "data/mi_numerical_results"

    if mi_function == mutual_information_1:
        result_subdir = "mi_1"
    elif mi_function == mutual_information_1_entropies_sum:
        result_subdir = "mi_sum"
    elif mi_function == mutual_information_binning_adaptive:
        result_subdir = "mi_binning"
    else:
        raise ValueError(f"Unsupported MI function: {mi_function}")

    # Construct the relative path
    relative_path = os.path.relpath(original_file_path, start="data/synthetic_data")
    output_file_path = os.path.join(base_output_dir, result_subdir, relative_path)
    return output_file_path




def calculate_missing_values_for_multiple_files(files, keys, mi_function, key_label="k"):
    """
    Handles the calculation of missing values (e.g., k or bin_size) for multiple files, avoiding redundant prompts.

    :param files: List of file paths.
    :param keys: List of keys to check and calculate (e.g., k values or bin sizes).
    :param mi_function: Mutual information function to use for calculation.
    :param key_label: Label for the keys (e.g., 'k' or 'bin_size').
    """
    global_decision = None
    summary = []

    # Check and summarize existing values for each file
    logging.info("Trying loading files") 
    for file in files:
        try:
            dataset = np.genfromtxt(file, delimiter=" ", skip_header=0)    
            if np.isnan(dataset).any():
                logging.warning(f"WARNING: file {file} contains NaN! Skipping.")
                continue
        except Exception as e:
            logging.error(f"Errore while reading {file}: {str(e)}")
            continue 

        output_file_path = get_output_file_path(file, mi_function)

        if os.path.exists(output_file_path):
            existing_values = read_existing_values(output_file_path, key_label=key_label)
        else:
            existing_values = {}

        missing_values = [key for key in keys if key not in existing_values]

        summary.append({
            "file": file,
            "output_file": output_file_path,
            "existing_values": sorted(existing_values.keys()),
            "missing_values": missing_values,
            "dataset": dataset,
        })
        
    logging.info("Files loaded")
    # Display summary for all files
    logging.info(f"\nSummary of {key_label} values for selected files:")
    for item in summary:
        logging.info(f"- File: {os.path.basename(item['file'])}")
        logging.info(f"Output file: {item['output_file']}")
        logging.info(f"Existing {key_label} values: {item['existing_values']}")
        logging.info(f"Missing {key_label} values: {item['missing_values']}")

    # If there are more than 3 files, ask for a global decision
    if len(files) > 3:
        while global_decision not in ['y', 'n']:
            global_decision = input(
                f"Do you want to calculate missing {key_label} values for all files? [y/n]: "
            ).strip().lower()

    # Process each file based on the decision
    for item in summary:
        file = item["file"]
        dataset = item["dataset"]
        output_file_path = item["output_file"]
        missing_values = item["missing_values"]

        if not missing_values:
            logging.info(f"All {key_label} values already calculated for {os.path.basename(file)}.")
            continue

        # If a global decision was made, use it; otherwise, ask for each file
        if global_decision == 'y':
            calculate_and_save_missing_values(dataset, output_file_path, missing_values, mi_function, key_label)
        elif global_decision == 'n' or global_decision is None:
            proceed = input(
                f"Do you want to calculate missing {key_label} values for {os.path.basename(file)}? [y/n]: "
            ).strip().lower()
            if proceed in ['y', 'yes']:
                calculate_and_save_missing_values(dataset, output_file_path, missing_values, mi_function, key_label)
            else:
                logging.info(f"Skipping {os.path.basename(file)}.")




def calculate_and_save_missing_values(dataset, output_file_path, missing_values, mi_function, key_label="k"):
    """
    Calculates and saves missing values (e.g., k or bin_size) for a single file.

    :param dataset: Data matrix as a NumPy array.
    :param output_file_path: Path to the output file.
    :param missing_values: List of missing values to calculate.
    :param mi_function: Mutual information function to use for calculation.
    :param key_label: Label for the keys (e.g., 'k' or 'bin_size').
    """
    # Determine the MI label based on the function
    if mi_function == mutual_information_1:
        mi_label = "mi_1"
    elif mi_function == mutual_information_1_entropies_sum:
        mi_label = "mi_sum"
    elif mi_function == mutual_information_binning_adaptive:
        mi_label = "mi_binning"
    else:
        raise ValueError(f"Unsupported MI function: {mi_function}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)


    # Branch handling mi-binning case
    if mi_function == mutual_information_binning_adaptive:
        results_list = []  # each element will be formatted as: (bins_asked_per_axis, bins_x, bins_y, total_cells, non_empty_cells, mi_binning)
        logging.info(f"Calculating missing {key_label} values for {os.path.basename(output_file_path)}: {missing_values}")
        for value in missing_values:
            try:
                mi, bins_x, bins_y, total_cells, non_empty_cells = mi_function(dataset, value)
                results_list.append((value, bins_x, bins_y, total_cells, non_empty_cells, mi))
            except Exception as e:
                logging.error(f"Error calculating MI for {output_file_path}, {key_label}={value}: {str(e)}")
                results_list.append((value, None, None, None, None, None))
        save_mi_binning_values_to_file(output_file_path, results_list)
        logging.info(f"Results saved successfully to: {output_file_path}")
    else:
        #Branch handling mi-1 and mi-sum cases
        mi_results = []
        non_empty_bins_results = []
        logging.info(f"Calculating missing {key_label} values for {os.path.basename(output_file_path)}: {missing_values}")
        for value in missing_values:
            try:
                mi, non_empty_cells = mi_function(dataset, value)
                mi_results.append(mi)
                non_empty_bins_results.append(non_empty_cells)
            except Exception as e:
                logging.error(f"Error calculating MI for {output_file_path}, {key_label}={value}: {str(e)}")
                mi_results.append(None)
                non_empty_bins_results.append(None)
        save_values_to_file(output_file_path, missing_values, mi_results, key_label, mi_label)
        logging.info(f"Results saved successfully to: {output_file_path}")
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    print("Welcome to the Mutual Information Analysis Tool\n")
    setup_logger()

    base_output_dir = "data/."

    # Select MI estimation function
    mi_functions_map = {
        "1": mutual_information_1,
        "2": mutual_information_1_entropies_sum,
        "3": mutual_information_binning_adaptive,
    }

    selected_function = None
    while not selected_function:
        print("Select the MI estimation function to use:")
        print("1: mutual_information_1 (requires k values)")
        print("2: mutual_information_1_entropies_sum (requires k values)")
        print("3: mutual_information_binning_adaptive (requires bins number)")
        mi_input = input("Enter the number corresponding to the desired function: ").strip()
        if mi_input in mi_functions_map:
            selected_function = mi_functions_map[mi_input]
        else:
            print("Invalid selection. Please try again.")
    print(f"Function selected successfully: {selected_function.__name__}")

    # Get k values or bin sizes based on the selected function
    if selected_function in [mutual_information_1, mutual_information_1_entropies_sum]:
        # Get k values
        k_values = []
        while not k_values:
            print("Specify k values as a list of ranges and/or individual values (e.g., 1-15,17,30-35,45):")
            k_input = input("Enter k values: ").strip()
            k_values = parse_k_values(k_input)
            if not k_values:
                print("No valid k values provided. Please try again.")
        print("Valid k values:", k_values)
    elif selected_function == mutual_information_binning_adaptive:
        # Get bins number
        bins_number = []
        while not bins_number:
            print("Specify bins number as a list of individual values (e.g., 10,20,30):")
            bins_input = input("Enter bins number: ").strip()
            bins_number = [int(value.strip()) for value in bins_input.split(",") if value.strip().isdigit()]
            if not bins_number:
                print("No valid bins number provided. Please try again.")
        print("Valid bins number:", bins_number)


    # Ask the user which type of data files to process and filter di conseguenza
    selected_files = []
    while not selected_files:
        print("Select the type of data files to process:")
        print("1: Linear data files (.txt) (excluding files containing '_log')")
        print("2: Log data files (_log.txt)")
        file_choice = input("Enter 1 or 2: ").strip()
        
        if file_choice == "1":
            # Prendi tutti i file che terminano con .txt, poi escludi quelli che contengono '_log'
            all_files = navigate_directories(start_path='.', multi_select=True, file_extension=".txt")
            selected_files = [f for f in all_files if "_log" not in os.path.basename(f)]
        elif file_choice == "2":
            # Prendi i file che terminano con _log.txt
            selected_files = navigate_directories(start_path='.', multi_select=True, file_extension="_log.txt")
        else:
            print("Invalid selection. Please try again.")
    
        if not selected_files:
            print("No files selected for the chosen option. Please try again.")

    # Process files based on the selected function
    if selected_function in [mutual_information_1, mutual_information_1_entropies_sum]:
        calculate_missing_values_for_multiple_files(
            selected_files, k_values, selected_function, key_label="k"
        )
    elif selected_function == mutual_information_binning_adaptive:
        calculate_missing_values_for_multiple_files(
            selected_files, bins_number, selected_function, key_label="bins_number"
        )

    print("Mutual Information Analysis completed successfully!")








