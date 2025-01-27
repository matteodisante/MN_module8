import os
import numpy as np


from utils.mutual_information_utils import process_and_save_mi_table
from utils.interface_utils import navigate_directories, setup_logging
from utils.io_utils import save_transformed_file


from core.mutual_information_1 import mutual_information_1
from core.mutual_information_1_entropies_sum import mutual_information_1_entropies_sum
from core.mutual_information_binning import mutual_information_binning_adaptive





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
    Reads existing values (k or bin_size) from the file and returns them as a dictionary.

    :param file_path: Path to the file containing existing values.
    :param key_label: The label for the key column in the file (e.g., "k" or "bin_size").
    :return: A dictionary where keys are values and values are corresponding mi values.
    """
    existing_values = {}
    if not os.path.exists(file_path):
        return existing_values  # Return an empty dictionary if the file doesn't exist

    with open(file_path, 'r') as file:
        header = file.readline().strip()
        if not header.startswith(f"{key_label} mi"):
            print(f"Unexpected header in file {file_path}: {header}")
            return {}

        for line in file:
            try:
                key, mi = line.strip().split()
                existing_values[int(key)] = float(mi)
            except ValueError:
                print(f"Malformed line in file {file_path}: {line.strip()}")

    return existing_values




def save_values_to_file(file_path, keys, mi_results, key_label, mi_label):
    """
    Updates the file with new values (k or bin_size) while preserving existing values.

    :param file_path: Path to the output file.
    :param keys: List of keys (k or bin_size) to save.
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
    print(f"Updated file saved to: {file_path}")




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
    for file in files:
        dataset = np.loadtxt(file)
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

    # Display summary for all files
    print(f"\nSummary of {key_label} values for selected files:")
    for item in summary:
        print(f"- File: {os.path.basename(item['file'])}")
        print(f"  Output file: {item['output_file']}")
        print(f"  Existing {key_label} values: {item['existing_values']}")
        print(f"  Missing {key_label} values: {item['missing_values']}")
    print()

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
            print(f"All {key_label} values already calculated for {os.path.basename(file)}.")
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
                print(f"Skipping {os.path.basename(file)}.")




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

    # Calculate the missing values
    print(f"Calculating missing {key_label} values for {os.path.basename(output_file_path)}: {missing_values}")
    mi_results = [mi_function(dataset, value) for value in missing_values]

    # Save the results
    save_values_to_file(output_file_path, missing_values, mi_results, key_label, mi_label)
    print(f"Results saved to: {output_file_path}")

    
    
    
    
    
    
    
if __name__ == "__main__":
    print("Welcome to the Mutual Information Analysis Tool\n")
    setup_logging()

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
        print("3: mutual_information_binning_adaptive (requires bin sizes)")
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

    # Select files
    selected_files = navigate_directories(start_path='.', multi_select=True, file_extension=".txt")
    if not selected_files:
        print("No files selected. Exiting.")
        exit()

    # Process files based on the selected function
    if selected_function in [mutual_information_1, mutual_information_1_entropies_sum]:
        calculate_missing_values_for_multiple_files(
            selected_files, k_values, selected_function, key_label="k"
        )
    elif selected_function == mutual_information_binning_adaptive:
        calculate_missing_values_for_multiple_files(
            selected_files, bin_sizes, selected_function, key_label="bins_number"
        )

    print("Mutual Information Analysis completed successfully!")








