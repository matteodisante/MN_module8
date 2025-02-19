import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import inspect

#from utils.plot_utils import load_config_distributions, get_user_choice, select_files_by_extension, find_matching_files, \
 #   extract_parameters_from_filename, extract_parameters_from_paths, filter_files_by_parameters, extract_k_or_bins_values_from_files
from utils.config_utils import load_config
from utils.math_utils import circular_mi_theoretical, gamma_exponential_mi_theoretical, \
    correlated_gaussian_rv_mi_theoretical, independent_exponential_rv_mi_theoretical, \
        independent_gaussian_rv_mi_theoretical, independent_uniform_rv_mi_theoretical, \
            ordered_wienman_exponential_mi_theoretical



def load_config_distributions():
    """Loads the config.json file from the same directory as the script and returns the available distributions."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    if not os.path.exists(config_path):
        print(f"Error: The config.json file was not found in {script_dir}.")
        exit(1)

    config = load_config(config_path)
    return [dist["name"] for dist in config["distributions"]]

def get_user_choice(options, prompt):
    """Asks the user to choose a valid option."""
    while True:
        print(prompt)
        for idx, option in enumerate(options, 1):
            print(f"{idx}. {option}")
        choice = input("Enter the corresponding number: ")
        
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(options):
                return options[choice - 1]
        print("Invalid choice. Please try again.")

def get_user_choices(options, prompt, multiple=False):
    """Asks the user to choose one or more valid options."""
    while True:
        print(prompt)
        for idx, option in enumerate(options, 1):
            print(f"{idx}. {option}")
        
        if multiple:
            choice_input = input("Enter the corresponding numbers separated by commas (e.g., 1, 3): ")
            # Parse the input into a list of integers
            try:
                choices = [int(x.strip()) for x in choice_input.split(',')]
                if all(1 <= choice <= len(options) for choice in choices):
                    return [options[choice - 1] for choice in choices]
                else:
                    print("Some choices are out of range. Please try again.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
        else:
            choice = input("Enter the corresponding number: ")
            if choice.isdigit():
                choice = int(choice)
                if 1 <= choice <= len(options):
                    return options[choice - 1]
            print("Invalid choice. Please try again.")


def select_files_by_extension(start_path=".", file_extension=".txt"):
    """Selects all files with a specified extension from the given directory and its subdirectories."""
    selected_paths = []

    # Walk through all directories and files starting from start_path
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.endswith(file_extension):
                selected_paths.append(os.path.join(root, file))
    return selected_paths

def find_matching_files(directory, distribution_name, mi_estimate, log_transformed=False):
    """Searches for .txt files in the specified folder and selects those matching the 
    distribution and estimate provided."""
    all_files = select_files_by_extension(directory, file_extension=".txt")
    pattern = re.compile(rf"summary_{distribution_name}_(.*?)_size_\d+_{mi_estimate}.txt")
    if log_transformed:
        pattern = re.compile(rf"summary_{distribution_name}_(.*?)_size_\d+_log_transformed_{mi_estimate}.txt")
    matching_files = [file for file in all_files if pattern.match(os.path.basename(file))]
    return matching_files


# Extracts the {params_values} string from the filename
def extract_parameters_from_filename(filename, distribution_name, mi_estimate, log_transformed=False):
    """
    Extracts the parameter string from the filename, given the format:
    'summary_{distribution_name}_{params_values}_size_{value_size}_{mi_estimate}.txt'
    """
    pattern = rf"summary_{distribution_name}_(.*?)_size_\d+_{mi_estimate}.txt"
    if log_transformed:
        pattern = rf"summary_{distribution_name}_(.*?)_size_\d+_log_transformed_{mi_estimate}.txt"
    match = re.match(pattern, filename)
    return match.group(1) if match else None


def extract_parameters_from_paths(paths, distribution_name, mi_estimate, log_transformed=False):
    """
    Extracts parameters and their values from all the files in the paths list and
    creates a dictionary associating each file with its extracted parameters.
    """
    # Two dictionaries, one for parameter-value, one for file-(parameter-value)
    all_parameters = {}
    extracted_params_per_file = {}

    for path in paths:
        param_string = extract_parameters_from_filename(os.path.basename(path), distribution_name, mi_estimate, log_transformed)
        if param_string:
            param_pairs = param_string.split('_')

            if len(param_pairs) % 2 != 0:
                print(f"Error in file: {path}. The number of parameter-value pairs is odd.")
                continue

            file_params = {param_pairs[i]: param_pairs[i + 1] for i in range(0, len(param_pairs), 2)}
            extracted_params_per_file[path] = file_params  # Saves the file's parameters

            for param_name, param_value in file_params.items():
                if param_name not in all_parameters:
                    all_parameters[param_name] = set()
                all_parameters[param_name].add(param_value)

    return all_parameters, extracted_params_per_file


def filter_files_by_parameters(extracted_params_per_file, selected_params):
    """
    Filters files based on the parameters selected by the user.

    :param extracted_params_per_file: Dictionary {file_path: {param_name: param_value}}.
    :param selected_params: Dictionary {param_name: param_value} selected by the user.
    :return: List of filtered files.
    """
    return [
        path for path, file_params in extracted_params_per_file.items()
        if all(file_params.get(param) == value for param, value in selected_params.items())
    ]


# Extracts the possible k values or bins from the files
def extract_k_or_bins_values_from_files(files):
    """Extracts k values or the number of bins from the first column of the files. The first row contains the header, 
    and the k values or the number of bins are in the first column starting from the second row."""
    x_values = set()
    
    for file in files:
        with open(file, 'r') as f:
            # Skip the first row (header)
            next(f)
            for line in f:
                # Split the line to separate the columns
                columns = line.strip().split()
                if columns:  # Check that the line is not empty
                    try:
                        x_value = int(columns[0])  # The first column contains the k value
                        x_values.add(x_value)
                    except ValueError:
                        # If the value in the first column is not an integer, ignore that line
                        continue
    
    # Return the sorted k values
    return sorted(x_values)





def plot_figure_4(files, distribution_name, mi_estimate, theoretical_mi, log_transformed):
    """
    Generates the plot for Figure 4 (one file for each N).
    """
    plt.figure(figsize=(8, 6))

    # Dictionary to store the data for each N
    data_dict = {}

    # Variable to store the parameter extracted from the first group of the regex
    extracted_param = None

    for file in files:
        # Create the appropriate regular expression for matching the file name
        if log_transformed:
            match = re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_log_transformed_({mi_estimate}).txt", os.path.basename(file))
        else:
            match = re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_({mi_estimate}).txt", os.path.basename(file))
        
        if match:
            N = int(match.group(2))

            # Extract the parameter from the first group
            extracted_param = match.group(1)  # This extracts the parameter from the filename

            # List to store valid data
            valid_data = []


            with open(file, 'r') as f:
                # Skip the header (first row)
                next(f)

                # Loop to read each line from the file
                for line in f:
                    # Strip any leading/trailing whitespace
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Convert the line into a float array (split by whitespace or tab)
                    row = np.fromstring(line, sep=' ')

                    if "binning" in mi_estimate.lower():
                        # Ensure the row has at least 7 columns (since you expect 7 values per row)
                        if row.size >= 7:
                            fifth_column_value = row[4]

                            # Check if it is not infinity or NaN
                            if not np.isinf(fifth_column_value) and not np.isnan(fifth_column_value):
                                valid_data.append(row)  # Add the row to the list if it's valid
                        else:
                            print(f"Skipping line due to insufficient columns: {line}")
                    else:
                        # Ensure the row has at least 4 columns (since you expect 4 values per row)
                        if row.size >= 4:
                            second_column_value = row[1]  # The second column (index 1)

                            # Check if the second column is not infinity or NaN
                            if not np.isinf(second_column_value) and not np.isnan(second_column_value):
                                valid_data.append(row)  # Add the row to the list if it's valid
                        else:
                            print(f"Skipping line due to insufficient columns: {line}")

            if "binning" in mi_estimate.lower():
                # Convert the valid data into a numpy array
                data_cleaned = np.array(valid_data).reshape(-1, 7)

                # Separate the columns
                first_column = data_cleaned[:, 0]
                means = data_cleaned[:, 4]
                sigmas = data_cleaned[:, 6]
                x_values = data_cleaned[:, 2]
                xlolims = data_cleaned[:, 1]
                xuplims = data_cleaned[:, 3]
            else:
                # Convert the valid data into a numpy array
                data_cleaned = np.array(valid_data).reshape(-1, 4)

                # Separate the columns
                x_values = data_cleaned[:, 0]
                means = data_cleaned[:, 1]
                sigmas = data_cleaned[:, 3]

            # If mi_estimate is "mi_binning", the first value represents bins_number
            # Otherwise, it represents k_vals
            x_vals = x_values / N

            if "binning" in mi_estimate.lower():
                # Store the data in the dictionary, using N as the key
                data_dict[N] = (x_vals, xlolims, xuplims, means, sigmas)
            else: 
                data_dict[N] = (x_vals, means, sigmas)

    # Sort the dictionary by N in ascending order
    sorted_N_values = sorted(data_dict.keys())

    # List to collect the legend labels
    legend_labels = []

    # Plot the data for each sorted N value
    for N in sorted_N_values:
        if "binning" in mi_estimate.lower():
            x_vals, xlolims, xuplims, means, sigmas = data_dict[N]
        else:
            x_vals, means, sigmas = data_dict[N]

        # Points with error bars of the same color
        plt.errorbar(x_vals, means, yerr=sigmas, xlolims=xlolims, xuplims=xuplims, linestyle='--', fmt='.', capsize=1, alpha=0.7, label=f'N={N}')
        legend_labels.append(f'N={N}')

    # Customize the plot
    plt.xscale('log')
    plt.xlabel(r"num $_{\mathrm{bins}}$ /N" if mi_estimate == "mi_binning" else "k/N", fontsize=15)

    # Plot the theoretical line without including it in the legend
    plt.axhline(y=theoretical_mi, color='r', linestyle='-', linewidth=1, label='_nolegend_')

    # Map to transform mi_estimate to the subscript format
    subscript_map = {
        "mi_1": "1",
        "mi_sum": "sum",
        "mi_binning": "binning"
    }

    # Get the correct subscript
    subscript = subscript_map.get(mi_estimate, "")

    # Set the label with the subscript not in italics
    plt.ylabel(r"I$_{\mathrm{" + subscript + r"}}$", fontsize=15)

    # Format the title: Add the distribution name and the extracted parameter
    formatted_distribution_name = distribution_name.replace('_', r'\_')
    
    # Add the extracted parameter from the filename
    if extracted_param:
        title = r"$\mathrm{" + formatted_distribution_name + r"}$ (" + extracted_param + r")"
    else:
        title = r"$\mathrm{" + formatted_distribution_name + r"}$"

    # Set the title with distribution name and extracted parameter
    plt.title(title, fontsize=15)

    # Add the legend and other plot elements
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def process_figure_4(files, distribution_name, mi_estimate, log_transformed):
    """
    Manages parameter selection for Figure 4, filters the corresponding files, and makes the plot.
    """
    # Gets the parameter-value dictionary and the one for associating paths
    all_parameters, extracted_params_per_file = extract_parameters_from_paths(files, distribution_name, mi_estimate, log_transformed)
    # Defines the selected parameters dictionary and asks the user to select values for 
    # the parameters for the plot
    selected_params = {}

    for param_name, param_values in all_parameters.items():
        chosen_value = get_user_choice(list(param_values), f"Choose the value for the parameter {param_name}:")
        selected_params[param_name] = chosen_value

    # Filters the files with the selected parameter values from the user
    filtered_files = filter_files_by_parameters(extracted_params_per_file, selected_params)

    print("Selected files after filtering:")
    for file in filtered_files:
        print(file)

    # Calculate the theoretical mutual information
    theoretical_mi_function = globals().get(f"{distribution_name}_mi_theoretical")
    if theoretical_mi_function:
        required_params = inspect.signature(theoretical_mi_function).parameters
        filtered_params = {key: float(value) for key, value in selected_params.items() if key in required_params}
        theoretical_mi = theoretical_mi_function(**filtered_params)
        print(f"Theoretical mutual information for {distribution_name}: {theoretical_mi}")
    else:
        print(f"No theoretical function found for {distribution_name}.")
        theoretical_mi = None

    # Step 1: Ask the user whether they want to choose one or more N, or plot all available N values
    if not filtered_files:
        print("No files available after filtering.")
        return

    # Step 2: Extract all unique values of N from the filtered files
    N_values = set()
    for file in filtered_files:
        match = re.search(r"size_(\d+)", os.path.basename(file))
        if match:
            N_values.add(int(match.group(1)))

    if not N_values:
        print("No valid N values found in the filtered files.")
        return

    # Ask the user what they want to do before proceeding with N extraction
    print(f"Available N values: {sorted(N_values)}")
    choice = input("Do you want to (1) Choose one or more values of N or (2) Plot all available N values? (Enter 1 or 2): ").strip()

    if choice == '1':
            # Step 3: User selects one or more N values
            chosen_Ns = get_user_choices(list(N_values), f"Choose one or more values for N from the available options (comma separated):", multiple=True)

            # Filter the files again based on the chosen Ns
            final_filtered_files = [file for file in filtered_files if any(f"size_{chosen_N}_" in os.path.basename(file) for chosen_N in chosen_Ns)]
            print(f"Selected files for N={chosen_Ns}:")
            for file in final_filtered_files:
                print(file)

    elif choice == '2':
        # Step 4: Plot all N values (no filtering by N)
        final_filtered_files = filtered_files
        print("Plotting all available N values...")

    else:
        print("Invalid choice. Please enter 1 or 2.")
        return

    # Make the plot
    plot_figure_4(final_filtered_files, distribution_name, mi_estimate, theoretical_mi, log_transformed)



def process_figure_7_9(files, distribution_name, mi_estimate, figure, log_transformed):
    """
    Manages the parameter selection and the k or number of bins for Figures 7 and 9, filters the corresponding files, and plots.
    """
    # Determine whether to ask for k or bins
    if "binning" in mi_estimate.lower():
        k_bins_values = extract_k_or_bins_values_from_files(files)  # Extract bins
        param_type = "bins"
    else:
        k_bins_values = extract_k_or_bins_values_from_files(files)  # Extract k
        param_type = "k"

    if not k_bins_values:
        print(f"No {param_type} values found for {mi_estimate}. Exiting.")
        exit()

    # The user chooses the value of k or bins based on the MI method
    k_bins_choice = get_user_choice(k_bins_values, f"Choose the value of {param_type} to use for the plot (for {mi_estimate}):")


    # Extract parameters from the files
    all_parameters, extracted_params_per_file = extract_parameters_from_paths(files, distribution_name, mi_estimate, log_transformed)
    selected_param = get_user_choice(list(all_parameters.keys()), "Choose the parameter to analyze:")

    # Group files by the selected parameter's value
    grouped_files = {}
    for param_value in all_parameters[selected_param]:
        selected_files = filter_files_by_parameters(extracted_params_per_file, {selected_param: param_value})
        if selected_files:
            grouped_files[param_value] = selected_files



    plt.figure(figsize=(8, 6))

    # Sort the parameter values in ascending order
    sorted_param_values = sorted(grouped_files.keys())

    # Loop through the sorted parameter values
    for idx, param_value in enumerate(sorted_param_values):
        file_list = grouped_files[param_value]
        x_vals, y_vals, y_errs = [], [], []

        # Calculate theoretical mutual information
        theoretical_mi_function = globals().get(f"{distribution_name}_mi_theoretical")
        theoretical_mi = None

        if theoretical_mi_function:
            required_params = inspect.signature(theoretical_mi_function).parameters

            # Create the dictionary for required parameters
            filtered_params = {
                key: (float(param_value) if key == selected_param else float(next(iter(value))))
                for key, value in all_parameters.items() if key in required_params
            }

            # Calculate theoretical mutual information
            theoretical_mi = theoretical_mi_function(**filtered_params)
            print(f"Theoretical mutual information for {distribution_name}, {selected_param}={param_value}: {theoretical_mi}")

        if figure == "7":
            # If theoretical mutual information is zero, exclude from the legend
            if theoretical_mi is None or theoretical_mi == 0:
                print(f"Warning: theoretical mutual information is zero for {selected_param}={param_value}. Point excluded from the legend.")
                continue  # Skip the rest of the loop for this parameter value

            for file in file_list:
                # List to store valid data
                valid_data = []
                with open(file, 'r') as f:
                    # Skip the header (first row)
                    next(f)

                    # Loop to read each line from the file
                    for line in f:
                        # Strip any leading/trailing whitespace
                        line = line.strip()
                        
                        # Skip empty lines
                        if not line:
                            continue
                        
                        # Convert the line into a float array (split by whitespace or tab)
                        row = np.fromstring(line, sep=' ')

                        if "binning" in mi_estimate.lower():
                            # Ensure the row has at least 7 columns (since you expect 7 values per row)
                            if row.size >= 7:
                                fifth_column_value = row[4]

                                # Check if it is not infinity or NaN
                                if not np.isinf(fifth_column_value) and not np.isnan(fifth_column_value):
                                    valid_data.append(row)  # Add the row to the list if it's valid
                            else:
                                print(f"Skipping line due to insufficient columns: {line}")
                        else:
                            # Ensure the row has at least 4 columns (since you expect 4 values per row)
                            if row.size >= 4:
                                second_column_value = row[1]  # The second column (index 1)

                                # Check if the second column is not infinity or NaN
                                if not np.isinf(second_column_value) and not np.isnan(second_column_value):
                                    valid_data.append(row)  # Add the row to the list if it's valid
                            else:
                                print(f"Skipping line due to insufficient columns: {line}")

                if "binning" in mi_estimate.lower():
                    # Convert the valid data into a numpy array
                    data_cleaned = np.array(valid_data).reshape(-1, 7)

                    # Separate the columns
                    first_column = data_cleaned[:, 0]
                    means = data_cleaned[:, 4]
                    sigmas = data_cleaned[:, 6]
                    x_values = data_cleaned[:, 2]

                    for i in range(len(first_column)):
                        if int(first_column[i]) == k_bins_choice:
                            match = re.search(r"size_(\d+)", file)
                            if match:
                                N = int(match.group(1))
                                # If theoretical mutual information is valid, add points
                                x_vals.append(1/N)
                                y_vals.append(float(means[i]) / theoretical_mi)
                                y_errs.append(float(sigmas[i]) / np.abs(theoretical_mi))
                            else:
                                print(f"Warning: unable to extract N from file name {file}")

                else:
                    # Convert the valid data into a numpy array
                    data_cleaned = np.array(valid_data).reshape(-1, 4)

                    # Separate the columns
                    first_column = data_cleaned[:, 0]
                    means = data_cleaned[:, 1]
                    sigmas = data_cleaned[:, 3]

                    for i in range(len(first_column)):
                        if int(first_column[i]) == k_bins_choice:
                            match = re.search(r"size_(\d+)", file)
                            if match:
                                N = int(match.group(1))
                                # If theoretical mutual information is valid, add points
                                x_vals.append(1/N)
                                y_vals.append(float(means[i]) / theoretical_mi)
                                y_errs.append(float(sigmas[i]) / np.abs(theoretical_mi))
                            else:
                                print(f"Warning: unable to extract N from file name {file}")


        elif figure == "9":
            for file in file_list:
                    # List to store valid data
                    valid_data = []
                    with open(file, 'r') as f:
                        # Skip the header (first row)
                        next(f)

                        # Loop to read each line from the file
                        for line in f:
                            # Strip any leading/trailing whitespace
                            line = line.strip()
                            
                            # Skip empty lines
                            if not line:
                                continue
                            
                            # Convert the line into a float array (split by whitespace or tab)
                            row = np.fromstring(line, sep=' ')

                            if "binning" in mi_estimate.lower():
                                # Ensure the row has at least 7 columns (since you expect 7 values per row)
                                if row.size >= 7:
                                    fifth_column_value = row[4]

                                    # Check if it is not infinity or NaN
                                    if not np.isinf(fifth_column_value) and not np.isnan(fifth_column_value):
                                        valid_data.append(row)  # Add the row to the list if it's valid
                                else:
                                    print(f"Skipping line due to insufficient columns: {line}")
                            else:
                                # Ensure the row has at least 4 columns (since you expect 4 values per row)
                                if row.size >= 4:
                                    second_column_value = row[1]  # The second column (index 1)

                                    # Check if the second column is not infinity or NaN
                                    if not np.isinf(second_column_value) and not np.isnan(second_column_value):
                                        valid_data.append(row)  # Add the row to the list if it's valid
                                else:
                                    print(f"Skipping line due to insufficient columns: {line}")

                    if "binning" in mi_estimate.lower():
                        # Convert the valid data into a numpy array
                        data_cleaned = np.array(valid_data).reshape(-1, 7)

                        # Separate the columns
                        first_column = data_cleaned[:, 0]
                        means = data_cleaned[:, 4]
                        sigmas = data_cleaned[:, 6]
                        x_values = data_cleaned[:, 2]

                        for i in range(len(first_column)):
                            if int(first_column[i]) == k_bins_choice:
                                match = re.search(r"size_(\d+)", file)
                                if match:
                                    N = int(match.group(1))
                                    # If theoretical mutual information is valid, add points
                                    x_vals.append(1/N)
                                    y_vals.append(float(means[i]) - theoretical_mi)
                                    y_errs.append(float(sigmas[i]))
                                else:
                                    print(f"Warning: unable to extract N from file name {file}")

                    else:
                        # Convert the valid data into a numpy array
                        data_cleaned = np.array(valid_data).reshape(-1, 4)

                        # Separate the columns
                        first_column = data_cleaned[:, 0]
                        means = data_cleaned[:, 1]
                        sigmas = data_cleaned[:, 3]

                        for i in range(len(first_column)):
                            if int(first_column[i]) == k_bins_choice:
                                match = re.search(r"size_(\d+)", file)
                                if match:
                                    N = int(match.group(1))
                                    # If theoretical mutual information is valid, add points
                                    x_vals.append(1/N)
                                    y_vals.append(float(means[i]) - theoretical_mi)
                                    y_errs.append(float(sigmas[i]))
                                else:
                                    print(f"Warning: unable to extract N from file name {file}")


        # Sort the data by x
        sorted_indices = np.argsort(x_vals)
        x_vals = np.array(x_vals)[sorted_indices]
        y_vals = np.array(y_vals)[sorted_indices]
        y_errs = np.array(y_errs)[sorted_indices]

        # Plot with error bars using the same color
        plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='.', linestyle='--', capsize=1, alpha = 0.7, label=param_value)
    plt.xscale('log')
    plt.xlabel(f"1/N", fontsize=15)

    # Map to transform mi_estimate into the subscript format
    subscript_map = {
        "mi_1": "1",
        "mi_sum": "sum",
        "mi_binning": "binning"
    }

    # Get the correct subscript
    subscript = subscript_map.get(mi_estimate, "")

    # Set the label with the subscript not in italics
    if figure == "7":
        plt.ylabel(r"$\mathrm{I}_{\mathrm{" + subscript + r"}} / \mathrm{I}_{\mathrm{exact}}$", fontsize=15)
    else:
        if figure == "9":
            plt.ylabel(r"$\mathrm{I}_{\mathrm{" + subscript + r"}} - \mathrm{I}_{\mathrm{exact}}$", fontsize=15)

    plt.legend(title=selected_param, fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()






def process_figure_8(files, distribution_name, mi_estimate, log_transformed):
    """
    Manages the parameter selection for Figure 8, filters the corresponding files, and plots.
    """
    # Determine if the estimator uses k or bins
    k_or_bins_name = "bins" if mi_estimate == "mi_binning" else "k"
    k_or_bins_values = extract_k_or_bins_values_from_files(files)  # Extract available values
    
    if not k_or_bins_values:
        print(f"No {k_or_bins_name} values found in the files. Exiting.")
        exit()
    
    # User chooses the value of k or bins
    k_or_bins_choice = get_user_choice(k_or_bins_values, f"Choose the value of {k_or_bins_name} to use for the plot:")
    print(f"You chose {k_or_bins_name}: {k_or_bins_choice}")

    # Obtain the parameter-value dictionary and its association with the paths
    all_parameters, extracted_params_per_file = extract_parameters_from_paths(files, distribution_name, mi_estimate, log_transformed)
    
    # Define the selected parameters dictionary and ask the user to select the values for the plot
    selected_params = {}

    for param_name, param_values in all_parameters.items():
        chosen_value = get_user_choice(list(param_values), f"Choose the value for the parameter {param_name}:")
        selected_params[param_name] = chosen_value

    # Filter the files with the selected parameter values
    filtered_files = filter_files_by_parameters(extracted_params_per_file, selected_params)

    print("Files selected after filtering:")
    for file in filtered_files:
        print(file)

    # List to gather N values and the corresponding standard deviations for the chosen k
    N_values = []
    sigma_values = []

    # Loop through the filtered files to extract the standard deviation for the chosen k
    for file in filtered_files:
        match = re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_({mi_estimate}).txt", os.path.basename(file))
        if log_transformed:
            match = re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_log_transformed_({mi_estimate}).txt", os.path.basename(file))
        if match:
            N = int(match.group(2))  # Extract N from the file name

            # List to store valid data
            valid_data = []

            with open(file, 'r') as f:
                # Skip the header (first row)
                next(f)

                # Loop to read each line from the file
                for line in f:
                    # Strip any leading/trailing whitespace
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Convert the line into a float array (split by whitespace or tab)
                    row = np.fromstring(line, sep=' ')


                    if "binning" in mi_estimate.lower():
                        # Ensure the row has at least 7 columns (since you expect 7 values per row)
                        if row.size >= 7:
                            fifth_column_value = row[4]

                            # Check if it is not infinity or NaN
                            if not np.isinf(fifth_column_value) and not np.isnan(fifth_column_value):
                                valid_data.append(row)  # Add the row to the list if it's valid
                        else:
                            print(f"Skipping line due to insufficient columns: {line}")
                    else:
                        # Ensure the row has at least 4 columns (since you expect 4 values per row)
                        if row.size >= 4:
                            second_column_value = row[1]  # The second column (index 1)

                            # Check if the second column is not infinity or NaN
                            if not np.isinf(second_column_value) and not np.isnan(second_column_value):
                                valid_data.append(row)  # Add the row to the list if it's valid
                        else:
                            print(f"Skipping line due to insufficient columns: {line}")

            if "binning" in mi_estimate.lower():
                # Convert the valid data into a numpy array
                data_cleaned = np.array(valid_data).reshape(-1, 7)

                # Separate the columns
                x_vals = data_cleaned[:, 0]
                sigmas = data_cleaned[:, 6]
            else:
                # Convert the valid data into a numpy array
                data_cleaned = np.array(valid_data).reshape(-1, 4)

                # Separate the columns
                x_vals = data_cleaned[:, 0]
                sigmas = data_cleaned[:, 3]

            # Look for the chosen k value in the data and get the corresponding standard deviation
            if k_or_bins_choice in x_vals:
                x_index = np.where(x_vals == k_or_bins_choice)[0][0]  # Find the index of the chosen k
                sigma_values.append(sigmas[x_index])  # Add the standard deviation
                N_values.append(N)  # Add N

    # Check if sigma values were found for the chosen k
    if not sigma_values:
        print(f"No standard deviation values found for {k_or_bins_name} = {k_or_bins_choice}.")
        return

    # Sort the values based on N
    sorted_indices = np.argsort(N_values)
    N_values_sorted = np.array(N_values)[sorted_indices]
    sigma_values_sorted = np.array(sigma_values)[sorted_indices]

    # Plot the standard deviation as a function of N
    plt.figure(figsize=(8, 6))
    plt.plot(N_values_sorted, sigma_values_sorted, linestyle='--', marker='.', color='b', label=f'Sigma for {k_or_bins_name}={k_or_bins_choice}')
    plt.xscale('log')
    plt.xlabel('N', fontsize=11)
    plt.ylabel('Standard Deviation', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



def process_figure_20(files, distribution_name, mi_estimators, log_transformed):
    """
    Plots Figure 20 by selecting a single parameter value and showing multiple MI estimators.
    """
    # Gets the parameter-value dictionary and the one for associating paths
    all_parameters = []
    extracted_params_per_file = []
    for mi in mi_estimators:
        params, extracted_params = extract_parameters_from_paths(files, distribution_name, mi, log_transformed)
        all_parameters.append(params)
        extracted_params_per_file.append(extracted_params)

    selected_params = {}

    for param_name, param_values in all_parameters[0].items():
        chosen_value = get_user_choice(list(param_values), f"Choose the value for the parameter {param_name}:")
        selected_params[param_name] = chosen_value

    filtered_files = []
    # Filters the files with the selected parameter values from the user
    for i, mi in enumerate(mi_estimators):
        filtered = filter_files_by_parameters(extracted_params_per_file[i], selected_params)
        filtered_files.append(filtered)

    # Theoretical MI
    theoretical_mi_function = globals().get(f"{distribution_name}_mi_theoretical")
    theoretical_mi = None
    if theoretical_mi_function:
        required_params = inspect.signature(theoretical_mi_function).parameters
        filtered_params = {
            key: float(selected_params[key]) if key in selected_params else float(next(iter(value)))
            for key, value in all_parameters[0].items() if key in required_params
        }
        theoretical_mi = theoretical_mi_function(**filtered_params)

    if theoretical_mi is None or theoretical_mi == 0:
        print(f"Warning: theoretical mutual information is zero for selected parameter values.")
        
        # Richiesta di input all'utente se MI è zero o None
        print("Please provide the missing parameters:")
        for param_name, param_values in all_parameters[0].items():
            if param_name not in selected_params:
                chosen_value = get_user_choice(list(param_values), f"Enter the value for the parameter {param_name}:")
                selected_params[param_name] = chosen_value

        # Ricalcola MI teorica con i nuovi parametri
        filtered_params = {
            key: float(selected_params[key]) for key in selected_params
        }
        theoretical_mi = theoretical_mi_function(**filtered_params)



    plt.figure(figsize=(8, 6))

    for idx, mi_estimate in enumerate(mi_estimators):
        # Determine whether to ask for k or bins
        if "binning" in mi_estimate.lower():
            k_bins_values = extract_k_or_bins_values_from_files(filtered_files[idx])  # Extract bins
            param_type = "bins"
        else:
            k_bins_values = extract_k_or_bins_values_from_files(filtered_files[idx])  # Extract k
            param_type = "k"

        if not k_bins_values:
            print(f"No {param_type} values found for {mi_estimate}. Exiting.")
            exit()

        # The user chooses the value of k or bins based on the MI method
        k_bins_choice = get_user_choice(k_bins_values, f"Choose the value of {param_type} to use for the plot (for {mi_estimate}):")

        x_vals, y_vals, y_errs, xmin, xmax = [], [], [], [], []
        
        # Process files and extract data
        for file in filtered_files[idx]:
            # List to store valid data
            valid_data = []

            with open(file, 'r') as f:
                # Skip the header (first row)
                next(f)

                # Loop to read each line from the file
                for line in f:
                    # Strip any leading/trailing whitespace
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Convert the line into a float array (split by whitespace or tab)
                    row = np.fromstring(line, sep=' ')

                    if "binning" in mi_estimate.lower():
                        # Ensure the row has at least 7 columns (since you expect 7 values per row)
                        if row.size >= 7:
                            fifth_column_value = row[4]

                            # Check if it is not infinity or NaN
                            if not np.isinf(fifth_column_value) and not np.isnan(fifth_column_value):
                                valid_data.append(row)  # Add the row to the list if it's valid
                        else:
                            print(f"Skipping line due to insufficient columns: {line}")
                    else:
                        # Ensure the row has at least 4 columns (since you expect 4 values per row)
                        if row.size >= 4:
                            second_column_value = row[1]  # The second column (index 1)

                            # Check if the second column is not infinity or NaN
                            if not np.isinf(second_column_value) and not np.isnan(second_column_value):
                                valid_data.append(row)  # Add the row to the list if it's valid
                        else:
                            print(f"Skipping line due to insufficient columns: {line}")

            if "binning" in mi_estimate.lower():
                # Convert the valid data into a numpy array
                data_cleaned = np.array(valid_data).reshape(-1, 7)

                # Separate the columns
                first_column = data_cleaned[:, 0]
                means = data_cleaned[:, 4]
                sigmas = data_cleaned[:, 6]
                xlolims = data_cleaned[:, 1]
                xuplims = data_cleaned[:, 3]
                for i in range(len(first_column)):
                    if int(first_column[i]) == k_bins_choice:
                        match = re.search(r"size_(\d+)", file)
                        if match:
                            N = int(match.group(1))
                            x_vals.append(1/N)
                            y_vals.append(float(means[i]) / theoretical_mi)
                            y_errs.append(float(sigmas[i]) / np.abs(theoretical_mi))
                            xmin.append(xlolims)
                            xmax.append(xuplims)
                        else:
                            print(f"Warning: unable to extract N from file name {file}")
            else:
                # Convert the valid data into a numpy array
                data_cleaned = np.array(valid_data).reshape(-1, 4)

                # Separate the columns
                first_column = data_cleaned[:, 0]
                means = data_cleaned[:, 1]
                sigmas = data_cleaned[:, 3]

                for i in range(len(first_column)):
                    if int(first_column[i]) == k_bins_choice:
                        match = re.search(r"size_(\d+)", file)
                        if match:
                            N = int(match.group(1))
                            x_vals.append(1/N)
                            y_vals.append(float(means[i]) / theoretical_mi)
                            y_errs.append(float(sigmas[i]) / np.abs(theoretical_mi))
                        else:
                            print(f"Warning: unable to extract N from file name {file}")

        if "binning" in mi_estimate.lower():
            # Sort data by x (N)
            sorted_indices = np.argsort(x_vals)
            x_vals = np.array(x_vals)[sorted_indices]
            y_vals = np.array(y_vals)[sorted_indices]
            y_errs = np.array(y_errs)[sorted_indices]
            xmin = np.array(xmin)[sorted_indices]
            xmax = np.array(xmax)[sorted_indices]

            # Assign color and plot
            plt.errorbar(x_vals, y_vals, yerr=y_errs, xlolims=xlolims, xuplims=xuplims, fmt='.', linestyle='--', capsize=1, label=mi_estimate)
        else:
            # Sort data by x (N)
            sorted_indices = np.argsort(x_vals)
            x_vals = np.array(x_vals)[sorted_indices]
            y_vals = np.array(y_vals)[sorted_indices]
            y_errs = np.array(y_errs)[sorted_indices]

            # Assign color and plot
            plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='.', linestyle='--', capsize=1, label=mi_estimate)

    # Format the title: Add the distribution name, the extracted parameter, and selected parameters
    formatted_distribution_name = distribution_name.replace('_', r'\_')
    
    # Add the extracted parameter from the filename
    title = r"$\mathrm{" + formatted_distribution_name + r"}$"

    # Add selected parameters to the title
    selected_params_str = " (" + ", ".join([f"{param_name}={param_value}" for param_name, param_value in selected_params.items()]) + ")"
    title += selected_params_str

    # Set the title with distribution name and extracted parameter
    plt.title(title, fontsize=15)
    plt.xscale('log')
    plt.xlabel(r"1/N", fontsize=15)
    plt.ylabel(r"$\mathrm{I}_{\mathrm{est}} / \mathrm{I}_{\mathrm{exact}}$", fontsize=15)
    plt.legend(title="MI Estimators", fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



def process_figure_21(files, distribution_name, mi_estimators, log_transformed):
    """
    Plots Figure 21 by showing multiple MI estimators with a fixed N value and a parameter chosen for the x-axis.
    """
    # Gets the parameter-value dictionary and the one for associating paths
    all_parameters = []
    extracted_params_per_file = []
    for mi in mi_estimators:
        params, extracted_params = extract_parameters_from_paths(files, distribution_name, mi, log_transformed)
        all_parameters.append(params)
        extracted_params_per_file.append(extracted_params)

    # Choose the parameter to analyze
    selected_param = get_user_choice(list(all_parameters[0].keys()), "Choose the parameter to analyze:")

    # Calculate theoretical MI for each value of the selected parameter
    theoretical_mi_values = {}
    theoretical_mi_function = globals().get(f"{distribution_name}_mi_theoretical")
    
    if theoretical_mi_function:
        for param_value in all_parameters[0][selected_param]:
            required_params = inspect.signature(theoretical_mi_function).parameters
            filtered_params = {
                key: float(param_value) if key == selected_param else float(next(iter(value)))
                for key, value in all_parameters[0].items() if key in required_params
            }
            theoretical_mi_values[param_value] = theoretical_mi_function(**filtered_params)

    # Get all possible values of N from the file names
    available_sizes = set()
    for file in files:
        match = re.search(r"size_(\d+)", os.path.basename(file))
        if match:
            available_sizes.add(int(match.group(1)))
    
    available_sizes = sorted(available_sizes)

    # Ask the user to choose a value for N
    N_choice = get_user_choice(available_sizes, "Choose a value for N:")
    print(f"N_choice selected: {N_choice}")

    # Group the files by the selected parameter and N value
    filtered_files = [{} for _ in range(len(mi_estimators))]
    for i, mi_estimate in enumerate(mi_estimators):
        for param_value in all_parameters[0][selected_param]:
            selected_files = filter_files_by_parameters(extracted_params_per_file[i], {selected_param: param_value})
            selected_files = [file for file in selected_files if f"size_{N_choice}_" in os.path.basename(file)]
            if selected_files:
                filtered_files[i][param_value] = selected_files

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Select any value from the set (get a value from the set)
    param_value_choice = next(iter(all_parameters[0][selected_param]))

    # Iterate over each MI estimator
    for idx, mi_estimate in enumerate(mi_estimators):
        # Get the files corresponding to the current estimator and the selected parameter value
        estimator_files = filtered_files[idx].get(param_value_choice, [])

        # Determine whether to ask for k or bins
        if "binning" in mi_estimate.lower():
            k_bins_values = extract_k_or_bins_values_from_files(estimator_files)  # Extract bins
            param_type = "bins"
        else:
            k_bins_values = extract_k_or_bins_values_from_files(estimator_files)  # Extract k
            param_type = "k"

        if not k_bins_values:
            print(f"No {param_type} values found for {mi_estimate}. Exiting.")
            exit()

        # Ask the user to choose the value of k or bins for the plot
        k_bins_choice = get_user_choice(k_bins_values, f"Choose the value of {param_type} to use for the plot (for {mi_estimate}):")

        # Prepare x, y values for the plot
        x_vals, y_vals, y_err, xmin, xmax = [], [], [], [], []

        for param_value in sorted(filtered_files[idx].keys()):
            files_for_param_value = filtered_files[idx][param_value]

            for file in files_for_param_value:
                # List to store valid data
                valid_data = []

                with open(file, 'r') as f:
                    # Skip the header (first row)
                    next(f)

                    # Loop to read each line from the file
                    for line in f:
                        # Strip any leading/trailing whitespace
                        line = line.strip()
                        
                        # Skip empty lines
                        if not line:
                            continue
                        
                        # Convert the line into a float array (split by whitespace or tab)
                        row = np.fromstring(line, sep=' ')

                        if "binning" in mi_estimate.lower():
                            # Ensure the row has at least 7 columns (since you expect 7 values per row)
                            if row.size >= 7:
                                fifth_column_value = row[4]

                                # Check if it is not infinity or NaN
                                if not np.isinf(fifth_column_value) and not np.isnan(fifth_column_value):
                                    valid_data.append(row)  # Add the row to the list if it's valid
                            else:
                                print(f"Skipping line due to insufficient columns: {line}")
                        else:
                            # Ensure the row has at least 4 columns (since you expect 4 values per row)
                            if row.size >= 4:
                                second_column_value = row[1]  # The second column (index 1)

                                # Check if the second column is not infinity or NaN
                                if not np.isinf(second_column_value) and not np.isnan(second_column_value):
                                    valid_data.append(row)  # Add the row to the list if it's valid
                            else:
                                print(f"Skipping line due to insufficient columns: {line}")

                if "binning" in mi_estimate.lower():
                    # Convert the valid data into a numpy array
                    data_cleaned = np.array(valid_data).reshape(-1, 7)

                    # Separate the columns
                    first_column = data_cleaned[:, 0]
                    means = data_cleaned[:, 4]
                    sigmas = data_cleaned[:, 6]
                    xlolims = data_cleaned[:, 1]
                    xuplims = data_cleaned[:, 3]

                    for i in range(len(first_column)):
                        if int(first_column[i]) == k_bins_choice:
                                # Compute the theoretical MI
                                x_vals.append(param_value)  # Parameter value will be on the x-axis
                                theoretical_mi = theoretical_mi_values.get(param_value, 1)
                                y_vals.append(float(means[i]) / theoretical_mi)
                                y_err.append(float(sigmas[i]) / np.abs(theoretical_mi))
                                xmin.append(xlolims)
                                xmax.append(xuplims)
                else:
                    # Convert the valid data into a numpy array
                    data_cleaned = np.array(valid_data).reshape(-1, 4)

                    # Separate the columns
                    first_column = data_cleaned[:, 0]
                    means = data_cleaned[:, 1]
                    sigmas = data_cleaned[:, 3]

                    for i in range(len(first_column)):
                        if int(first_column[i]) == k_bins_choice:
                            # Compute the theoretical MI
                            x_vals.append(param_value)  # Parameter value will be on the x-axis
                            theoretical_mi = theoretical_mi_values.get(param_value, 1)
                            y_vals.append(float(means[i]) / theoretical_mi)
                            y_err.append(float(sigmas[i]) / np.abs(theoretical_mi))

        if "binning" in mi_estimate.lower():
            # Assign color and plot
            plt.errorbar(x_vals, y_vals, yerr=y_err, xlolims=xlolims, xuplims=xuplims, fmt='.', linestyle='--', capsize=1, label=mi_estimate)
        else:
            # Assign color and plot
            plt.errorbar(x_vals, y_vals, yerr=y_err, fmt='.', linestyle='--', capsize=1, label=mi_estimate)

    # Title formatting
    formatted_distribution_name = distribution_name.replace('_', r'\_')
    title = r"$\mathrm{" + formatted_distribution_name + r"}$"
    title += f" (N={N_choice})"
    plt.title(title, fontsize=15)
    plt.xscale('log')
    plt.xlabel(rf"${selected_param}$", fontsize=15)
    plt.ylabel(r"$\mathrm{I}_{\mathrm{est}} / \mathrm{I}_{\mathrm{theoretical}}$", fontsize=15)
    plt.legend(title="MI Estimators", fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()




def main():
    # Load available distributions and estimators
    available_distributions = load_config_distributions()
    mi_estimators = ["mi_1", "mi_sum", "mi_binning"]

    # Choose the figure
    figure_choice = get_user_choice(["4", "7", "8", "9", "20", "21"], "Choose the figure to reproduce:")
    print(f"You have chosen figure {figure_choice}.")

    # Choose the mutual information estimate
    if figure_choice in ["20", "21"]:
        mi_choice = mi_estimators
    else:
        mi_choice = get_user_choice(mi_estimators, "Choose the mutual information estimate to use:")
        print(f"You have chosen estimate {mi_choice}.")

    # Choose the distribution
    distribution_choice = get_user_choice(available_distributions, "Choose the distribution to use for the plot:")
    print(f"You have chosen distribution {distribution_choice}.")

    # Check if log-log transformation is required
    log_transformed = False
    if distribution_choice in ["ordered_wienman_exponential", "gamma_exponential"]:
        log_transformed_choice = get_user_choice(["No", "Yes"], "Do you want to use log-log transformed values?")
        log_transformed = log_transformed_choice == "Yes"

    # Get the current directory
    directory_path = os.path.join(os.getcwd(), "data", "mi_summaries")

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"The folder {directory_path} does not exist.")

    print(f"You have chosen the folder: {directory_path}")

    # Search for matching files
    if figure_choice in ["20","21"]:
        matching_files = []
        for mi in mi_choice:
            matching_files.extend(find_matching_files(directory_path, distribution_choice, mi, log_transformed))
    else:
        matching_files = find_matching_files(directory_path, distribution_choice, mi_choice, log_transformed)

    if matching_files:
        print("Files found:")
        for file in matching_files:
            print(file)
    else:
        print("No matching files found in the selected folder.")


    # If figure 4 is chosen
    if figure_choice == "4":
        process_figure_4(matching_files, distribution_choice, mi_choice, log_transformed)

    # If figure 7 is chosen
    if figure_choice == "7":
        process_figure_7_9(matching_files, distribution_choice, mi_choice, figure_choice, log_transformed)

    # If figure 8 is chosen
    if figure_choice == "8":
        process_figure_8(matching_files, distribution_choice, mi_choice, log_transformed)

    # If figure 9 is chosen
    if figure_choice == "9":
        process_figure_7_9(matching_files, distribution_choice, mi_choice, figure_choice, log_transformed)

    # If figure 20 is chosen
    if figure_choice == "20":
        process_figure_20(matching_files, distribution_choice, mi_choice, log_transformed)

    # If figure 21 is chosen
    if figure_choice == "21":
        process_figure_21(matching_files, distribution_choice, mi_choice, log_transformed)


if __name__ == "__main__":
    main()

    while True:
        ask_for_another_plot = get_user_choice(["Yes", "No"], "Do you want to create another plot?") == "Yes"

        if not ask_for_another_plot:
            print("Exiting the plot creation process.")
            break

        main()

