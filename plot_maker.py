import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import scipy.interpolate as spi
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
    The y-axis represents the estimated mutual information.
    """
    plt.figure(figsize=(8, 6))

    # Get all available N values to assign colors
    N_values = sorted(set(int(re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_({mi_estimate}).txt", 
                                        os.path.basename(f)).group(2)) for f in files if re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_({mi_estimate}).txt", 
                                        os.path.basename(f))))
    if log_transformed:
        N_values = sorted(set(int(re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_log_transformed_({mi_estimate}).txt", 
                                            os.path.basename(f)).group(2)) for f in files if re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_log_transformed_({mi_estimate}).txt", 
                                            os.path.basename(f))))

    # Create a color map for each unique N value
    color_map = {N: plt.cm.Set1(i / (len(N_values)-1)) for i, N in enumerate(N_values)}

    # List to collect labels for the legend
    legend_labels = []

    for file in sorted(files):
        match = re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_({mi_estimate}).txt", os.path.basename(file))
        if log_transformed:
            match = re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_log_transformed_({mi_estimate}).txt", os.path.basename(file))
        if match:
            N = int(match.group(2))
            color = color_map[N]  # Get the corresponding color for this N

            # Load data
            first_column, means, sigmas = np.loadtxt(file, skiprows=1, unpack=True)

            # If mi_estimate is "mi_binning", the first value represents bins_number
            # Otherwise, it represents k_vals
            x_vals = first_column / N  # Normalize by N

            # Sort to ensure the points are connected
            sorted_indices = np.argsort(x_vals)
            x_vals_sorted = x_vals[sorted_indices]
            means_sorted = means[sorted_indices]
            sigmas_sorted = sigmas[sorted_indices]

            # Plot a line between the points with the assigned color
            plt.plot(x_vals_sorted, means_sorted, linestyle='--', color=color)

            # Points with error bars of the same color
            plt.errorbar(x_vals_sorted, means_sorted, yerr=sigmas_sorted, fmt='.', color=color, capsize=1)

            # Add legend based on the N value
            legend_labels.append(f'N={N}')

    # Add the theoretical mutual information
    plt.axhline(y=theoretical_mi, color='r', linestyle='-', linewidth=1, label=f'I theoretical = {theoretical_mi:.4f}')

    # Customize the plot
    plt.xlabel(r"num $_{\mathrm{bins}}$ /N" if mi_estimate == "mi_binning" else "k/N", fontsize=11)

    # Map to transform mi_estimate to the subscript format
    subscript_map = {
        "mi_1": "1",
        "mi_sum": "sum",
        "mi_binning": "binning"
    }

    # Get the correct subscript
    subscript = subscript_map.get(mi_estimate, "")

    # Set the label with the subscript not in italics
    plt.ylabel(r"I$_{\mathrm{" + subscript + r"}}$", fontsize=11)

    # Sort the labels by N in ascending order
    sorted_labels = sorted(legend_labels, key=lambda label: int(label.split('=')[1]))

    plt.legend(title="Sample Size (N)", fontsize=11, loc='best', labels=sorted_labels)
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

    # Make the plot
    plot_figure_4(filtered_files, distribution_name, mi_estimate, theoretical_mi, log_transformed)



def process_figure_7_9(files, distribution_name, mi_estimate, figure, log_transformed):
    """
    Manages the parameter selection and the k or number of bins for Figures 7 and 9, filters the corresponding files, and plots.
    """
    # Extract the available k values from the files or bins if mi_estimate is "mi_binning"
    if mi_estimate == "mi_binning":
        bins_values = extract_k_or_bins_values_from_files(files)  # Function to extract bins from files
        if not bins_values:
            print("No bin values found in the files. Exiting.")
            exit()
        # User chooses the number of bins
        bins_choice = get_user_choice(bins_values, "Choose the number of bins to use for the plot:")
        print(f"You chose the number of bins: {bins_choice}")
        k_choice = None  # In this case, k is not used
    else:
        # Extract the available k values from the files
        k_values = extract_k_or_bins_values_from_files(files)
        if not k_values:
            print("No k values found in the files. Exiting.")
            exit()
        # User chooses the value of k
        k_choice = get_user_choice(k_values, "Choose the value of k to use for the plot:")
        print(f"You chose the value of k: {k_choice}")
        bins_choice = None  # In this case, bins are not used

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

    # Create a soft color map for each parameter value
    color_map = plt.cm.get_cmap('Set1', len(grouped_files))  # Use a predefined colormap (tab10)

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
                with open(file, 'r') as f:
                    next(f)  # Skip the header
                    for line in f:
                        columns = line.strip().split()
                        if columns and ((k_choice and int(columns[0]) == k_choice) or (bins_choice and int(columns[0]) == bins_choice)):
                            match = re.search(r"size_(\d+)", file)
                            if match:
                                N = int(match.group(1))
                                # If theoretical mutual information is valid, add points
                                x_vals.append((k_choice or bins_choice) / N)
                                y_vals.append(float(columns[1]) / theoretical_mi)
                                y_errs.append(float(columns[2]) / theoretical_mi)
                            else:
                                print(f"Warning: unable to extract N from file name {file}")
        elif figure == "9":
            for file in file_list:
                with open(file, 'r') as f:
                    next(f)  # Skip the header
                    for line in f:
                        columns = line.strip().split()
                        if columns and ((k_choice and int(columns[0]) == k_choice) or (bins_choice and int(columns[0]) == bins_choice)):
                            match = re.search(r"size_(\d+)", file)
                            if match:
                                N = int(match.group(1))
                                # If theoretical mutual information is valid, add points
                                x_vals.append((k_choice or bins_choice) / N)
                                # Subtract theoretical mutual information from mean (y_vals)
                                y_vals.append(float(columns[1]) - theoretical_mi)
                                # Keep error the same (y_errs)
                                y_errs.append(float(columns[2]))
                            else:
                                print(f"Warning: unable to extract N from file name {file}")

        # Sort the data by x
        sorted_indices = np.argsort(x_vals)
        x_vals = np.array(x_vals)[sorted_indices]
        y_vals = np.array(y_vals)[sorted_indices]
        y_errs = np.array(y_errs)[sorted_indices]

        # Assign a color for this parameter value
        color = color_map(idx)

        # Connect the points with a line using the color
        plt.plot(x_vals, y_vals, linestyle='--', label=f"{selected_param}={param_value}", color=color)

        # Plot with error bars using the same color
        plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='.', color=color, capsize=1)

    plt.xlabel(f"{bins_choice}/N" if mi_estimate == "mi_binning" else f"{k_choice}/N", fontsize=11)

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
        plt.ylabel(r"$\mathrm{I}_{\mathrm{" + subscript + r"}} / \mathrm{I}_{\mathrm{exact}}$", fontsize=11)
    else:
        if figure == "9":
            plt.ylabel(r"$\mathrm{I}_{\mathrm{" + subscript + r"}} - \mathrm{I}_{\mathrm{exact}}$", fontsize=11)

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

            # Load the data
            x_vals, means, sigmas = np.loadtxt(file, skiprows=1, unpack=True)

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
    color_map = plt.cm.get_cmap('Set1', len(mi_estimators))  # Assign different colors to estimators

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

        x_vals, y_vals, y_errs = [], [], []
        
        # Process files and extract data
        for file in filtered_files[idx]:
            with open(file, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    columns = line.strip().split()
                    if columns and int(columns[0]) == k_bins_choice:
                        match = re.search(r"size_(\d+)", file)
                        if match:
                            N = int(match.group(1))
                            x_vals.append(N)
                            y_vals.append(float(columns[1]) / theoretical_mi)
                            y_errs.append(float(columns[2]) / theoretical_mi)
                        else:
                            print(f"Warning: unable to extract N from file name {file}")
        
        # Sort data by x (N)
        sorted_indices = np.argsort(x_vals)
        x_vals = np.array(x_vals)[sorted_indices]
        y_vals = np.array(y_vals)[sorted_indices]
        y_errs = np.array(y_errs)[sorted_indices]
        
        # Assign color and plot
        color = color_map(idx)
        plt.plot(x_vals, y_vals, linestyle='--', label=mi_estimate, color=color)
        plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='.', color=color, capsize=1)

    plt.xlabel(r"N", fontsize=11)
    plt.ylabel(r"$\mathrm{I}_{\mathrm{est}} / \mathrm{I}_{\mathrm{exact}}$", fontsize=11)
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


if __name__ == "__main__":
    main()

