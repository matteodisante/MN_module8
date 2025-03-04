import os
import re
import numpy as np
import matplotlib.pyplot as plt
import inspect

from utils.math_utils import circular_mi_theoretical, gamma_exponential_mi_theoretical, \
    correlated_gaussian_rv_mi_theoretical, independent_exponential_rv_mi_theoretical, \
        independent_gaussian_rv_mi_theoretical, independent_uniform_rv_mi_theoretical, \
            ordered_wienman_exponential_mi_theoretical

from utils.plot_utils import load_config_distributions, get_user_choices, \
    find_matching_files, extract_parameters_from_paths, filter_files_by_parameters, \
        process_data_structure, read_and_clean_data, \
            format_plot, extract_k_or_bins_values_from_files



def process_figure_7_9(files, distribution_name, mi_estimate, figure, log_transformed):

    # Determine whether to ask for k or bins
    param_type = "bins" if "binning" in mi_estimate.lower() else "k"
    k_bins_values = extract_k_or_bins_values_from_files(files)
    
    if not k_bins_values:
        print(f"No {param_type} values found for {mi_estimate}. Exiting.")
        exit()

    # The user chooses the value of k or bins based on the MI method
    k_bins_choice = get_user_choices(k_bins_values, f"Choose the value of {param_type} to use for the plot (for {mi_estimate}):", multiple=False)

    # Extract parameters from the files
    all_parameters, extracted_params_per_file = extract_parameters_from_paths(files, distribution_name, mi_estimate, log_transformed)

    selected_param = get_user_choices(list(all_parameters.keys()), "Choose the parameter to analyze:", multiple=False)

    # Group files by the selected parameter's value
    grouped_files = {}
    for param_value in all_parameters[selected_param]:
        selected_files = filter_files_by_parameters(extracted_params_per_file, {selected_param: param_value})
        if selected_files:
            grouped_files[param_value] = selected_files

    plt.figure(figsize=(10, 5))

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
            if "independent" in distribution_name:
                figure="9"
            elif "independent" not in distribution_name and theoretical_mi == 0:
                print(f"Warning: theoretical mutual information is zero for {selected_param}={param_value}. Point excluded from the legend.")
                continue  # Skip the rest of the loop for this parameter value

        if theoretical_mi is None:
            print(f"Warning: theoretical mutual information is None for {selected_param}={param_value}. Point excluded from the legend.")
            continue  # Skip the rest of the loop for this parameter value

        for file in file_list:
            valid_data = []
            valid_data = read_and_clean_data(file, mi_estimate)

            first_column, _, _, _, means, sigmas = process_data_structure(valid_data, mi_estimate)

            for i in range(len(first_column)):
                if int(first_column[i]) == k_bins_choice:
                    match = re.search(r"size_(\d+)", file)
                    if match:
                        N = int(match.group(1))
                        # If theoretical mutual information is valid, add points
                        x_vals.append(1/N)
                        if figure=="7":
                            y_vals.append(float(means[i]) / theoretical_mi)
                            y_errs.append(float(sigmas[i]) / np.abs(theoretical_mi))
                        elif figure=="9":
                            y_vals.append(float(means[i]) - theoretical_mi)
                            y_errs.append(float(sigmas[i]))
                    else:
                        print(f"Warning: unable to extract N from file name {file}")

        # Sort the data by x_vals
        sorted_indices = np.argsort(x_vals)
        x_vals, y_vals, y_errs = np.array(x_vals)[sorted_indices], np.array(y_vals)[sorted_indices], np.array(y_errs)[sorted_indices]

        # Mapping for parameter names in legend and get the correct subscript
        param_name_mapping = {
        "corr": "correlation", "theta": r"$\theta$", "a": "a", "b": "b", "c": "c",
        "lambda": "lambda", "mu": "mu", "sigma": "sigma", "low": "low", "high": "high"
        }.get(selected_param, "")

        label = None
        if distribution_name != "circular" and "independent" not in distribution_name:
           label = f"{param_name_mapping}: {param_value}"
        plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='.', linestyle='--', capsize=1, markersize=3, linewidth=0.93, alpha = 0.7, label=label)

    format_plot(mi_estimate=mi_estimate, figure_choice=figure, distribution_name=distribution_name, 
                param_name_mapping=param_name_mapping)



def process_figure_8(files, distribution_name, mi_estimate, log_transformed):

    # Determine if the estimator uses k or bins
    k_or_bins_name = "bins" if mi_estimate == "mi_binning" else "k"
    k_or_bins_values = extract_k_or_bins_values_from_files(files)  # Extract available values
    
    if not k_or_bins_values:
        print(f"No {k_or_bins_name} values found in the files. Exiting.")
        exit()
    
    # User chooses the value of k or bins
    k_or_bins_choice = get_user_choices(k_or_bins_values, f"Choose the value of {k_or_bins_name} to use for the plot:", multiple=False)
    print(f"You chose {k_or_bins_name}: {k_or_bins_choice}")

    # Obtain the parameter-value dictionary and its association with the paths
    all_parameters, extracted_params_per_file = extract_parameters_from_paths(files, distribution_name, mi_estimate, log_transformed)
    
    # Define the selected parameters dictionary and ask the user to select the values for the plot
    selected_params = {}

    for param_name, param_values in all_parameters.items():
        chosen_value = get_user_choices(list(param_values), f"Choose the value for the parameter {param_name}:", multiple=False)
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
            valid_data = read_and_clean_data(file, mi_estimate)
            x_vals, _, _, _, _, sigmas = process_data_structure(valid_data, mi_estimate)

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
    plt.figure(figsize=(10, 5))
    plt.errorbar(N_values_sorted, sigma_values_sorted, fmt='.', linestyle='--', markersize=3, linewidth=0.93, alpha = 0.7, color='b')
    format_plot(mi_estimate=mi_estimate, figure_choice="8", distribution_name=distribution_name, selected_params=selected_params)




def main():
    # Load available distributions and estimators
    available_distributions = load_config_distributions()
    mi_estimators = ["mi_1", "mi_sum", "mi_binning"]

    # Choose the figure
    figure_choice = get_user_choices(["7", "8", "9"], "Choose the figure to reproduce:", multiple=False)
    print(f"You have chosen figure {figure_choice}.")

    # Choose the mutual information estimate
    mi_choice = get_user_choices(mi_estimators, "Choose the mutual information estimate to use:", multiple=False)
    print(f"You have chosen estimate {mi_choice}.")

    # Choose the distribution
    distribution_choice = get_user_choices(available_distributions, "Choose the distribution to use for the plot:", multiple=False)
    print(f"You have chosen distribution {distribution_choice}.")

    # Check if log-log transformation is required (only for specific distributions)
    log_transformed = False
    if distribution_choice in ["ordered_wienman_exponential", "gamma_exponential"]:
        log_transformed_choice = get_user_choices(["No", "Yes"], "Do you want to use log-log transformed values?", multiple=False)
        if log_transformed_choice=="Yes":
            log_transformed=True

    # Get the current directory
    directory_path = os.path.join(os.getcwd(), "data", "mi_summaries")

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"The folder {directory_path} does not exist.")

    print(f"You have chosen the folder: {directory_path}")

    # Search for matching files
    matching_files = []

    matching_files = find_matching_files(directory_path, distribution_choice, mi_choice, log_transformed)

    if matching_files:
        print("Files found:")
        for file in matching_files:
            print(file)
    else:
        print("No matching files found in the selected folder.")


    # If figure 7 or 9 is chosen
    if figure_choice in ["7", "9"]:
        process_figure_7_9(matching_files, distribution_choice, mi_choice, figure_choice, log_transformed)

    # If figure 8 is chosen
    if figure_choice == "8":
        process_figure_8(matching_files, distribution_choice, mi_choice, log_transformed)

if __name__ == "__main__":
    main()

    while True:
        ask_for_another_plot = get_user_choices(["Yes", "No"], "Do you want to create another plot?", multiple=False) == "Yes"

        if not ask_for_another_plot:
            print("Exiting the plot creation process.")
            break

        main()

