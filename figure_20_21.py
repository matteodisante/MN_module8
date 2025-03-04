import os
import re
import numpy as np
import matplotlib.pyplot as plt
import inspect

from utils.plot_utils import load_config_distributions, get_user_choices, \
    find_matching_files, extract_parameters_from_paths, filter_files_by_parameters, extract_k_or_bins_values_from_files, \
        process_theoretical_mi, process_data_structure, read_and_clean_data, format_plot, extract_N_values



def process_figure_20(files, distribution_name, mi_estimators, log_transformed, is_independent):

    # Gets the parameter-value dictionary and the one for associating paths
    all_parameters = []
    extracted_params_per_file = []
    for mi in mi_estimators:
        params, extracted_params = extract_parameters_from_paths(files, distribution_name, mi, log_transformed)
        all_parameters.append(params)
        extracted_params_per_file.append(extracted_params)

    selected_params = {}

    for param_name, param_values in all_parameters[0].items():
        chosen_value = get_user_choices(list(param_values), f"Choose the value for the parameter {param_name}:", multiple=False)
        selected_params[param_name] = chosen_value

    filtered_files = []
    # Filters the files with the selected parameter values from the user
    for i, mi in enumerate(mi_estimators):
        filtered = filter_files_by_parameters(extracted_params_per_file[i], selected_params)
        filtered_files.append(filtered)

    # Theoretical MI
    theoretical_mi = process_theoretical_mi(distribution_name, selected_params)

    if theoretical_mi is None:
        print(f"Warning: theoretical mutual information is None for selected parameter values.")

    plt.figure(figsize=(10, 5))

    for idx, mi_estimate in enumerate(mi_estimators):
        # Determine whether to ask for k or bins
        param_type = "bins" if "binning" in mi_estimate.lower() else "k"
        k_bins_values = extract_k_or_bins_values_from_files(filtered_files[idx])  

        if not k_bins_values:
            print(f"No {param_type} values found for {mi_estimate}. Exiting.")
            exit()

        k_bins_choice = get_user_choices(k_bins_values, f"Choose the value of {param_type} to use for the plot (for {mi_estimate}):", multiple=False)
        x_vals, y_vals, y_errs = [], [], []
        
        # Process files and extract data
        for file in filtered_files[idx]:
            # List to store valid data
            valid_data = []
            valid_data = read_and_clean_data(file, mi_estimate)

            first_column, _, _, _, means, sigmas = process_data_structure(valid_data, mi_estimate)

            for i in range(len(first_column)):
                if int(first_column[i]) == k_bins_choice:
                    match = re.search(r"size_(\d+)", file)
                    if match:
                        N = int(match.group(1))
                        x_vals.append(1/N)
                        if is_independent:
                            y_vals.append(float(means[i]))
                            y_errs.append(float(sigmas[i]))
                        else:
                            y_vals.append(float(means[i]) / theoretical_mi)
                            y_errs.append(float(sigmas[i]) / np.abs(theoretical_mi))

        # Sort data by N and plot 
        sorted_indices = np.argsort(x_vals)
        x_vals, y_vals, y_errs = np.array(x_vals)[sorted_indices], np.array(y_vals)[sorted_indices], np.array(y_errs)[sorted_indices]

        name = next((n for n in ["1", "sum", "binning"] if n in mi_estimate.lower()), "")
        plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='.', linestyle='--', capsize=1, markersize=3, linewidth=0.93, alpha=0.7, label=r"$\mathrm{I}_{\mathrm{" + name + r"}}$")

    format_plot(figure_choice="20", is_independent=is_independent, distribution_name=distribution_name, selected_params=selected_params)




def process_figure_21(files, distribution_name, mi_estimators, log_transformed, is_independent):

    # Gets the parameter-value dictionary and the one for associating paths
    all_parameters = []
    extracted_params_per_file = []
    for mi in mi_estimators:
        params, extracted_params = extract_parameters_from_paths(files, distribution_name, mi, log_transformed)
        all_parameters.append(params)
        extracted_params_per_file.append(extracted_params)

    # Choose the parameter to analyze
    selected_param = get_user_choices(list(all_parameters[0].keys()), "Choose the parameter to analyze:", multiple=False)

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

    # Get all possible values of N from the file names and ask the user to choose a value for N
    available_sizes = extract_N_values(files)
    N_choice = get_user_choices(available_sizes, "Choose a value for N:", multiple=False)
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
    plt.figure(figsize=(10, 5))

    # Select any value from the set of parameters for the given distribution
    param_value_choice = next(iter(all_parameters[0][selected_param]))

    # Iterate over each MI estimator
    for idx, mi_estimate in enumerate(mi_estimators):
        # Get the files corresponding to the current estimator and the selected parameter value
        estimator_files = filtered_files[idx].get(param_value_choice, [])

        # Determine whether to ask for k or bins
        param_type = "bins" if "binning" in mi_estimate.lower() else "k"
        k_bins_values = extract_k_or_bins_values_from_files(estimator_files)  

        if not k_bins_values:
            print(f"No {param_type} values found for {mi_estimate}. Exiting.")
            exit()

        k_bins_choice = get_user_choices(k_bins_values, f"Choose the value of {param_type} to use for the plot (for {mi_estimate}):", multiple=False)

        # Prepare x, y values for the plot
        x_vals, y_vals, y_err = [], [], []

        for param_value in sorted(filtered_files[idx].keys()):
            files_for_param_value = filtered_files[idx][param_value]

            for file in files_for_param_value:
                # List to store valid data
                valid_data = []
                valid_data = read_and_clean_data(file, mi_estimate)

                first_column, _, _, _, means, sigmas = process_data_structure(valid_data, mi_estimate)

                for i in range(len(first_column)):
                    if int(first_column[i]) == k_bins_choice:
                            # Compute the theoretical MI
                            x_vals.append(float(param_value))  # Parameter value will be on the x-axis
                            theoretical_mi = theoretical_mi_values.get(param_value, 1)
                            if is_independent:
                                y_vals.append(float(means[i]))
                                y_err.append(float(sigmas[i]))
                            else:
                                y_vals.append(float(means[i]) / theoretical_mi)
                                y_err.append(float(sigmas[i]) / np.abs(theoretical_mi))

        # Sort data by params and plot
        sorted_indices = np.argsort(x_vals)
        x_vals, y_vals, y_err = np.array(x_vals)[sorted_indices], np.array(y_vals)[sorted_indices], np.array(y_err)[sorted_indices]

        name = next((n for n in ["1", "sum", "binning"] if n in mi_estimate.lower()), "")
        plt.errorbar(x_vals, y_vals, yerr=y_err, fmt='.', linestyle='--', capsize=1, markersize=3, linewidth=0.93, alpha=0.7, label=r"$\mathrm{I}_{\mathrm{" + name + r"}}$")

    # Mapping for parameter names in legend and get the correct subscript
    param_name_mapping = {
    "corr": "correlation", "theta": r"$\theta$", "a": "a", "b": "b", "c": "c",
    "lambda": "lambda", "mu": "mu", "sigma": "sigma", "low": "low", "high": "high"
    }.get(selected_param, "")

    format_plot(figure_choice="21", is_independent=is_independent, N_value=N_choice, distribution_name=distribution_name, param_name_mapping=param_name_mapping)




def main():
    # Load available distributions and estimators
    available_distributions = load_config_distributions()
    mi_estimators = ["mi_1", "mi_sum", "mi_binning"]

    # Choose the figure
    figure_choice = get_user_choices(["20", "21"], "Choose the figure to reproduce:", multiple=False)
    print(f"You have chosen figure {figure_choice}.")

    # Choose the distribution
    distribution_choice = get_user_choices(available_distributions, "Choose the distribution to use for the plot:", multiple=False)
    print(f"You have chosen distribution {distribution_choice}.")

    is_independent=False
    if "independent" in distribution_choice:
        is_independent=True

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

    for mi in mi_estimators:
            matching_files.extend(find_matching_files(directory_path, distribution_choice, mi, log_transformed))

    if matching_files:
        print("Files found:")
        for file in matching_files:
            print(file)
    else:
        print("No matching files found in the selected folder.")

    # If figure 20 is chosen
    if figure_choice == "20":
        process_figure_20(matching_files, distribution_choice, mi_estimators, log_transformed, is_independent)

    # If figure 21 is chosen
    if figure_choice == "21":
        process_figure_21(matching_files, distribution_choice, mi_estimators, log_transformed, is_independent)


if __name__ == "__main__":
    main()
    while True:

        ask_for_another_plot = get_user_choices(["Yes", "No"], "Do you want to create another plot?", multiple=False) == "Yes"
        
        if not ask_for_another_plot:
            print("Exiting the plot creation process.")
            break

        main()
