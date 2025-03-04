import os
import re
import numpy as np
import matplotlib.pyplot as plt

from utils.plot_utils import load_config_distributions, get_user_choices, \
    find_matching_files, extract_parameters_from_paths, filter_files_by_parameters, \
        process_theoretical_mi, select_and_filter_files, process_data_structure, read_and_clean_data, format_plot, extract_N_values, extract_file_info



def plot_figure_4(files, distribution_name, mi_estimate, theoretical_mi, log_transformed, selected_params):

    plt.figure(figsize=(10, 5))

    # Dictionary to store the data for each N
    data_dict = {}

    # List to collect the legend labels
    legend_labels = []

    for file in files:
        _, N = extract_file_info(file, distribution_name, mi_estimate, log_transformed)
        if N is not None:

            valid_data = read_and_clean_data(file, mi_estimate)
            first_column, x_values, xlolims, xuplims, means, sigmas = process_data_structure(valid_data, mi_estimate)

            if "binning" in mi_estimate.lower():
                # Store the data in the dictionary, using N as the key
                data_dict[N] = (x_values, xlolims, xuplims, means, sigmas)
            else: 
                data_dict[N] = (first_column, means, sigmas)

    # Sort the dictionary by N in ascending order
    sorted_N_values = sorted(data_dict.keys())

    # Plot the data for each sorted N value
    for N in sorted_N_values:
        if "binning" in mi_estimate.lower():
            x_values, xlolims, xuplims, means, sigmas = data_dict[N]
        else:
            first_column, means, sigmas = data_dict[N]

        if "binning" in mi_estimate.lower():
            x_err_left = x_values - xlolims
            x_err_right = xuplims - x_values
            x_errors = [x_err_left/N, x_err_right/N]
            plt.errorbar(x_values/N, means, yerr=sigmas, xerr=x_errors, linestyle='--', fmt='.', capsize=1, markersize=3, linewidth=0.93, alpha=0.7, label=f'N={N}') 
        else:
            plt.errorbar(first_column/N, means, yerr=sigmas, linestyle='--', fmt='.', capsize=1, markersize=3, linewidth=0.93, alpha=0.7, label=f'N={N}')
        legend_labels.append(f'N={N}')

    format_plot(mi_estimate=mi_estimate, figure_choice="4" ,distribution_name=distribution_name, selected_params=selected_params, theoretical_mi=theoretical_mi)



def process_figure_4(files, distribution_name, mi_estimate, log_transformed):

    # Gets the parameter-value dictionary and the one for associating paths, filters the files with the selected parameter values from the user
    filtered_files, selected_params = select_and_filter_files(files, distribution_name, mi_estimate, log_transformed)
    if not filtered_files:
        print("No files available after filtering.")
        return

    # Calculate the theoretical mutual information
    theoretical_mi = process_theoretical_mi(distribution_name, selected_params)

    # Ask the user whether they want to choose one or more N, or plot all available N values
    N_values = extract_N_values(filtered_files)

    print(f"Available N values: {N_values}")
    choice = input("Do you want to (1) Choose one or more values of N or (2) Plot all available N values? (Enter 1 or 2): ").strip()

    if choice == '1':
            # User selects one or more N values
            chosen_Ns = get_user_choices(list(N_values), f"Choose one or more values for N from the available options (comma separated):", multiple=True)

            final_filtered_files = [file for file in filtered_files if any(f"size_{chosen_N}_" in os.path.basename(file) for chosen_N in chosen_Ns)]
            print(f"Selected files for N={chosen_Ns}:")
            for file in final_filtered_files:
                print(file)

    elif choice == '2':
        # Plot all N values (no filtering by N)
        final_filtered_files = filtered_files
        print("Plotting all available N values...")

    else:
        print("Invalid choice. Please enter 1 or 2.")
        return

    # Make the plot
    plot_figure_4(final_filtered_files, distribution_name, mi_estimate, theoretical_mi, log_transformed, selected_params)



def plot_figure4b(distribution_choices, distribution_data, mi_estimate, N_value, param_names_per_distribution, is_independent):

    plt.figure(figsize=(10, 5))

    legend_labels = []  # Collect legend labels

    # Loop through all the distributions
    for distribution_name in distribution_choices:

        if distribution_name not in distribution_data:
            print(f"No data available for {distribution_name}. Skipping this distribution.")
            continue

        param_name = param_names_per_distribution.get(distribution_name, "Unknown")  # Prendi il parametro corretto

        # Iterate through the parameter values for the distribution (sorted order)
        for param_value in sorted(distribution_data[distribution_name].keys(), key=lambda x: float(x)):
            param_info = distribution_data[distribution_name][param_value]

            # Extract the files and theoretical MI for this parameter value
            filtered_files_for_param = param_info['files']
            theoretical_mi = param_info['theoretical_mi']

            if not filtered_files_for_param:
                print(f"No files found for {distribution_name} with parameter value {param_value}. Skipping.")
                continue  # Skip if no files exist for this parameter value

            # Collect data from the files
            valid_data = []
            for file in filtered_files_for_param:
                # Match the file based on its naming convention
                match = re.search(rf"summary_{distribution_name}_(.*?)_size_{N_value}_(.*?).txt", os.path.basename(file))
                if match:
                    valid_data = read_and_clean_data(file, mi_estimate)

            first_column, x_values, xlolims, xuplims, means, sigmas = process_data_structure(valid_data, mi_estimate)

            # Normalize the mean values by the theoretical MI (if available)
            if theoretical_mi is not None:
                cleaned_distribution_name = re.sub(r'^correlated_', '', re.sub(r'_rv$', '', distribution_name)).replace('_', ' ')

                # Mapping for parameter names in legend and get the correct subscript
                param_name_mapping = {
                "corr": "correlation", "theta": r"$\theta$", "a": "a", "b": "b", "c": "c",
                "lambda": "lambda", "mu": "mu", "sigma": "sigma", "low": "low", "high": "high"
                }.get(param_name, "")

                if is_independent or ("circular" in distribution_name):
                    label = f"{cleaned_distribution_name}"
                else:
                    label = f"{cleaned_distribution_name}, {param_name_mapping}: {param_value}"
                legend_labels.append(label)

                if "binning" in mi_estimate.lower():
                    x_err_left = x_values - xlolims
                    x_err_right = xuplims - x_values
                    x_errors = [x_err_left, x_err_right]
                    if is_independent:
                        plt.errorbar(x_values, means, yerr=sigmas, xerr=x_errors, linestyle='--', fmt='.', capsize=1, markersize=3, linewidth=0.93, alpha=0.5, label=label)
                    else:
                        plt.errorbar(x_values, means/theoretical_mi, yerr=sigmas/np.abs(theoretical_mi), xerr=x_errors, linestyle='--', fmt='.', capsize=1, markersize=3, linewidth=0.93, alpha=0.5, label=label)
                else:
                    if is_independent:
                        plt.errorbar(first_column, means, yerr=sigmas, linestyle='--', fmt='.', capsize=1, markersize=3, linewidth=0.93, alpha=0.5, label=label)
                    else:
                        plt.errorbar(first_column, means/theoretical_mi, yerr=sigmas/np.abs(theoretical_mi), linestyle='--', fmt='.', capsize=1, markersize=3, linewidth=0.93, alpha=0.5, label=label)
            else:
                print(f"Skipping plot for {distribution_name} with parameter value {param_value} due to missing theoretical MI.")

    format_plot(mi_estimate=mi_estimate, figure_choice="4b",is_independent=is_independent, N_value=N_value, distribution_choices=distribution_choices)



def process_figure_4b(matching_files, distribution_choices, mi_choice, is_independent):

    # Ask the user to choose a value of N
    print("\nAvailable N values from files:")
    N_values = extract_N_values(matching_files)
    N_choice = get_user_choices(list(N_values), "Choose an N value to plot against:", multiple=False)

    # Filter the files for the chosen value of N
    filtered_files = []

    for file in matching_files:
        if f"size_{N_choice}_" in os.path.basename(file):
            filtered_files.append(file)

    # Create a dictionary to organize files based on distribution
    files_by_distribution = {dist: [] for dist in distribution_choices}

    for file in filtered_files:
        for distribution_name in distribution_choices:
            if distribution_name in os.path.basename(file):
                files_by_distribution[distribution_name].append(file)
                break  # If the file matches the distribution, there is no need to check other distributions

    # Create a structure to store distribution files, chosen parameters, and theoretical MI for those parameters
    distribution_data = {}
    param_names_per_distribution = {}  # New dictionary to store the names of selected parameters for plotting

    for distribution_name in distribution_choices:
        print(f"\nFor the distribution {distribution_name}, you need to choose a parameter to plot against.")

        filtered_files_for_dist = files_by_distribution[distribution_name]
        if not filtered_files_for_dist:
            print(f"No files found for the distribution {distribution_name}. Skipping this distribution.")
            continue

        # Determine if log_transformed is used for the chosen distribution
        log_transformed = False
        for file in filtered_files_for_dist:
            if "log_transformed" in os.path.basename(file):
                log_transformed = True
                break  # If any file is log_transformed, stop checking

        # Extract parameters from file names
        all_parameters, extracted_params_per_file = extract_parameters_from_paths(
            filtered_files_for_dist, distribution_name, mi_choice, log_transformed
        )

        selected_params = {}

        # Display available parameters and extract the parameter for plotting, while fixing values for the others
        param_name = get_user_choices(list(all_parameters.keys()), f"Choose the parameter to plot for {distribution_name}:", multiple=False)

        param_names_per_distribution[distribution_name] = param_name  # Save the name for plotting

        # For each parameter different from param_name, choose a fixed value
        for param, param_values in all_parameters.items():
            if param != param_name:
                chosen_value = get_user_choices(list(param_values), f"Choose the value for the parameter {param}:", multiple=False)
                selected_params[param] = chosen_value
            else:
                continue

        # Extract possible values for the selected parameter and ask whether to plot all or only specific values
        possible_values = all_parameters[param_name]

        choice = get_user_choices(["All", "Select specific values"], "Do you want to plot for (1) All values or (2) Select specific values?", multiple=False)

        if choice == "All":
            # If "All" is chosen, take all available values
            selected_values = possible_values
        else:
            # If "Select specific values" is chosen, ask which values to select
            selected_values = get_user_choices(list(possible_values), f"Choose the values for {param_name}: ", multiple=True)

        # Save data related to this distribution and parameter
        param_data = {}
        for param_value in selected_values:
            selected_params[param_name] = param_value
            filtered_files_for_param_value = filter_files_by_parameters(extracted_params_per_file, selected_params)

            # Compute the theoretical mutual information value for the chosen parameter
            theoretical_mi = process_theoretical_mi(distribution_name, selected_params)

            # Save files and theoretical MI
            param_data[param_value] = {
                'files': filtered_files_for_param_value,
                'theoretical_mi': theoretical_mi
            }

        # Add distribution data to the global structure
        distribution_data[distribution_name] = param_data

    # Step 7: Pass the data to the plotting function
    plot_figure4b(distribution_choices, distribution_data, mi_choice, N_choice, param_names_per_distribution, is_independent)



def main():
    # Load available distributions and estimators
    available_distributions = load_config_distributions()
    mi_estimators = ["mi_1", "mi_sum", "mi_binning"]

    # Choose the figure
    figure_choice = get_user_choices(["4", "4b"], "Choose the figure to reproduce:", multiple=False)
    print(f"You have chosen figure {figure_choice}.")

    # Choose the mutual information estimate
    mi_choice = get_user_choices(mi_estimators, "Choose the mutual information estimate to use:", multiple=False)
    print(f"You have chosen estimate {mi_choice}.")

    if figure_choice == "4b":
        # Ask user to choose between independent and non-independent distributions
        group_choice = get_user_choices(["Independent", "Non-Independent"], "Choose distribution group:", multiple=False)
        if group_choice == "Independent":
            is_independent = True
            filtered_distributions = [d for d in available_distributions if "independent" in d]
        else:
            is_independent = False
            filtered_distributions = [d for d in available_distributions if "independent" not in d]
        
        # Let user choose distributions within the selected group
        distribution_choices = get_user_choices(filtered_distributions, "Choose one or more distributions:", multiple=True)
        print(f"You have chosen distributions: {', '.join(distribution_choices)}")
    else:
        distribution_choice = get_user_choices(available_distributions, "Choose the distribution to use for the plot:", multiple=False)
        print(f"You have chosen distribution {distribution_choice}.")
    
    # Get the current directory
    directory_path = os.path.join(os.getcwd(), "data", "mi_summaries")

    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"The folder {directory_path} does not exist.")
    
    print(f"You have chosen the folder: {directory_path}")

    # Search for matching files
    matching_files = []
    if figure_choice == "4b":
        for dist in distribution_choices:
            if dist in ["ordered_wienman_exponential", "gamma_exponential"]:
                    log_transformed=False
                    log_transformed_choice = get_user_choices(["No", "Yes"], "Do you want to use log-log transformed values?", multiple=False)
                    if log_transformed_choice=="Yes":
                       log_transformed=True
                    matching_files.extend(find_matching_files(directory_path, dist, mi_choice, log_transformed))
            else:
                log_transformed=False
                matching_files.extend(find_matching_files(directory_path, dist, mi_choice, log_transformed))

    else:
        if distribution_choice in ["ordered_wienman_exponential", "gamma_exponential"]:
                    log_transformed=False
                    log_transformed_choice = get_user_choices(["No", "Yes"], "Do you want to use log-log transformed values?", multiple=False)
                    if log_transformed_choice=="Yes":
                       log_transformed=True
                    matching_files = find_matching_files(directory_path, distribution_choice, mi_choice, log_transformed)
        else:
                log_transformed=False
                matching_files = find_matching_files(directory_path, distribution_choice, mi_choice, log_transformed)

    if matching_files:
        print("Files found:")
        for file in matching_files:
            print(file)
    else:
        print("No matching files found in the selected folder.")
    
    if figure_choice == "4":
        process_figure_4(matching_files, distribution_choice, mi_choice, log_transformed)
    if figure_choice == "4b":
        process_figure_4b(matching_files, distribution_choices, mi_choice, is_independent)



if __name__ == "__main__":
    main()
    while True:

        ask_for_another_plot = get_user_choices(["Yes", "No"], "Do you want to create another plot?", multiple=False) == "Yes"
        
        if not ask_for_another_plot:
            print("Exiting the plot creation process.")
            break

        main()