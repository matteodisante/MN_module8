import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scipy.stats as stats
import re
import inspect

from utils.io_utils import ensure_directory_and_handle_file_conflicts, load_data_csv
from utils.config_utils import load_config
from utils.math_utils import circular_mi_theoretical, gamma_exponential_mi_theoretical, \
    correlated_gaussian_rv_mi_theoretical, independent_exponential_rv_mi_theoretical, \
        independent_gaussian_rv_mi_theoretical, independent_uniform_rv_mi_theoretical, \
            ordered_wienman_exponential_mi_theoretical


def plot_histograms(series1, series2, bins):
    """Disegna istogrammi sovrapposti per due serie temporali."""
    plt.figure(figsize=(10, 5))
    plt.hist(series1, bins=bins, alpha=0.5, label='Serie 1', color='blue', density=True)
    plt.hist(series2, bins=bins, alpha=0.5, label='Serie 2', color='green', density=True)
    plt.title('Istogrammi Sovrapposti')
    plt.xlabel('Valore')
    plt.ylabel('Densità')
    plt.legend()
    plt.show()


def plot_density(series1, series2=None, bins=50):
    """
    Generate a density plot for one or two series of data using KDE.

    Parameters:
        series1 (array-like): First data series.
        series2 (array-like, optional): Second data series (default: None).
        bins (int): Number of bins for the histogram (used for reference).
    """
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KDE for series1
    sns.kdeplot(series1, fill=True, label="Density - Series 1", color="blue", ax=ax)
    
    # Optionally plot KDE for series2
    if series2 is not None:
        sns.kdeplot(series2, fill=True, label="Density - Series 2", color="green", ax=ax)
    
    # Overlay histograms (optional, can be removed)
    ax.hist(series1, bins=bins, density=True, alpha=0.3, color="blue", label="Histogram - Series 1")
    if series2 is not None:
        ax.hist(series2, bins=bins, density=True, alpha=0.3, color="green", label="Histogram - Series 2")
    
    # Plot labels and legend
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Density Plot")
    ax.legend()
    
    plt.show()
    
    
    

def plot_3d_histogram(samples, title, output_path, bins=30):
    """Plotta un istogramma 3D con densità."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, density=True)
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    ax.bar3d(xpos.ravel(), ypos.ravel(), 0, 0.1, 0.1, hist.ravel(), shade=True)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


def plot_marginals(samples, output_path):
    """Plotta le marginali standard e in scala log-log."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))

    axes[0, 0].hist(samples[:, 0], bins=50, density=True)
    axes[0, 0].set_title("Marginale X")

    axes[0, 1].hist(samples[:, 1], bins=50, density=True)
    axes[0, 1].set_title("Marginale Y")

    axes[1, 0].hist(samples[:, 0], bins=50, density=True)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("Marginale X (semilogy)")

    axes[1, 1].hist(samples[:, 1], bins=50, density=True)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title("Marginale Y (log-log)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    
    
    


def generate_plot(
    file_paths, x_col, y_col, yerr_col, title, xlabel, ylabel, output_dir, 
    combine=False, theoretical_value=0, k_or_bins=None, x_transform=None
):
    """
    Generates plots for the provided datasets based on the configuration.

    Parameters:
        file_paths (list of str): Paths to the CSV files.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        yerr_col (str): Column name for error on y-axis.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        output_dir (str): Directory to save the plots.
        combine (bool): Whether to combine multiple curves in one plot.
        theoretical_value (float): Theoretical value to subtract from y_col.
        k_or_bins (list of int): Selected k or bins_number values.
        x_transform (callable): Optional transformation for x-axis values.
    """
    plt.style.use('seaborn-v0_8-darkgrid')

    # Group files by distribution and parameters
    grouped_files = {}
    for file in file_paths:
        distribution_key = '_'.join(os.path.basename(file).split('_')[2:-1])
        grouped_files.setdefault(distribution_key, []).append(file)

    for k_or_bin in k_or_bins:
        plt.figure(figsize=(10, 6))

        for group, files in grouped_files.items():
            curve_x, curve_y, curve_yerr = [], [], []

            for file in files:
                data = load_data_csv(file)
                if not all(col in data.columns for col in [x_col, y_col, yerr_col]):
                    print(f"[WARNING] Skipping file {file}: required columns missing.")
                    continue

                filtered_data = data[data[x_col] == k_or_bin]
                if filtered_data.empty:
                    print(f"[WARNING] Skipping {file}: k_or_bin {k_or_bin} not found in {x_col}.")
                    continue

                size = int(os.path.basename(file).split('_')[-1].replace('.csv', '').replace('size', ''))
                x_value = size if x_transform is None else x_transform(size)
                curve_x.append(x_value)
                curve_y.append(filtered_data[y_col].values[0] - theoretical_value)
                curve_yerr.append(filtered_data[yerr_col].values[0])

            sorted_indices = np.argsort(curve_x)
            curve_x, curve_y, curve_yerr = np.array(curve_x)[sorted_indices], np.array(curve_y)[sorted_indices], np.array(curve_yerr)[sorted_indices]

            plt.errorbar(curve_x, curve_y, yerr=curve_yerr, fmt="o", capsize=4, label=f"{group}")

        plt.legend(loc='upper right', fontsize=9, frameon=True, edgecolor="black")
        plt.title(f"{title} (k = {k_or_bin})", fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        combined_dir = os.path.join(output_dir, "combined", title.replace(' ', '_'), f"k_{k_or_bin}")
        ensure_directory_and_handle_file_conflicts(combined_dir)
        plt.savefig(os.path.join(combined_dir, f"figure_{title.replace(' ', '_')}_k{k_or_bin}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    



def load_config_distributions():
    """Loads the config.json file from the same directory as the script and returns the available distributions."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), "config.json")
    
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




def format_plot(mi_estimate="", figure_choice="4", is_independent=False, N_value=1, distribution_name='gaussian', param_name_mapping = {} ,distribution_choices=['gauss','exp'], selected_params={}):
    """Imposta le etichette e lo stile del plot."""
    subscript = {"mi_1": "1", "mi_sum": "sum", "mi_binning": "binning"}.get(mi_estimate, "")

    if figure_choice in ["4b"]:
        ylabel = (r"I$_{\mathrm{" + subscript + r"}}$" if is_independent else
                r"I$_{\mathrm{" + subscript + r"}}$/I$_{\mathrm{theoretical}}$")
    elif figure_choice=="4":
        ylabel = (r"I$_{\mathrm{" + subscript + r"}}$")
    elif figure_choice in ["20", "21"]:
        ylabel = (r"I$_{\mathrm{estimate}}$" if is_independent else
                r"I$_{\mathrm{estimate}}$/I$_{\mathrm{theoretical}}$")
    plt.ylabel(ylabel, fontsize=14)

    plt.axhline(y=(0.0 if is_independent else 1.0), color='r', linestyle='--', linewidth=0.5, label='_nolegend_')

    plt.xscale('log')
    if figure_choice=="4b":
        plt.xlabel(r"Number of non-zero cells" if mi_estimate == "mi_binning" else "k", fontsize=14)
    elif figure_choice=="4": 
        plt.xlabel(r"Number of non-zero cells/N" if mi_estimate == "mi_binning" else "k/N", fontsize=14)
    elif figure_choice in ["20", "7", "9"]:
        plt.xlabel(r"1/N", fontsize=14)
    elif figure_choice in ["8"]:
        plt.xlabel(r"N", fontsize=14)
    else:
        plt.xlabel(param_name_mapping, fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.6)
    if figure_choice in ["20", "21"]:
        plt.legend(title="MI Estimators", fontsize=11, loc='best')
    else:
       plt.legend(fontsize=11, loc='best')

    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    from matplotlib.ticker import FormatStrFormatter
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()

    if figure_choice=="4b":
        filename = f"figure_4b_{'_'.join(distribution_choices)}_{N_value}_{mi_estimate}.png"
    elif figure_choice=="4":
        filename = f"figure_4_{'_'.join({distribution_name})}_{selected_params}_{mi_estimate}.png"
    elif figure_choice=="20":
        filename = f"figure_20_{'_'.join({distribution_name})}_{selected_params}.png"
    elif figure_choice=="21":
        filename = f"figure_21_{'_'.join({distribution_name})}_{N_value}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Force rendering before showing the plot to prevent resizing issues
    plt.draw()

    plt.show()



def read_and_clean_data(file, mi_estimate):
    """Legge i dati dal file e li pulisce rimuovendo valori inf e NaN."""
    valid_data = []
    with open(file, 'r') as f:
        next(f)  # Salta l'intestazione
        for line in f:
            # Strip any leading/trailing whitespace
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Convert the line into a float array (split by whitespace or tab)
            row = np.fromstring(line, sep=' ')
            if row.size >= (7 if "binning" in mi_estimate.lower() else 4):
                col_idx = 4 if "binning" in mi_estimate.lower() else 1
                # Check if it is not infinity or NaN
                if not np.isinf(row[col_idx]) and not np.isnan(row[col_idx]):
                    valid_data.append(row) # Add the row to the list if it's valid
            else:
                print(f"Skipping line due to insufficient columns: {line}")
    return valid_data

def process_theoretical_mi(distribution_name, selected_params):
    """Calcola la mutua informazione teorica se disponibile."""
    theoretical_mi_function = globals().get(f"{distribution_name}_mi_theoretical")
    if theoretical_mi_function:
        if "independent" in distribution_name:
            return 0
        required_params = inspect.signature(theoretical_mi_function).parameters
        filtered_params = {key: float(value) for key, value in selected_params.items() if key in required_params}
        theoretical_mi = theoretical_mi_function(**filtered_params)
        print(f"Theoretical mutual information for {distribution_name}: {theoretical_mi}")
        return theoretical_mi
    print(f"No theoretical function found for {distribution_name}.")
    return None

def extract_N_values(files):
    """Estrae tutti i valori unici di N dai file."""
    N_values = sorted(set(int(re.search(r"size_(\d+)", os.path.basename(f)).group(1)) for f in files if re.search(r"size_(\d+)", os.path.basename(f))))
    if not N_values:
        print("No valid N values found..")
        return
    return N_values

def extract_file_info(file, distribution_name, mi_estimate, log_transformed):
    """Estrae i parametri e il valore di N dal nome del file."""
    pattern = (rf"summary_{distribution_name}_(.*?)_size_(\d+)_log_transformed_({mi_estimate}).txt"
               if log_transformed else
               rf"summary_{distribution_name}_(.*?)_size_(\d+)_({mi_estimate}).txt")
    
    match = re.search(pattern, os.path.basename(file))
    if match:
        return match.group(1), int(match.group(2))  # (parametro, N)
    return None, None

def select_and_filter_files(files, distribution_name, mi_estimate, log_transformed):
    """Seleziona i parametri e filtra i file in base alle scelte dell'utente."""
    all_parameters, extracted_params_per_file = extract_parameters_from_paths(files, distribution_name, mi_estimate, log_transformed)
    selected_params = {}

    for param_name, param_values in all_parameters.items():
        selected_params[param_name] = get_user_choices(list(param_values), f"Choose the value for {param_name}:", multiple=False)

    filtered_files = filter_files_by_parameters(extracted_params_per_file, selected_params)

    print("Selected files after filtering:")
    for file in filtered_files:
        print(file)

    return filtered_files, selected_params


def process_data_structure(valid_data, mi_estimate):
    """Organizza i dati in colonne e restituisce una struttura adatta per il plotting."""
    if "binning" in mi_estimate.lower():
        data_cleaned = np.array(valid_data).reshape(-1, 7)
        return data_cleaned[:, 0], data_cleaned[:, 2], data_cleaned[:, 1], data_cleaned[:, 3], data_cleaned[:, 4], data_cleaned[:, 6]  
    else:
        data_cleaned = np.array(valid_data).reshape(-1, 4)
        return data_cleaned[:, 0], None, None, None, data_cleaned[:, 1], data_cleaned[:, 3]  # xlolims e xuplims sono None
