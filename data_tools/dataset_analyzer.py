import sys
import os
import numpy as np
import logging
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/')))
from interface_utils import navigate_directories
from io_utils import load_data, save_data


def get_output_file_path_for_mean_statistics(original_file_path):
    """
    Generates the corresponding output file path for a given input file, 
    keeping the base directory '../data/synthetic_data_statistics/' 
    and changing only the output file name.

    :param original_file_path: Path to the input file.
    :return: Path to the corresponding output file.
    """
    # Base directory for the output files
    base_output_dir = "../data/synthetic_data_statistics"
    
    # Extract the base name of the input file and its directory path
    file_name = os.path.basename(original_file_path)
    file_dir = os.path.dirname(original_file_path)
    
    # Modify the file name to add the '_mean_statistics' suffix
    output_file_name = file_name.replace("01.txt", "mean_statistics_over_files.txt")
    
    # Ensure the output directory exists
    output_dir = os.path.join(base_output_dir, os.path.relpath(file_dir, "../data/synthetic_data/"))
    os.makedirs(output_dir, exist_ok=True)

    # Construct the final output file path
    output_file_path = os.path.join(output_dir, output_file_name)

    return output_file_path


def compute_percentile_interval(data, lower_percentile=5, upper_percentile=95):
    """Calcola l'intervallo che contiene il 90% dei dati, usando i percentili."""
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    
    return lower_bound, upper_bound
    
    

def compute_statistics(data):
    """Calcola statistiche per una singola matrice di dati."""
    x, y = data[:, 0], data[:, 1]
    
    stats = {
        "median_x": np.median(x), "median_y": np.median(y),
        "mean_x": np.mean(x), "mean_y": np.mean(y),
        "std_x": np.std(x, ddof=1), "std_y": np.std(y, ddof=1),
        "min_x": np.min(x), "min_y": np.min(y),
        "max_x": np.max(x), "max_y": np.max(y),
        "percentile_x": compute_percentile_interval(x),
        "percentile_y": compute_percentile_interval(y)
    }
    
    return stats
    
    

def f_compute_corr_of_n_files(data_list):
    """Calcola la correlazione per ogni dataset."""
    n_files = len(data_list)
    corr_array = np.zeros(n_files)
    
    for idx in range(n_files):
        if data_list[idx].shape[1] < 2:  # Controlla che ci siano almeno 2 colonne
            continue
        corr_array[idx] = np.corrcoef(data_list[idx][:, 0], data_list[idx][:, 1])[0, 1]
    
    return corr_array



if __name__ == '__main__': 
    file_paths = []  # Inizializza la variabile
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    while not file_paths:  # Se nessun file è selezionato, continua a chiedere
        file_paths = navigate_directories(start_path="../data/synthetic_data/", multi_select=True, file_extension=".txt")
    
    logging.info('Loading files')
    data_list = [load_data(file) for file in file_paths]
    
    if not data_list:
        logging.error("No valid files loaded. Exiting.")
        sys.exit(1)

    logging.info('All files loaded. Proceeding with correlation computation.')
    corr_array = f_compute_corr_of_n_files(data_list)

    # Calcolare statistiche medie globali
    mean_stats = {
        "mean_corr": np.mean(corr_array),
        "std_mean_corr": np.std(corr_array)/np.sqrt(len(data_list)),
        "mean_median_x": np.mean([compute_statistics(data)["median_x"] for data in data_list]),
        "mean_median_y": np.mean([compute_statistics(data)["median_y"] for data in data_list]),
        "mean_mean_x": np.mean([compute_statistics(data)["mean_x"] for data in data_list]),
        "mean_mean_y": np.mean([compute_statistics(data)["mean_y"] for data in data_list]),
        "mean_std_x": np.mean([compute_statistics(data)["std_x"] for data in data_list]),
        "mean_std_y": np.mean([compute_statistics(data)["std_y"] for data in data_list]),
        "mean_min_x": np.mean([compute_statistics(data)["min_x"] for data in data_list]),
        "mean_min_y": np.mean([compute_statistics(data)["min_y"] for data in data_list]),
        "mean_max_x": np.mean([compute_statistics(data)["max_x"] for data in data_list]),
        "mean_max_y": np.mean([compute_statistics(data)["max_y"] for data in data_list]),
        "mean_percentile_x_lower": np.mean([compute_statistics(data)["percentile_x"][0] for data in data_list]),
        "mean_percentile_x_upper": np.mean([compute_statistics(data)["percentile_x"][1] for data in data_list]),
        "mean_percentile_y_lower": np.mean([compute_statistics(data)["percentile_y"][0] for data in data_list]),
        "mean_percentile_y_upper": np.mean([compute_statistics(data)["percentile_y"][1] for data in data_list])
    }

    # Creazione della struttura di directory di output
    output_dir_base = "../data/synthetic_data_statistics/"


    # Call the get_output_file_path_for_mean_statistics to generate the path dynamically
    output_path = get_output_file_path_for_mean_statistics(file_paths[0])

    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
            
    
    with open(output_path, "w") as f:
        f.write(f"Mean Correlation: {mean_stats['mean_corr']:.5f} +- {mean_stats['std_mean_corr']:.5f}\n")
        f.write(f"Mean Median X: {mean_stats['mean_median_x']:.5f}\n")
        f.write(f"Mean Median Y: {mean_stats['mean_median_y']:.5f}\n")
        f.write(f"Mean Mean X: {mean_stats['mean_mean_x']:.5f}\n")
        f.write(f"Mean Mean Y: {mean_stats['mean_mean_y']:.5f}\n")
        f.write(f"Mean Std X: {mean_stats['mean_std_x']:.5f}\n")
        f.write(f"Mean Std Y: {mean_stats['mean_std_y']:.5f}\n")
        f.write(f"Mean Min X: {mean_stats['mean_min_x']:.5f}\n")
        f.write(f"Mean Min Y: {mean_stats['mean_min_y']:.5f}\n")
        f.write(f"Mean Max X: {mean_stats['mean_max_x']:.5f}\n")
        f.write(f"Mean Max Y: {mean_stats['mean_max_y']:.5f}\n")
        f.write(f"Mean Percentile X (Lower): {mean_stats['mean_percentile_x_lower']:.5f}\n")
        f.write(f"Mean Percentile X (Upper): {mean_stats['mean_percentile_x_upper']:.5f}\n")
        f.write(f"Mean Percentile Y (Lower): {mean_stats['mean_percentile_y_lower']:.5f}\n")
        f.write(f"Mean Percentile Y (Upper): {mean_stats['mean_percentile_y_upper']:.5f}\n")

    logging.info(f"Results saved in {output_path}")
    print(f'Results saved in {output_path}')