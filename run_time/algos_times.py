import os
import sys
import time
import numpy as np
import pandas as pd
import logging
from scipy.special import digamma

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './utils/')))
from interface_utils import navigate_directories, setup_logger
from io_utils import extract_file_details

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './core/')))
from mutual_information_1 import mutual_information_1
from mutual_information_1_entropies_sum import mutual_information_1_entropies_sum
from mutual_information_binning import mutual_information_binning_adaptive 



def measure_execution_time(func, *args, n_runs=10, warm_up_runs=3):
    """
    Measures the execution time of a function using high-precision timers.
    
    Parameters:
        func (callable): Function to be timed.
        *args: Arguments to pass to the function.
        n_runs (int): Number of repetitions for timing.
        warm_up_runs (int): Number of warm-up executions before measuring.

    Returns:
        tuple: (mean execution time, standard deviation)
    """
    # Warm-up phase
    for _ in range(warm_up_runs):
        func(*args)

    # Measure execution time using time.perf_counter()
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        func(*args)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    # Compute statistics
    mean_time = np.mean(times)
    std_dev_mean_time = np.std(times, ddof=1) / np.sqrt(n_runs)  # Standard error of the mean

    return mean_time, std_dev_mean_time
    
    
    
    

if __name__ == "__main__":
    setup_logger()
    logging.getLogger('core.mutual_information_1').setLevel(logging.WARNING)
    logging.getLogger('utils.core_utils').setLevel(logging.WARNING)

    
    logging.info("Select the file to analyze")
    files_path = navigate_directories(start_path="data/synthetic_data/", multi_select=True, file_extension=".txt")
    if not files_path:
        logging.warning("No file selected. Exiting.")
        sys.exit(0)
    
    files_path_detail = []
    
    for single_path in files_path:
        details = extract_file_details(single_path)
        details["size"] = int(details["size"]) 
        files_path_detail.append(details)
        

    all_sizes_int = sorted({ d["size"] for d in files_path_detail })
    dist_name = files_path_detail[0]["distribution_name"]
    params = files_path_detail[0]["params"]
    file_index = files_path_detail[0]["file_index"]
     
     
    k_list = [1, 5, 25, 100, 500, 2500, 5000]
    bins_list = [2, 4, 8, 16, 32, 64, 128, 256] 
    
    # Creazione delle matrici per i tempi di esecuzione e le relative incertezze (delta)
    mi1_time_matrix       = np.zeros((len(all_sizes_int), len(k_list)))
    d_mi1_time_matrix     = np.zeros((len(all_sizes_int), len(k_list)))
    misum_time_matrix     = np.zeros((len(all_sizes_int), len(k_list)))
    d_misum_time_matrix   = np.zeros((len(all_sizes_int), len(k_list)))
    mibinned_time_matrix  = np.zeros((len(all_sizes_int), len(bins_list)))
    d_mibinned_time_matrix= np.zeros((len(all_sizes_int), len(bins_list)))
    
    
    for i, size in enumerate(all_sizes_int):
        dataset_path = next(d['file_path'] for d in files_path_detail if d['size'] == size and d['file_index'] == "01")
        print(dataset_path)
        dataset = np.loadtxt(dataset_path)
        dataset_len = dataset.shape[0]
        
        
        for j, k_value in enumerate(k_list):
            if k_value < dataset_len:
                mi1_time_matrix[i,j], d_mi1_time_matrix[i,j] = measure_execution_time(mutual_information_1, dataset, k_value)
            else:
                 mi1_time_matrix[i,j], d_mi1_time_matrix[i,j] = np.nan, np.nan
           
        for j, k_value in enumerate(k_list):
            if k_value < dataset_len:
                misum_time_matrix[i,j], d_misum_time_matrix[i,j] = measure_execution_time(mutual_information_1_entropies_sum, dataset, k_value)
            else:
                misum_time_matrix[i,j], d_misum_time_matrix[i,j] = np.nan, np.nan
                    
                
        for j, bins_value in enumerate(bins_list):
            mibinned_time_matrix[i,j], d_mibinned_time_matrix[i,j] = measure_execution_time(mutual_information_binning_adaptive, dataset, bins_value)
                
        
   
        # Crea la cartella di output seguendo la struttura: computation_times/<dist_name>/<params>/
        output_folder = os.path.join("computation_times", dist_name, params)
        os.makedirs(output_folder, exist_ok=True)
        
        # Creazione dei DataFrame e salvataggio in file nella cartella di output
        df_mi1_time = pd.DataFrame(mi1_time_matrix, index=all_sizes_int, columns=k_list)
        df_mi1_time.to_csv(os.path.join(output_folder, "mi1.txt"), sep="\t")
        
        df_d_mi1_time = pd.DataFrame(d_mi1_time_matrix, index=all_sizes_int, columns=k_list)
        df_d_mi1_time.to_csv(os.path.join(output_folder, "d_mi1.txt"), sep="\t")
        
        df_misum_time = pd.DataFrame(misum_time_matrix, index=all_sizes_int, columns=k_list)
        df_misum_time.to_csv(os.path.join(output_folder, "misum.txt"), sep="\t")
        
        df_d_misum_time = pd.DataFrame(d_misum_time_matrix, index=all_sizes_int, columns=k_list)
        df_d_misum_time.to_csv(os.path.join(output_folder, "d_misum.txt"), sep="\t")
        
        df_mibinned_time = pd.DataFrame(mibinned_time_matrix, index=all_sizes_int, columns=bins_list)
        df_mibinned_time.to_csv(os.path.join(output_folder, "mi_binned.txt"), sep="\t")
        
        df_d_mibinned_time = pd.DataFrame(d_mibinned_time_matrix, index=all_sizes_int, columns=bins_list)
        df_d_mibinned_time.to_csv(os.path.join(output_folder, "d_mibinned.txt"), sep="\t")
    
    logging.info("Computation times saved in folder: " + output_folder)
        
        

    
