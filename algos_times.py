import os
import sys
import time
import numpy as np
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
    all_sizes = []
    
    for single_path in files_path:
        details = extract_file_details(single_path)
        files_path_detail.append(details)
        all_sizes.append(details["size"])
    
    dist_name = files_path_detail[0]["distribution_name"]
    params = files_path_detail[0]["params"]
    file_index = files_path_detail[0]["file_index"]
    
    size_list = list(set(all_sizes)) 
    k_list = [1, 5, 25, 100, 500, 2500, 5000]
    bins_list = [2, 4, 8, 16, 32, 64, 128, 256] 
    
    # Creazione delle matrici per i tempi di esecuzione e le relative incertezze (delta)
    mi1_time_matrix       = np.zeros((len(size_list), len(k_list)))
    d_mi1_time_matrix     = np.zeros((len(size_list), len(k_list)))
    misum_time_matrix     = np.zeros((len(size_list), len(k_list)))
    d_misum_time_matrix   = np.zeros((len(size_list), len(k_list)))
    mibinned_time_matrix  = np.zeros((len(size_list), len(bins_list)))
    d_mibinned_time_matrix= np.zeros((len(size_list), len(bins_list)))
    
    
    for i, size in enumerate(size_list):
        dataset_path = next(details['file_path'] for details in files_path_detail if details['size'] == size)
        print(dataset_path)
        dataset = np.loadtxt(dataset_path)
        dataset_len = dataset.shape[0]
        
        
        for j, k_value in enumerate(k_list):
            if k_value < dataset_len:
                mi1_time_matrix[i,j], d_mi1_time_matrix[i,j] = measure_execution_time(mutual_information_1, dataset, k_value)
            else:
                 mi1_time_matrix[i,j], d_mi1_time_matrix[i,j] = None, None
           
            for j, k_value in enumerate(k_list):
                if k_value < dataset_len:
                    misum_time_matrix[i,j], d_misum_time_matrix[i,j] = measure_execution_time(mutual_information_1_entropies_sum, dataset, k_value)
                else:
                    mi1_time_matrix[i,j], d_mi1_time_matrix[i,j] = None, None
                    
                
            for j, bins_value in enumerate(bins_list):
                mibinned_time_matrix[i,j], d_mibinned_time_matrix[i,j] = measure_execution_time(mutual_information_binning_adaptive, dataset, bins_value)
                
        
        output_files = ['mi1.txt', 'd_mi1.txt', 'misum.txt', 'd_misum.txt', 'mi_binned', 'd_mibinned']
        output_matrices = [mi1_time_matrix, d_mi1_time_matrix, misum_time_matrix, d_misum_time_matrix, mibinned_time_matrix, d_mibinned_time_matrix]
        
        for i, out_file in enumerate(output_files):
            np.savetxt(out_file, output_matrices[i])  
    
    
          
    
#    
#    dataset = np.loadtxt(file_path)
#    n_samples = len(dataset)
#
#    
#    results = []
#    
#    for k in k_list_default:
#        if k >= n_samples:
#            logging.warning(f"Skipping k={k} as it is greater than or equal to the number of samples ({n_samples})")
#            continue
#        mi_1_time, mi_1_uncertainty = measure_execution_time(mutual_information_1, dataset, k)
#        mi_sum_time, mi_sum_uncertainty = measure_execution_time(mutual_information_1_entropies_sum, dataset, k)
#        results.append(f"k={k}, Mutual Information 1: {mi_1_time:.5f} ± {mi_1_uncertainty:.5f} sec")
#        results.append(f"k={k}, Mutual Information 1 (Entropies Sum): {mi_sum_time:.5f} ± {mi_sum_uncertainty:.5f} sec")
#    
#    for bins_asked_per_axis in bins_asked_per_axis_default:
#        mi_binned_time, mi_binned_uncertainty = measure_execution_time(mutual_information_binning_adaptive, dataset, bins_asked_per_axis)
#        results.append(f"bins={bins_asked_per_axis}, Mutual Information (Binning Adaptive): {mi_binned_time:.5f} ± {mi_binned_uncertainty:.5f} sec")
#    
#    output_file = "execution_times.txt"
#    with open(output_file, "w") as f:
#        for line in results:
#            f.write(line + "\n")
#            logging.debug(line + "\n")
#    
#    logging.info(f"Execution times saved in {output_file}")
