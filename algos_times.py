import os
import sys
import time
import numpy as np
from scipy.special import digamma

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './utils/')))
from interface_utils import navigate_directories

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
    std_dev_time = np.std(times, ddof=1) / np.sqrt(n_runs)  # Standard error of the mean

    return mean_time, std_dev_time
    
    

if __name__ == "__main__":
    print("Select the file to analyze")
    file_path = navigate_directories(start_path=".", multi_select=False, file_extension=".txt")[0]
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)
    print(f"Selected file: {file_path}")
    
    dataset = np.loadtxt(file_path)
    n_samples = len(dataset)
    
    k_list_default = [1, 5, 25, 100, 500, 2500, 5000]
    bins_asked_per_axis_default = [2, 4, 8, 16, 32, 64, 128, 256]
    
    results = []
    
    for k in k_list_default:
        if k >= n_samples:
            print(f"Skipping k={k} as it is greater than or equal to the number of samples ({n_samples})")
            continue
        mi_1_time, mi_1_uncertainty = measure_execution_time(mutual_information_1, dataset, k)
        mi_sum_time, mi_sum_uncertainty = measure_execution_time(mutual_information_1_entropies_sum, dataset, k)
        results.append(f"k={k}, Mutual Information 1: {mi_1_time:.5f} ± {mi_1_uncertainty:.5f} sec")
        results.append(f"k={k}, Mutual Information 1 (Entropies Sum): {mi_sum_time:.5f} ± {mi_sum_uncertainty:.5f} sec")
    
    for bins_asked_per_axis in bins_asked_per_axis_default:
        mi_binned_time, mi_binned_uncertainty = measure_execution_time(mutual_information_binning_adaptive, dataset, bins_asked_per_axis)
        results.append(f"bins={bins_asked_per_axis}, Mutual Information (Binning Adaptive): {mi_binned_time:.5f} ± {mi_binned_uncertainty:.5f} sec")
    
    output_file = "execution_times.txt"
    with open(output_file, "w") as f:
        for line in results:
            print(line)
            f.write(line + "\n")
    
    print(f"Execution times saved in {output_file}")
