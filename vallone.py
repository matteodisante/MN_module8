import os
import numpy as np
import time
import pandas as pd
import sys
import logging
import argparse

# Configurazione del logging
logging.basicConfig(
    filename="logfile_vallone.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Separatore tra le esecuzioni
logging.info("\n================ New Execution =================\n")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'core/')))
from mutual_information_1 import mutual_information_1

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Directory created: {directory_path}")
    else:
        logging.info(f"Directory {directory_path} already existed. You might have overwritten the data!")

def import_raw_data(directory_path):
    data_dict_A = {}
    data_dict_C = {}

    if not os.path.exists(directory_path):
        logging.error(f"The directory '{directory_path}' does not exist.")
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist.")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path) and "readme" not in filename.lower():
            try:
                data = pd.read_csv(file_path, sep=r'\s+', header=None, on_bad_lines='skip').values

                if filename.startswith("A"):
                    data_dict_A[filename] = data
                elif filename.startswith("C"):
                    data_dict_C[filename] = data
                else:
                    logging.warning(f"File '{filename}' does not match expected A/C group pattern. Skipping.")

                logging.info(f"File '{filename}' processed successfully.")

            except pd.errors.ParserError as e:
                logging.error(f"Error processing file '{filename}': {str(e)}")
                logging.error("Possible cause: Malformed data (wrong number of fields per row).")
            except Exception as e:
                logging.error(f"Unexpected error processing file '{filename}': {str(e)}")

    return data_dict_A, data_dict_C

def f_corr1(max_lag, data):
    x = data[:,0]
    y = data[:,1]
    n = len(x)

    corr = np.correlate(x-np.mean(x), y-np.mean(y), mode="full") / (n * np.std(x) * np.std(y))
    corr = corr[(n - 1) - max_lag : (n - 1) + max_lag + 1]
    return np.abs(corr)

def f_corr0(raw_data, n, h, max_lag):
    if h == n // 2:
        overlap = 'half_overlap'
    elif h == n // 1:
        overlap = 'no_overlap'

    directory_path = f"test_vallone/n_{n}/{overlap}/corr/num_data"
    ensure_directory_exists(directory_path)

    name_A = ['A14F3-TE10034', 'A15F3-TE10034', 'A16F3-TE10034']
    name_C = ['C11F3-VE', 'C12F3-VE', 'C13F3-VE']
    name_dict = {'A': name_A, 'C': name_C}

    for choice in ['A', 'C']:
        data_dict = raw_data[choice]

        for name in name_dict[choice]:
            data_array = data_dict[name]

            i = 0
            while (i*h + n) <= len(data_array):
                i += 1
            i_max = i
            logging.info(f'Number of windows for {name} = {i_max}')

            corr_matrix = np.zeros((i_max, (2*max_lag+1)))

            start = time.time()
            for j in range(i_max):
                window_data = data_array[j*h: (j*h + n)]
                corr_matrix[j,:] = f_corr1(max_lag, window_data)

            file_path = f"test_vallone/corr/num_data/{name}.txt"
            np.savetxt(file_path, corr_matrix, fmt="%r", delimiter=" ")

            time_file = time.time() - start
            logging.info(f"Time to process {name} = {time_file} sec")

def f2_plus(x, y):
    data = np.column_stack((x, y))

    k_inf = 6
    k_sup = 50
    k_arr = np.arange(k_inf, k_sup+1)

    mi_supp = np.zeros(len(k_arr))

    for l in range(len(k_arr)):
        mi_supp[l] = np.maximum(mutual_information_1(data, k=k_arr[l]), 0)

    return np.median(mi_supp)

def f2(x, y, k):
    data = np.column_stack((x, y))
    return np.maximum(mutual_information_1(data, k=k), 0)

def f1(data, max_lag, k):
    x = data[:,0]
    y = data[:,1]
    n_lag = 2*max_lag+1
    mi_arr = np.zeros(n_lag)

    lag = np.arange(-max_lag, max_lag+1)
    for i in range(n_lag):
        if lag[i] > 0:
            mi_arr[i] = f2(x[lag[i]:], y[:-lag[i]], k)
        elif lag[i] < 0:
            mi_arr[i] = f2(x[:lag[i]], y[-lag[i]:], k)
        else:
            mi_arr[i] = f2(x, y, k)

    return mi_arr

def f0(raw_data, n, h, k, max_lag):
    if h == n // 2:
        overlap = 'half_overlap'
    elif h == n // 1:
        overlap = 'no_overlap'

    directory_path = f"test_vallone/n_{n}/{overlap}/mi/k_{k}/num_data"
    ensure_directory_exists(directory_path)          

    name_A = ['A14F3-TE10034', 'A15F3-TE10034', 'A16F3-TE10034']
    name_C = ['C11F3-VE', 'C12F3-VE', 'C13F3-VE']
    name_dict = {'A': name_A, 'C': name_C}

    for choice in ['A', 'C']:
        data_dict = raw_data[choice]

        for name in name_dict[choice]:
            data_array = data_dict[name]

            i = 0
            while (i*h + n) <= len(data_array):
                i += 1
            i_max = i
            logging.info(f'Number of windows for {name} = {i_max}')

            mi_matrix = np.zeros((i_max, (2*max_lag+1)))

            start = time.time()
            for j in range(i_max):
                window_data = data_array[j*h: (j*h + n)]
                mi_matrix[j,:] = f1(window_data, max_lag, k)

            file_path = f"test_vallone/n_{n}/{overlap}/mi/k_{k}/num_data/{name}.txt"
            np.savetxt(file_path, mi_matrix, fmt="%r", delimiter=" ")

            time_file = time.time() - start
            logging.info(f"Time to process {name} = {time_file} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process real data following the approach of Vallone et al (2016 ).")
    parser.add_argument("--n", type=int, required=True, help="Value for parameter n")
    parser.add_argument("--overlap", choices=['no', 'half'], required=True, help="Specify overlap choice")
    parser.add_argument("--coef", choices=['corr', 'mi'], required=True, help="Specify coefficient to analyze data: 'corr' or 'mi'")
    parser.add_argument("--k", type=int, default=25, help="Value for parameter k (default: 25)")

    args = parser.parse_args()
    n = args.n
    overlap = args.overlap
    h = {'half': n // 2, 'no': n // 1}
    coef = args.coef
    k = args.k

    max_lag = 30

    directory_path = "data/real_data/raw_data"
    data_dict_A, data_dict_C = import_raw_data(directory_path)
    raw_data = {'A': data_dict_A, 'C': data_dict_C}

    if coef == 'mi':
        f0(raw_data, n, h[overlap], k, max_lag)
    elif coef == 'corr':
        f_corr0(raw_data, n, h[overlap], max_lag)
