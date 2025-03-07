import os
#os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import numpy as np
import pandas as pd
import logging
import time
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'core/')))
from mutual_information_1 import mutual_information_1
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))
from interface_utils import setup_logger


def ensure_directory_exists(directory_path):
    """
    Checks if a directory exists, and creates it if it does not.xxs

    :param directory_path: Path of the directory to check/create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"\n Directory created: {directory_path}")
    else:
        logging.info(f"\n Directory {directory_path} already existed. You might have overwritten the data!")


def import_raw_data(directory_path):

    # Dictionaries to store data
    data_dict_A = {}
    data_dict_C = {}


    # Check if the directory exists
    if not os.path.exists(directory_path):
        logging.error(f"The directory '{directory_path}' does not exist.")
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist.")

    # Scan files in the folder
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if it is a file and ignore "readme" (case insensitive)
        if os.path.isfile(file_path) and filename.lower() != "readme":
            try:
                # Read the data and convert it into a NumPy array
                data = pd.read_csv(file_path, sep=r'\s+', header=None).values

                # Determine the group (A or C) and save it in the appropriate dictionary
                group = 'A' if filename.startswith("A") else 'C'
                if group == 'A':
                    data_dict_A[filename] = data
                else:
                    data_dict_C[filename] = data

                logging.info(f"File '{filename}' processed successfully.")

            except Exception as e:
                # Print a detailed error message
                logging.error(f"Error processing file '{filename}': {str(e)}")


    return data_dict_A, data_dict_C



def real_data_processing(data_dict, name_dict, n, h, k):
    for name in data_dict.keys():
        start = time.time()
        # data
        data_array = data_dict[name]

        # number of windows
        i = 0
        while (i*h + n) <= len(data_array):
            i+=1
        i_max = i  # actually we are interested in i-1, but for the subquent for cycle is reasonable to save i_max=i
        logging.info(f'\n number of windows for {name} = {i_max}')
        # mi computation
        mi_array = np.zeros(i_max)
        supp = 0

        for j in range(i_max): # j=0,...,i_max-1
            # compute mi for each window
            supp = mutual_information_1(data_array[j*h: (j*h + n)], k=k)
            if supp >= 0:
                mi_array[j] = supp
            else:
                mi_array[j] = 0

        # save reuslts
        directory_path = f'data/real_data/n_{n}/h_{h}/mi1/k_{k}/{name_dict}'
        ensure_directory_exists(directory_path)
        file_name = f"mi_{name}.txt"
        file_path = os.path.join(directory_path, file_name)
        np.savetxt(file_path, mi_array)

        end = time.time() - start 
        logging.info(f'\n run time for {name} = {end} sec \n \n')





if __name__ == "__main__":

    setup_logger()
    logging.getLogger('core.mutual_information_1').setLevel(logging.WARNING)
    logging.getLogger('utils.core_utils').setLevel(logging.WARNING)

    # directory path
    directory_path = "data/real_data/raw_data"

    # import raw data
    data_dict_A, data_dict_C = import_raw_data(directory_path)
    raw_data = {'A': data_dict_A, 'C': data_dict_C}
    
    parser = argparse.ArgumentParser(description="Process real data with given parameters.")
    parser.add_argument("--n", type=int, default=5000, help="Value for parameter n (default: 5000)")
    parser.add_argument("--overlap", choices=['no', 'half'], required=True, help="Specify overlapping mode: 'no' or 'half'")
    parser.add_argument("--k", type=int, default=5, help="Value for parameter k (default: 50)")
    
    args = parser.parse_args()
    n = args.n
    h = n if args.overlap == 'no' else n // 2
    k = args.k
    logging.info(f"\nParameters: n={n}, h={h}, k={k}")

        
    real_data_processing(raw_data['A'], 'A', n, h, k)
    real_data_processing(raw_data['C'], 'C', n, h, k)


