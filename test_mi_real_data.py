import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'core/')))
from mutual_information_1 import mutual_information_1
from core.mutual_information_binning import mutual_information_binning_adaptive


def ensure_directory_exists(directory_path):
    """
    Checks if a directory exists, and creates it if it does not.xxs

    :param directory_path: Path of the directory to check/create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"\nDirectory created: {directory_path}")
    else:
        print(f"\nDirectory {directory_path} already existed. You might have overwritten the data!")


def import_raw_data(directory_path):
    # Dictionaries to store data
    data_dict_A = {}
    data_dict_C = {}

    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory '{directory_path}' does not exist.")
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist.")

    # Scan files in the folder
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if it is a file and ignore "readme" (case insensitive)
        if os.path.isfile(file_path) and "readme" not in filename.lower():
            try:
                # Read the data and convert it into a NumPy array
                data = pd.read_csv(file_path, sep=r'\s+', header=None, on_bad_lines='skip').values

                # Determine the group (A or C) and save it in the appropriate dictionary
                if filename.startswith("A"):
                    data_dict_A[filename] = data
                elif filename.startswith("C"):
                    data_dict_C[filename] = data
                else:
                    print(f"Warning: File '{filename}' does not match expected A/C group pattern. Skipping.")

                print(f"File '{filename}' processed successfully.")

            except pd.errors.ParserError as e:
                print(f"Error processing file '{filename}': {str(e)}")
                print("Possible cause: Malformed data (wrong number of fields per row).")
            except Exception as e:
                print(f"Unexpected error processing file '{filename}': {str(e)}")

    return data_dict_A, data_dict_C


def test_mi1(raw_data, n, h):

    name_A = ['A14F3-TE10034', 'A15F3-TE10034', 'A16F3-TE10034']
    name_C = ['C11F3-VE', 'C12F3-VE', 'C13F3-VE']
    name_dict = {'A': name_A, 'C': name_C}

    # choose the values of k to explore
    k_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200]

    for choice in ['A', 'C']:
        data_dict = raw_data[choice]

        for name in name_dict[choice]:
            data_array = data_dict[name]

            # number of windows
            i = 0
            while (i*h + n) <= len(data_array):
                i+=1
            i_max = i  # actually we are interested in i-1, but for the subquent for cycle is reasonable to save i_max=i
            print(f'\nNumber of windows for {name} = {i_max}')

            i_max=10

            # mi computation
            mi_supp = np.zeros(len(k_list))

            directory_path = f"test_real_data/mi1/n_{n}/h_{h}/{name}"
            ensure_directory_exists(directory_path)

            for j in range(i_max): # j=0,...,i_max-1
                # compute mi for each window
                for l in range(len(k_list)):
                    mi_supp[l] = mutual_information_1(data_array[j*h: (j*h + n)], k=k_list[l])

                # plot window
                plt.figure()
                plt.scatter(k_list, mi_supp)
                plt.xlabel('k')
                plt.ylabel('I')
                plt.title(f"mi1/n_{n}/h_{h}/{name}/window{j}")
                plt.grid()
                plt.savefig(f"test_real_data/mi1/n_{n}/h_{h}/{name}/window{j}.png")
                plt.close()


def test_binning(raw_data, n, h):

    name_A = ['A14F3-TE10034', 'A15F3-TE10034', 'A16F3-TE10034']
    name_C = ['C11F3-VE', 'C12F3-VE', 'C13F3-VE']
    name_dict = {'A': name_A, 'C': name_C}

    # choose the values of k to explore
    k_list = [8, 16, 24, 32, 64]

    for choice in ['A', 'C']:
        data_dict = raw_data[choice]

        for name in name_dict[choice]:
            data_array = data_dict[name]

            # number of windows
            i = 0
            while (i*h + n) <= len(data_array):
                i+=1
            i_max = i  # actually we are interested in i-1, but for the subquent for cycle is reasonable to save i_max=i
            print(f'\nNumber of windows for {name} = {i_max}')

            i_max=10

            # mi computation
            mi_supp = np.zeros(len(k_list))

            directory_path = f"test_real_data/binning/n_{n}/h_{h}/{name}"
            ensure_directory_exists(directory_path)

            for j in range(i_max): # j=0,...,i_max-1
                # compute mi for each window
                for l in range(len(k_list)):
                    mi_supp[l] = mutual_information_binning_adaptive(data_array[j*h: (j*h + n)], num_bins=k_list[l])[0]

                # plot window
                plt.figure()
                plt.scatter(k_list, mi_supp)
                plt.xlabel('k')
                plt.ylabel('I')
                plt.title(f"binning/n_{n}/h_{h}/{name}/window{j}")
                plt.grid()
                plt.savefig(f"test_real_data/binning/n_{n}/h_{h}/{name}/window{j}.png")
                plt.close()


if __name__ == "__main__":
    # directory path
    directory_path = "data/real_data/raw_data"

    # import raw data
    data_dict_A, data_dict_C = import_raw_data(directory_path)
    raw_data = {'A': data_dict_A, 'C': data_dict_C}

    n = 5000
    h = n // 2

    test_mi1(raw_data, n, h)
    test_binning(raw_data, n, h)

    