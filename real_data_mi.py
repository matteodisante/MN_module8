import os
import numpy as np
import pandas as pd
import time
from core.mutual_information_1 import *

def ensure_directory_exists(directory_path):
    """
    Checks if a directory exists, and creates it if it does not.

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

                print(f"File '{filename}' processed successfully.")

            except Exception as e:
                # Print a detailed error message
                print(f"Error processing file '{filename}': {str(e)}")


    return data_dict_A, data_dict_C



def get_valid_integer(prompt, default_value, min_value=None):
    """Helper function to validate integer input with a default value."""
    while True:
        user_input = input(f"{prompt} (default {default_value}): ").strip()
        if not user_input:
            return default_value
        try:
            value = int(user_input)
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}. Try again.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a valid integer.")



def get_user_parameters():
    """
    Prompt the user to input parameters for the process with input validation.

    Returns:
        tuple: (n, h) with valid integer values.
    """
    # Default values
    default_n = 5000

    # Prompt for 'n' with validation
    n = get_valid_integer("Enter the value for 'n'", default_n, min_value=1)

    # Prompt for 'h' with options and validation
    while True:
        print("\nChoose the overlap type for 'h':")
        print("1 - Half overlapping")
        print("2 - No overlaps")
        h_choice = input("Enter your choice (1 or 2, default 1): ").strip()
        if not h_choice:
            h = n // 2
            break
        elif h_choice == "1":
            h = n // 2
            break
        elif h_choice == "2":
            h = n
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    return n, h

def real_data_processing(data_dict, name_dict, n, h, k_list):
    for name in data_dict.keys():
        start = time.time()
        # data
        data_array = data_dict[name]

        # number of windows
        i = 0
        while (i*h + n) <= len(data_array):
            i+=1
        i_max = i  # actually we are interested in i-1, but for the subquent for cycle is reasonable to save i_max=i
        print(f'\nnumber of windows for {name} = {i_max}')

        # mi computation
        mi_array = np.zeros(i_max)
        mi_supp = np.zeros(len(k_list))

        for j in range(i_max): # j=0,...,i_max-1
            # compute mi for each window
            for l in range(len(k_list)):
                mi_supp[l] = mutual_information_1(data_array[j*h: (j*h + n)], k=k_list[l])
                
            if np.max(mi_supp) >= 0:
                mi_array[j] = np.max(mi_supp)
            else:
                mi_array[j] = 0

        # save reuslts
        directory_path = f'data/real_data/n_{n}/h_{h}/{name_dict}'
        ensure_directory_exists(directory_path)
        file_name = f"mi_{name}.txt"
        file_path = os.path.join(directory_path, file_name)
        np.savetxt(file_path, mi_array)

        end = time.time() - start
        print(f'run time for {name}= {end} sec')





if __name__ == "__main__":
    # directory path
    directory_path = "data/real_data/raw_data"

    # import raw data
    data_dict_A, data_dict_C = import_raw_data(directory_path)
    raw_data = {'A': data_dict_A, 'C': data_dict_C}

    # choose the values of k to explore
    k_list = [5, 10, 20, 30, 40, 50, 60]

    while True:
                
        n, h = get_user_parameters()
        print(f"\nParameters: n={n}, h={h}")
        real_data_processing(raw_data['A'], 'A', n, h, k_list)
        real_data_processing(raw_data['C'], 'C', n, h, k_list)


        # Ask if the user wants to process more data or exit
        continue_choice = None
        while continue_choice not in ['y', 'n']:
            continue_choice = input("\nDo you want to process more data? (y/n): ").strip().lower()
        if continue_choice == 'n':
            print("Exiting program. Goodbye!")
            break