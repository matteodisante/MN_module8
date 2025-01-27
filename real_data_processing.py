import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from core.mutual_information_1 import *

def ensure_directory_exists(directory_path):
    """
    Checks if a directory exists, and creates it if it does not.

    :param directory_path: Path of the directory to check/create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory {directory_path} already existed. You might have overwritten the data!")


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

def real_data_processing_1(data_dict, name_dict, n, h, k):
    mi_list = []
    for name in data_dict.keys():
        # data
        data_array = data_dict[name]

        # number of windows
        i = 0
        while (i*h + n) <= len(data_array):
            i+=1
        i_max = i # actually we are interested in i-1, but for the subquent for cycle is reasonable to save i_max=i
        print(f'\nNumber of windows for {name} = {i_max}')

        # mi computation
        start = time.time()
        for j in range(i_max): # j=0,...,i_max-1
            mi_list.append(mutual_information_1(data_array[j*h: (j*h + n)], k=k)) # compute mi for each window
        end = time.time() - start
        print(f'Run time for {name}= {end/60} min')

    mi_array = np.array(mi_list)
    directory_path = f'data/real_data/n_{n}/h_{h}/k_{k}/{name_dict}'
    ensure_directory_exists(directory_path)
    file_name = f"mi_{name_dict}.txt"
    file_path = os.path.join(directory_path, file_name)
    np.savetxt(file_path, mi_array)


def real_data_processing_2(data_dict, name_dict, n, h, k):
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
        for j in range(i_max): # j=0,...,i_max-1
            mi_array[j] = mutual_information_1(data_array[j*h: (j*h + n)], k=k) # compute mi for each window

        # save reuslts
        directory_path = f'data/real_data/n_{n}/h_{h}/k_{k}/{name_dict}'
        ensure_directory_exists(directory_path)
        file_name = f"mi_{name}.txt"
        file_path = os.path.join(directory_path, file_name)
        np.savetxt(file_path, mi_array)

        end = time.time() - start
        print(f'run time for {name}= {end/60} min')


def get_user_parameters():
    """
    Prompt the user to input parameters for the process with default values provided.

    Parameters:
        None (all parameters are requested interactively).

    Returns:
        dict: A dictionary containing the following keys:
            - n (int): Number of data points (default is 5000).
            - h (int): Overlap parameter, based on user's choice between 'half overlapping' or 'no overlaps'.
            - k (int): Parameter for mutual information calculation (default is 1).
    """
    # Default values
    default_n = 5000
    default_k = 1

    # Prompt for 'n' with a default value
    try:
        n = int(input(f"\nEnter the value for 'n' (default {default_n}): ") or default_n)
    except ValueError:
        print(f"Invalid input. Using default value for 'n': {default_n}")
        n = default_n

    # Prompt for 'h' with options
    print("\nChoose the overlap type for 'h':")
    print("1 - Half overlapping")
    print("2 - No overlaps")
    try:
        h_choice = int(input("Enter your choice (1 or 2, default 1): ") or 1)
        if h_choice == 1:
            h = n // 2
        elif h_choice == 2:
            h = n // 1
        else:
            print("Invalid choice. Defaulting to 'half overlapping' (h = n // 2).")
            h = n // 2
    except ValueError:
        print("Invalid input. Defaulting to 'half overlapping' (h = n // 2).")
        h = n // 2

    # Prompt for 'k' with a default value
    try:
        k = int(input(f"\nEnter the value for 'k' (default {default_k}): ") or default_k)
    except ValueError:
        print(f"Invalid input. Using default value for 'k': {default_k}")
        k = default_k

    return n, h, k 


if __name__ == "__main__":
    # directory path
    directory_path = "data/real_data/raw_data"

    # import raw data
    data_dict_A, data_dict_C = import_raw_data(directory_path)
    raw_data = {'A': data_dict_A, 'C': data_dict_C}


    processing_choice = ['Processig A and C data', 'Processing every file A and C']

    while True:
        # Interactive selection of processing choice
        print("\nAvailable choices:")
        for i, choice_name in enumerate(processing_choice, start=1):
            print(f"{i}. {choice_name}")

        selected_choice = None
        while selected_choice is None:
            try:
                user_input = input("Select a choice by number: ").strip()
                selected_index = int(user_input) - 1
                if 0 <= selected_index < len(processing_choice):
                    selected_choice = processing_choice[selected_index]
                else:
                    print("Invalid selection. Please choose a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        print(f"\nSelected choice: {selected_choice}")

        if selected_index == 0:
                
            n, h, k = get_user_parameters()
            print(f"\nParameters: n={n}, h={h}, k={k}")
            real_data_processing_1(raw_data['A'], 'A', n, h, k)
            real_data_processing_1(raw_data['C'], 'C', n, h, k)


        elif selected_index == 1:

            n, h, k = get_user_parameters()
            print(f"\nParameters: n={n}, h={h}, k={k}")
            real_data_processing_2(raw_data['A'], 'A', n, h, k)
            real_data_processing_2(raw_data['C'], 'C', n, h, k)


        # Ask if the user wants to process more data or exit
        continue_choice = None
        while continue_choice not in ['y', 'n']:
            continue_choice = input("\nDo you want to process more data? (y/n): ").strip().lower()
        if continue_choice == 'n':
            print("Exiting program. Goodbye!")
            break