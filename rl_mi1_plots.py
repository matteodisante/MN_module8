import numpy as np
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.stats import ks_2samp
import argparse


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


def ensure_directory_exists2(directory_path):
    """
    Checks if a directory exists, and creates it if it does not.

    :param directory_path: Path of the directory to check/create.
    """
    if not os.path.exists(directory_path):
        print('\nThere are no data for these parameters!')
        return 0
    else:
        return 1

def import_data(file_path):
    # file A loading as numpy array
    try:

        data = np.loadtxt(file_path)
        print(f"File {file_path} successfully loaded")
        return data

    except Exception as e:
        print(f"Error while loading file: {e}")

def import_all_data(n, h):
    # A files
    file_path1 = f'data/real_data/n_{n}/h_{h}/mi1/A/mi_A14F3-TE10034.txt'
    data1 = import_data(file_path1)

    file_path2 = f'data/real_data/n_{n}/h_{h}/mi1/A/mi_A14F4-TE10034.txt'
    data2 = import_data(file_path2)

    file_path3 = f'data/real_data/n_{n}/h_{h}/mi1/A/mi_A15F3-TE10034.txt'
    data3 = import_data(file_path3)

    file_path4 = f'data/real_data/n_{n}/h_{h}/mi1/A/mi_A15F4-TE10034.txt'
    data4 = import_data(file_path4)

    file_path5 = f'data/real_data/n_{n}/h_{h}/mi1/A/mi_A16F3-TE10034.txt'
    data5 = import_data(file_path5)

    # C files
    file_path6 = f'data/real_data/n_{n}/h_{h}/mi1/C/mi_C11F3-VE.txt'
    data6 = import_data(file_path6)

    file_path7 = f'data/real_data/n_{n}/h_{h}/mi1/C/mi_C11F4-VE.txt'
    data7 = import_data(file_path7)

    file_path8 = f'data/real_data/n_{n}/h_{h}/mi1/C/mi_C12F3-VE.txt'
    data8 = import_data(file_path8)

    file_path9 = f'data/real_data/n_{n}/h_{h}/mi1/C/mi_C13F3-VE.txt'
    data9 = import_data(file_path9)

    file_path10 = f'data/real_data/n_{n}/h_{h}/mi1/C/mi_C13F4-VE.txt'
    data10 = import_data(file_path10)

    data = { 'A14F3': data1, 'A14F4':data2, 'A15F3':data3, 'A15F4':data4, 'A16F3': data5, 
            'C11F3':data6, 'C11F4':data7, 'C12F3':data8, 'C13F3':data9, 'C13F4': data10 }
    
    return data

def clear_file(file_path):
    """Clears the content of a text file."""
    with open(file_path, "w") as f:
        pass  # Opening in 'w' mode without writing anything clears the file

        
def two_sample_ks(mi_A, mi_C, n, h, label1, label2):
    alpha = 0.05

    # Two-sample Kolmogorov-Smirnov test
    test_statistics, p_value = ks_2samp(mi_A, mi_C)
    file_path = f'data/real_data/plots/n_{n}/h_{h}/mi1/ks_results.txt'
    # Scriviamo i risultati nel file di log
    with open(file_path, "a") as f:
        f.write(f"\n{label1}-{label2}\n")
        f.write(f"Statistic KS: {test_statistics}\n")
        f.write(f"p-value: {p_value}\n")

        if p_value < alpha:
            f.write(f"The two distributions are significantly different (at the {int(alpha * 100)}% significance level).\n")
        else:
            f.write(f"There is not enough evidence to conclude that the two distributions are different (at the {int(alpha * 100)}% significance level).\n")

def hist_2labels(n, h, data1, data2, label1, label2, directory_path):
    fontsize = 12
    title = 1
    figsize = (8, 6)

    plt.figure(figsize=figsize)
    if title == 0:
        plot_title = ''
    elif title == 1:
        plot_title = f'n_{n}/h_{h}/mi1'
    plt.title(plot_title, fontsize=fontsize)
    plt.hist(data1, bins=int(np.sqrt(len(data1))), density=True, label=label1)
    plt.hist(data2, bins=int(np.sqrt(len(data2))), density=True, alpha=0.6, label=label2)

    plt.xlabel('MI', fontsize=fontsize)
    plt.ylabel('density', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    # save plot
    file_name = f'{label1}_{label2}.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.savefig(file_path)
    plt.close()

def hist_3labels(n, h, data1, data2, data3, label1, label2, label3, directory_path):
    fontsize = 12
    title = 1
    figsize = (8, 6)

    plt.figure(figsize=figsize)
    if title == 0:
        plot_title = ''
    elif title == 1:
        plot_title = f'n_{n}/h_{h}/mi1'
    plt.title(plot_title, fontsize=fontsize)
    plt.hist(data1, bins=int(np.sqrt(len(data1))), density=True, label=label1)
    plt.hist(data2, bins=int(np.sqrt(len(data2))), density=True, alpha=0.6, label=label2)
    plt.hist(data3, bins=int(np.sqrt(len(data3))), density=True, alpha=0.6, label=label3)

    plt.xlabel('MI', fontsize=fontsize)
    plt.ylabel('density', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    # save plot
    file_name = f'{label1}_{label2}_{label3}.pdf'
    file_path = os.path.join(directory_path, file_name)
    plt.savefig(file_path)
    plt.close()

def same_mouse_diff_phase(n, h, data):
    file_path = f'data/real_data/plots/n_{n}/h_{h}/mi1/ks_results.txt'
    # Scriviamo i risultati nel file di log
    with open(file_path, "a") as f:
        f.write(f"\nSame mouse but different phase:")

    directory_path = f'data/real_data/plots/n_{n}/h_{h}/mi1/same_mouse_diff_phase'
    ensure_directory_exists(directory_path)

    ## A14F3 - A14F4 ##
    label1 = 'A14F3'
    label2 = 'A14F4'

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

    ## A15F3 - A15F4 ##
    label1 = 'A15F3'
    label2 = 'A15F4'

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

    ## C11F3 - C11F4 ##
    label1 = 'C11F3'
    label2 = 'C11F4'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

    ## C13F3 - C13F4 ##
    label1 = 'C13F3'
    label2 = 'C13F4'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

def diff_mouses_same_class(n, h, data):
    file_path = f'data/real_data/plots/n_{n}/h_{h}/mi1/ks_results.txt'
    # Scriviamo i risultati nel file di log
    with open(file_path, "a") as f:
        f.write(f"\nDifferent mouses but same class (A/C):")

    directory_path = f'data/real_data/plots/n_{n}/h_{h}/mi1/diff_mouses_same_class'
    ensure_directory_exists(directory_path)

    ## A14F3 - A15F3 ##
    label1 = 'A14F3'
    label2 = 'A15F3'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

    ## A14F3 - A16F3 ##
    label1 = 'A14F3'
    label2 = 'A16F3'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

    ## A15F3 - A16F3 ##
    label1 = 'A15F3'
    label2 = 'A16F3'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

    ## SUMMARY A ##
    label1 = 'A14F3'
    label2 = 'A15F3'
    label3 = 'A16F3'

    hist_3labels(n, h, data[label1], data[label2], data[label3], label1, label2, label3, directory_path)

    ## C11F3 - C12F3 ##
    label1 = 'C11F3'
    label2 = 'C12F3'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

    ## C13F3 - C12F3 ##
    label1 = 'C13F3'
    label2 = 'C12F3'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)
    ## C11F3 - C13F3 ##
    label1 = 'C11F3'
    label2 = 'C13F3'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

    ## SUMMARY C ##
    label1 = 'C11F3'
    label2 = 'C13F3'
    label3 = 'C12F3'

    hist_3labels(n, h, data[label1], data[label2], data[label3], label1, label2, label3, directory_path)

def diff_mouses_diff_classes(n, h, data):
    file_path = f'data/real_data/plots/n_{n}/h_{h}/mi1/ks_results.txt'
    with open(file_path, "a") as f:
        f.write(f"\nDifferent mouses from different classes:")

    directory_path = f'data/real_data/plots/n_{n}/h_{h}/mi1/diff_mouses_diff_classes'
    ensure_directory_exists(directory_path)

    ## A14F3 - C11F3 ##
    label1 = 'A14F3'
    label2 = 'C11F3'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

    ## A14F3 - C12F3 ##
    label1 = 'A14F3'
    label2 = 'C12F3'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

    ## A15F3 - C11F3 ##
    label1 = 'A15F3'
    label2 = 'C11F3'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

    ## A15F3 - C12F3 ##
    label1 = 'A15F3'
    label2 = 'C12F3'  

    two_sample_ks(data[label1], data[label2], n, h, label1, label2)
    hist_2labels(n, h, data[label1], data[label2], label1, label2, directory_path)

def compute_median_matrix(data_dict):
    """Computes a matrix where each element is the absolute difference between medians of selected datasets."""
    
    # Exclude specified keys
    exclude_keys = {'A14F4', 'A15F4', 'C11F4', 'C13F4'}
    
    # Keep only the keys that are not in exclude_keys
    filtered_data = OrderedDict((key, value) for key, value in data_dict.items() if key not in exclude_keys)
    
    # Compute medians
    medians = {key: np.median(value) for key, value in filtered_data.items() if value is not None}
    
    # Create an empty matrix
    size = len(filtered_data)
    median_matrix = np.zeros((size, size))

    # Fill the matrix with absolute differences of medians
    keys = list(filtered_data.keys())
    for i in range(size):
        for j in range(size):
            median_matrix[i, j] = abs(medians[keys[i]] - medians[keys[j]])

    return median_matrix, keys

def plot_heatmap(matrix, labels, n, h):
    """Plots a heatmap for the given matrix while keeping the original order."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(matrix, cmap="viridis", interpolation="nearest")

    # Add colorbar
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_title(f"n_{n}/h_{h}/mi1 Median Absolute Differences")
    plt.savefig(f'data/real_data/plots/n_{n}/h_{h}/mi1/heatmap.pdf')
    plt.close()

if __name__ == "__main__":
    print("Welcome!\nThis is a script to generate plots for real data processed with mi1 algorithm.")

    while True:

        check_par = 0
        while check_par == 0:
            n, h = get_user_parameters()
            print(f"\nParameters: n={n}, h={h}")
            check_par = ensure_directory_exists2(f'data/real_data/n_{n}/h_{h}/mi1')

    

        data = import_all_data(n, h)
        # create a directory to save plots we are going to generate
        ensure_directory_exists(f'data/real_data/plots/n_{n}/h_{h}/mi1')
        # clean ks_results.txt
        clear_file(f'data/real_data/plots/n_{n}/h_{h}/mi1/ks_results.txt')
        # same mouse but different phase
        same_mouse_diff_phase(n, h, data)
        # different mouses but same class (A/C)
        diff_mouses_same_class(n, h, data)
        # different mouses from different classes
        diff_mouses_diff_classes(n, h, data)
        # Compute matrix and plot heatmap
        matrix, labels = compute_median_matrix(data)
        plot_heatmap(matrix, labels, n, h)

        print(f'\nPlots for algorithm=mi1, n={n}, h={h} have been created!')

        # Ask if the user wants to generate more plots or exit
        continue_choice = None
        while continue_choice not in ['y', 'n']:
            continue_choice = input("\nDo you want to generate more plots? (y/n): ").strip().lower()
        if continue_choice == 'n':
            print("Exiting program. Goodbye!")
            break