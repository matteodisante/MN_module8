import os
import shutil 
import numpy as np
import pandas as pd


def load_data_csv(file_path, delimiter=','):
    return pd.read_csv(file_path, delimiter=delimiter)

def load_data(file_path):
    """Carica i dati da un file .txt, tentando automaticamente di determinare il delimitatore."""
    with open(file_path, 'r') as file:
        # Leggi la prima riga per determinare il delimitatore
        first_line = file.readline()
        if ',' in first_line:
            delimiter = ','
        elif '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = ' '  # Default: spazio
    return np.loadtxt(file_path, delimiter=delimiter)


def save_results(results, file_path):
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        f.write("Size,K,Mutual_Information\n")
        for size, k, mi in results:
            f.write(f"{size},{k},{mi}\n")
    print(f"Risultati salvati in: {file_path}")



def ensure_directory(dir_path):
    """
    Ensure the target directory exists. If it exists, clear its contents 
    (including files and subdirectories).

    Parameters:
        dir_path (str): Path to the directory.

    Returns:
        bool: True if the user wants to proceed, False otherwise.
    """
    if os.path.exists(dir_path):
        response = input(f"The directory '{dir_path}' already exists. Do you want to delete it? (yes/no): ").strip().lower()
        if response == 'yes':
            # Elimina completamente la directory e ricreala vuota
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
            print(f"Directory '{dir_path}' has been cleared and recreated.")
        else:
            print(f"Files in '{dir_path}' may be overwritten if files with the same names are generated.")
            proceed = input("Do you want to continue? (yes/no): ").strip().lower()
            if proceed != 'yes':
                print("Operation cancelled by the user.")
                return False
    else:
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' has been created.")
    return True
    
    

    
    
def save_data(data, file_path):
    ensure_directory(os.path.dirname(file_path))
    np.savetxt(file_path, data, delimiter=',')
    print(f"Dati salvati in: {file_path}")
