import os
import shutil 
import numpy as np
import pandas as pd


def load_data_csv(file_path, delimiter=','):
    try:
        return pd.read_csv(file_path, delimiter=delimiter)
    except Exception as e:
        print(f"[ERROR] Could not load file {file_path}: {e}")
        return None
    

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
    
    
    
def ensure_directory_and_handle_file_conflicts(dir_path, file_name=None):
    """
    Ensure the target directory exists and handle conflicts if a file with the same name exists.

    Parameters:
        dir_path (str): Path to the directory to ensure exists.
        file_name (str, optional): Name of the file to check for conflicts. If provided,
                                   the function checks if the file exists and prompts the user
                                   to decide whether to overwrite it.

    Returns:
        bool: True if the directory is prepared and the user wants to proceed (or no conflicts exist),
              False otherwise.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' has been created.")
    
    # If a file name is provided, check for conflicts
    if file_name:
        file_path = os.path.join(dir_path, file_name)
        if os.path.exists(file_path):
            response = input(f"File '{file_path}' already exists. Do you want to overwrite it? (yes/no): ").strip().lower()
            if response != 'yes':
                print(f"Operation cancelled for file: {file_path}")
                return False
    return True
    

    
    
def save_data(data, file_path):
    ensure_directory(os.path.dirname(file_path))
    np.savetxt(file_path, data, delimiter=',')
    print(f"Dati salvati in: {file_path}")
