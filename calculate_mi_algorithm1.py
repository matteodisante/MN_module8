import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))
from io_utils import load_data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'core/')))
from mutual_information_1 import mutual_information_1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analizza due serie temporali da un file di testo.")
    parser.add_argument("file_path", type=str, help="Percorso del file di testo contenente le serie temporali.")
    args = parser.parse_args()

    data = load_data(args.file_path)
    print(data.shape)
    correlation = np.corrcoef(data.T)[0,1]
    print(correlation)
    print(1)
    mi_val_empirical = mutual_information_1(data, k = 15)
    print(2)
    mi_th = -0.5*np.log(1-correlation**2)
    print(f"Th. val = {mi_th}, Emp. val = {mi_val_empirical}")
    
    
