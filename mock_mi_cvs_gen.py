import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './utils/')))
from io_utils import ensure_directory

# Funzione per generare dataset di esempio
def generate_example_datasets(output_dir):
    ensure_directory(output_dir)

    # Dataset tipo 1ccot 
    data1 = {
        "k": np.arange(1, 11),
        "mean_mi_1": np.random.uniform(0.2, 0.8, 10),
        "sigma_mi_1": np.random.uniform(0.01, 0.05, 10),
    }
    pd.DataFrame(data1).to_csv(
        os.path.join(output_dir, "mi_1_gamma_exponential_theta5_size_1000.csv"),
        index=False,
    )

    # Dataset tipo 2
    data2 = {
        "k": np.arange(1, 11),
        "mean_mi_sum": np.random.uniform(1.0, 2.0, 10),
        "sigma_mi_sum": np.random.uniform(0.1, 0.2, 10),
    }
    pd.DataFrame(data2).to_csv(
        os.path.join(output_dir, "mi_sum_gamma_exponential_theta5_size_1000.csv"),
        index=False,
    )

    # Dataset tipo 3
    data3 = {
        "bin_size": np.linspace(0.1, 1.0, 10),
        "mean_mi_binning": np.random.uniform(0.3, 0.7, 10),
        "sigma_mi_binning": np.random.uniform(0.02, 0.05, 10),
    }
    pd.DataFrame(data3).to_csv(
        os.path.join(output_dir, "mi_binning_gamma_exponential_theta5_size_1000.csv"),
        index=False,
    )
    
    
if __name__ == "__main__":
    example_data_dir = "./data/mi_csv_example_data"
    generate_example_datasets(example_data_dir)


