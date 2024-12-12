import sys
import numpy as np
import os 


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sampling')))
from univariate_generator import data_generator


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from io_utils import ensure_directory
from config_utils import load_config





if __name__ == "__main__":
    # Define distributions and their parameters
    distributions_to_generate = [
        {
            'distribution': 'gaussian',
            'size': int(3e7),
            'num_files': 3,
            'params': {'mu': 0, 'sigma': 1}
        }#,
#        {
#            'distribution': 'circular',
#            'size': 1000,
#            'num_files': 3,
#            'params': {'l': 0, 'm': 4, 'r':7}
#        }
    ]

    # Base directory setup: ../data/generated_data
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
    base_dir = os.path.join(base_data_dir, 'synthetic_data')

    # Ensure only the generated_data directory is checked
    if not ensure_directory(base_dir):
        print("Operation aborted. No data will be generated.")
        exit(1)

    # Generate and save data for each distribution
    for dist_config in distributions_to_generate:
        distribution = dist_config['distribution']
        size = dist_config['size']
        num_files = dist_config['num_files']
        params = dist_config['params']

        # Create a directory for this distribution inside generated_data
        dist_dir_path = os.path.join(base_dir, distribution)

        # Create the directory if it doesn't exist (no need for checks here)
        if not os.path.exists(dist_dir_path):
            os.makedirs(dist_dir_path)
            print(f"Directory '{dist_dir_path}' has been created.")

        # Generate and save data
        for i in range(num_files):
            data = data_generator(distribution, size, params, correlation=0.7, seed=None)
            file_name = f"{distribution}_size{size}_file{i+1}.txt"
            # Creazione di una stringa con i parametri
            param_str = "_".join(f"{k}{v}" for k, v in params.items())
            # Creazione del nome del file
            file_name = f"{distribution}_{param_str}_size{size}_file{i+1}.txt"
            save_data_to_txt(data, file_name, dist_dir_path)            
            
