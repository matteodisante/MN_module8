import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './utils/')))
from io_utils import ensure_directory_and_handle_file_conflicts

def generate_mi_files(output_dir, sizes, distributions):
    for dist in distributions:
        dist_name = dist["name"]
        dist_params = dist["params"]
        correlation = dist.get("correlation", None)  # Only for correlated_gaussian_rv
        
        for size in sizes:
            for mi_type in ["mi_1", "mi_sum", "mi_binning"]:
                # Prepare directory structure with parameter-based subfolder
                param_str = "_".join(f"{k}{v}" for k, v in dist_params.items())
                if correlation is not None:
                    param_str += f"_corr{correlation}"
                dist_dir = os.path.join(output_dir, mi_type, dist_name, param_str)

                # Check directory and file conflicts
                file_name = f"{mi_type}_{dist_name}_{param_str}_size{size}.csv"
                if not ensure_directory_and_handle_file_conflicts(dist_dir, file_name):
                    continue

                file_path = os.path.join(dist_dir, file_name)
                
                # Generate data based on MI type
                if mi_type == "mi_1":
                    data = {
                        "k": np.arange(1, 11),
                        "mean_mi_1": np.random.uniform(0.2, 0.8, 10),
                        "sigma_mi_1": np.random.uniform(0.01, 0.05, 10),
                    }
                elif mi_type == "mi_sum":
                    data = {
                        "k": np.arange(1, 11),
                        "mean_mi_sum": np.random.uniform(1.0, 2.0, 10),
                        "sigma_mi_sum": np.random.uniform(0.1, 0.2, 10),
                    }
                elif mi_type == "mi_binning":
                    data = {
                        "bins_number": np.arange(5, 15),
                        "mean_mi_binning": np.random.uniform(0.3, 0.7, 10),
                        "sigma_mi_binning": np.random.uniform(0.02, 0.05, 10),
                    }
                
                # Save to CSV
                pd.DataFrame(data).to_csv(file_path, index=False)
                print(f"[INFO] File generated: {file_path}")

if __name__ == "__main__":
    output_dir = "./data/mi_csv_example_data"
    sizes = [100, 250, 500, 1000, 2500]
    distributions = [
        {
            "name": "correlated_gaussian_rv",
            "params": {
                "mu": 0,
                "sigma": 1
            },
            "correlation": 0.9
        }
#        {
#            "name": "independent_gaussian_rv",
#            "params": {
#                "mu": 0,
#                "sigma": 1
#            }
#        },
#        {
#            "name": "independent_uniform_rv",
#            "params": {
#                "low": 0,
#                "high": 1
#            }
#        },
#        {
#            "name": "independent_exponential_rv",
#            "params": {
#                "lambda": 2
#            }
#        },
#        {
#            "name": "gamma_exponential",
#            "params": {
#                "theta": 5
#            }
#        },
#        {
#            "name": "ordered_wienman_exponential",
#            "params": {
#                "theta": 7
#            }
#        },
#        {
#            "name": "circular",
#            "params": {
#                "a": 1,
#                "b": 2,
#                "c": 3
#            }
#        }
    ]

    generate_mi_files(output_dir, sizes, distributions)