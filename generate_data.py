import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sampling/')))
from univariate_generator import generate_univariate_data
from multivariate_generator import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))
# Utility functions for file management and configuration loading
from io_utils import save_data
from config_utils import load_config

# Plotting utilities for visualizing data
from plot_utils import (
    plot_3d_histogram,
    plot_marginals
)

# Mathematical functions and target distributions
from math_utils import (
    configure_target_function,       # Function to configure target distributions
    gaussian_target_distribution,    # Gaussian distribution function
    gamma_exponential_target_distribution,  # Gamma-exponential target distribution
    weinman_ordered_target_distribution,    # Ordered Weinman exponential distribution
    circle_target_distribution,      # Circle target distribution
    bivariate_uniform_target_distribution   # Bivariate uniform target distribution
)




def generate_synthetic_data(config_file, selected_distributions):
    config = load_config(config_file)
    output_dir = config['output_dir']

    target_distributions = {
        "bivariate_gaussian": gaussian_target_distribution,
        "bivariate_uniform": bivariate_uniform_target_distribution,
        "gamma_exponential": gamma_exponential_target_distribution,
        "ordered_weinman": weinman_ordered_target_distribution,
        "circle_distribution": circle_target_distribution
        }

    for dist in config['distributions']:
        dist_name = dist['name']
        if selected_distributions and dist_name not in selected_distributions:
            continue #Avoiding non-selected distributions  
        
        params = dist['params']
        sizes = dist['sizes']
        correlation = dist.get('correlation', {})
        sim_settings = dist.get('simulation_settings', {})
        

        for size in sizes:
            print(f"Generazione dati: {dist_name}, size={size}")
            
            #Handling (x,y) generation from univariate distribution
            if not sim_settings:
                data = generate_univariate_data(dist_name, size, params, correlation)
                print(f"Generated data shape: {data.shape}")
                print(f"First few rows of data:\n{data}")  # Mostra le prime righe dei dati
                #Save generated data in a .csv file
                data_path = os.path.join(output_dir, "generated_data", dist_name, f"size_{size}.csv")
                save_data(data, data_path)
                continue                 
            
            #From here till the function end: handling (x,y) generation from bivariate distribution
            required_keys = {"x0", "y0", "delta", "n_cores", "burn_in", "check"}
            missing_keys = required_keys - sim_settings.keys()
            
            if missing_keys:
                raise KeyError(f"Missing required keys in 'simulation_settings': {', '.join(missing_keys)}")
            
            #  Extraction of key-related values
            x0 = sim_settings["x0"]
            y0 = sim_settings["y0"]
            delta = sim_settings["delta"]
            n_cores = sim_settings["n_cores"]
            burn_in = sim_settings["burn_in"]
            check = sim_settings["check"]
            
            target_function = target_distributions.get(dist_name)
            if not target_function:
                raise ValueError(f"No target function defined for {dist_name}.")
                
            f_target = configure_target_function(target_function, **params)
                
            data = run_parallel_metropolis_hastings(
                f=f_target,
                q_sampler=proposal_sampler,
                q_probability=proposal_probability,
                x0=x0,
                y0=y0,
                steps=size,
                delta=delta,
                n_cores=n_cores,
                burn_in=burn_in,
                check=check
            )

                
            print(f"Generated data shape: {data.shape}")
            print(f"First few rows of data:\n{data}")  # Mostra le prime righe dei dati
            #Save generated data in a .csv file
            data_path = os.path.join(output_dir, "generated_data", dist_name, f"size_{size}.csv")
            save_data(data, data_path)
            
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for selected distributions.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file (JSON).")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)
    available_distributions = [dist['name'] for dist in config['distributions']]

    # Interactive selection of distributions
    print("\nAvailable distributions:")
    for i, dist_name in enumerate(available_distributions, start=1):
        print(f"{i}. {dist_name}")
    
    print("\nChoose the distributions to generate:")
    print(" - Enter the numbers separated by spaces (e.g., 1 3 4).")
    print(" - Enter 'all' to generate all distributions.")

    while True:
        user_input = input("Your choice: ").strip()
        if user_input.lower() == 'all':
            selected_distributions = available_distributions
            break
        try:
            selected_indices = [int(i) - 1 for i in user_input.split()]
            selected_distributions = [available_distributions[i] for i in selected_indices if 0 <= i < len(available_distributions)]
            if selected_distributions:
                break
            else:
                print("Invalid selection. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Use valid numbers or 'all'.")

    # Generate data for the selected distributions
    generate_synthetic_data(args.config_file, selected_distributions)
    
    

    
    