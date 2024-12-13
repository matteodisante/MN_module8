import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sampling/')))
from univariate_generator import data_generator
from multivariate_generator import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))
from io_utils import save_data
from plot_utils import plot_3d_histogram, plot_marginals
from config_utils import load_config
from math_utils import (gaussian_target_distribution, gamma_exponential_target_distribution,
                        weinman_ordered_target_distribution, circle_target_distribution,
                        bivariate_uniform_target_distribution)


def generate_data(config_file, selected_distributions):
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
            continue
        
        params = dist['params']
        sizes = dist['sizes']
        correlation = dist.get('correlation', {})
        sim_settings = dist.get('simulation_settings', {})
        

        for size in sizes:
            print(f"Generazione dati: {dist_name}, size={size}")
            
            if sim_settings is None:
                data = data_generator(dist_name, size, params, correlation)
            
            if sim_settings is not None:
                x0 = sim_settings.get("x0", 0)
                y0 = sim_settings.get("y0", 0)
                delta = sim_settings.get("delta", 1.0)
                n_cores = sim_settings.get("n_cores", 4)
                burn_in = sim_settings.get("burn_in", 10000)
                
                target_function = target_distributions.get(dist_name)
                if not target_function:
                    raise ValueError(f"No target function defined for {dist_name}.")
                    
                data = run_parallel_metropolis_hastings(
                    f=target_function,
                    q_sampler=proposal_sampler,
                    q_probability=proposal_probability,
                    x0=x0,
                    y0=y0,
                    steps=size,
                    delta=delta,
                    n_cores=n_cores,
                    burn_in=burn_in
                )
                
            else:
                raise ValueError(f"Unsupported distribution: {dist_name}") 
                


            data_path = os.path.join(output_dir, "generated_data", dist_name, f"size_{size}.csv")
            print(f"Generated data shape: {data.shape}")
            print(f"First few rows of data:\n{data[:5]}")  # Mostra le prime righe dei dati
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
    generate_data(args.config_file, selected_distributions)