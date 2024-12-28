import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sampling/')))
from multivariate_generator import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))
# Utility functions for file management and configuration loading
from io_utils import *
from config_utils import *




def generate_synthetic_data(config_file, selected_distributions):
    config = load_config(config_file)
    output_dir = config['output_dir']



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
    
    

    
    