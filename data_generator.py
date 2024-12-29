import os
import sys
import argparse
import pandas as pd


# Import helper modules from utils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from math_utils import *
from config_utils import load_config
from io_utils import * # in realtà non è stato tale script!

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sampling')))
from multivariate_generator import *


def generate_and_save_data(config, selected_distribution_name, selected_size=None, all_sizes=False, num_files=20):
    output_dir = config.get('output_dir', 'synthetic_data')

    # Trovare la distribuzione selezionata nel config
    distribution = next((d for d in config['distributions'] if d['name'] == selected_distribution_name), None)
    if not distribution:
        raise ValueError(f"Distribution {selected_distribution_name} not found in config.")

    # Parametri generali
    params = distribution['params']

    # Se all_sizes è True, genera i dati per tutte le dimensioni disponibili
    sizes_to_generate = distribution['sizes'] if all_sizes else [selected_size]

    for size in sizes_to_generate:
        # Creazione della directory principale, se non esiste
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Creazione della sottocartella per la distribuzione scelta
        dist_dir = os.path.join(output_dir, selected_distribution_name)
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)

        # Creazione della sottocartella specifica per la dimensione
        size_dir = os.path.join(dist_dir, f"{selected_distribution_name}_size_{size}")
        if not os.path.exists(size_dir):
            os.makedirs(size_dir)

        # Preparazione dei percorsi dei file
        file_paths = [
            os.path.join(size_dir, f"{selected_distribution_name}_size_{size}_{file_num:02d}.txt")
            for file_num in range(1, num_files + 1)
        ]
        
        # Controllo se almeno un file esiste
        existing_files = [file_path for file_path in file_paths if os.path.exists(file_path)]
        if existing_files:
            print(f"\n{len(existing_files)} files already exist for this configuration (size {size}).")
            overwrite = None
            while overwrite not in ['y', 'n']:
                overwrite = input(f"Do you want to overwrite these files for size {size}? (y/n): ").strip().lower()
            if overwrite == 'n':
                print(f"Skipping generation for size {size}.")
                continue

        # Dizionario per mappare le distribuzioni alle rispettive funzioni
        distribution_functions = {
            "independent_gaussian_rv": lambda params, size: independent_gaussian_rv(params['mu'], params['sigma'], size),
            "correlated_gaussian_rv": lambda params, size: correlated_gaussian_rv(params['mu'], params['sigma'], distribution['correlation'], size),
            "independent_uniform_rv": lambda params, size: independent_uniform_rv(params['low'], params['high'], size),
            "independent_exponential_rv": lambda params, size: independent_exponential_rv(params['lambda'], size),
            "gamma_exponential": lambda params, size: gamma_exponential(params['theta'], size),
            "ordered_wienman_exponential": lambda params, size: ordered_wienman_exponential(params['theta'], size),
            "circular": lambda params, size: circular(params['a'], params['b'], params['c'], size),
        }

        # Recupera la funzione di generazione
        generate_function = distribution_functions.get(distribution['name'])
        if not generate_function:
            raise ValueError(f"Unsupported distribution: {distribution['name']}")

        # Generazione dei dati
        for file_num, output_file in enumerate(file_paths, start=1):
            data = generate_function(params, size)

            # Salvataggio dei dati in un file TXT senza header e con alta precisione
            np.savetxt(output_file, data, fmt="%.15f")
            print(f"File {file_num}/{num_files} generated and saved in: {output_file}")





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate synthetic data for a selected distribution.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file (JSON).")
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config_file)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

    available_distributions = [dist['name'] for dist in config['distributions']]

    while True:
        # Interactive selection of a single distribution
        print("\nAvailable distributions:")
        for i, dist_name in enumerate(available_distributions, start=1):
            print(f"{i}. {dist_name}")

        selected_distribution = None
        while selected_distribution is None:
            try:
                user_input = input("Select a distribution by number: ").strip()
                selected_index = int(user_input) - 1
                if 0 <= selected_index < len(config['distributions']):
                    selected_distribution = config['distributions'][selected_index]
                else:
                    print("Invalid selection. Please choose a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        print(f"\nSelected distribution: {selected_distribution['name']}")

        # Ask if the user wants to generate data for all sizes or a specific size
        print(f"\nAvailable sizes for {selected_distribution['name']}:")
        for i, size in enumerate(selected_distribution['sizes'], start=1):
            print(f"{i}. {size}")

        generate_all = None
        while generate_all not in ['y', 'n']:
            generate_all = input("Do you want to generate data for all sizes? (y/n): ").strip().lower()

        if generate_all == 'y':
            # Ask for the number of files to generate
            while True:
                try:
                    user_input = input("Enter the number of files to generate for each size (leave blank for default 20): ").strip()
                    if not user_input:  # Default value if input is empty
                        num_files = 20
                        print("Using default value of 20 files.")
                        break
                    num_files = int(user_input)
                    if num_files > 0:  # Ensure the value is positive
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

            # Generate data for all sizes
            generate_and_save_data(
                config=config,
                selected_distribution_name=selected_distribution['name'],
                all_sizes=True,
                num_files=num_files
            )
        else:
            # Generate data for a specific size
            selected_size = None
            while selected_size is None:
                try:
                    user_input = input("Select a size by number: ").strip()
                    selected_index = int(user_input) - 1
                    if 0 <= selected_index < len(selected_distribution['sizes']):
                        selected_size = selected_distribution['sizes'][selected_index]
                    else:
                        print("Invalid selection. Please choose a valid number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Ask for the number of files to generate
            while True:
                try:
                    user_input = input("Enter the number of files to generate (leave blank for default 20): ").strip()
                    if not user_input:  # Default value if input is empty
                        num_files = 20
                        print("Using default value of 20 files.")
                        break
                    num_files = int(user_input)
                    if num_files > 0:  # Ensure the value is positive
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

            generate_and_save_data(
                config=config,
                selected_distribution_name=selected_distribution['name'],
                selected_size=selected_size,
                num_files=num_files
            )

        # Ask if the user wants to generate more data or exit
        continue_choice = None
        while continue_choice not in ['y', 'n']:
            continue_choice = input("\nDo you want to generate more data? (y/n): ").strip().lower()
        if continue_choice == 'n':
            print("Exiting program. Goodbye!")
            break

