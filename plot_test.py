import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './utils/')))
from interface_utils import navigate_directories
from io_utils import load_data_csv, ensure_directory

# Funzione per verificare la presenza delle colonne richieste
def check_required_columns(data, required_columns):
    for column in required_columns:
        if column not in data.columns:
            return False
    return True

# Funzione per generare la figura 2 (MI di primo ordine)
def generate_figure_2(file_paths, output_path):
    ensure_directory(os.path.dirname(output_path))
    plt.figure()
    for file in file_paths:
        data = load_data_csv(file)
        if not check_required_columns(data, ["k", "mean_mi_1", "sigma_mi_1"]):
            print(f"[WARNING] Skipping file {file} for Figure 2: required columns missing.")
            continue
        plt.errorbar(
            data["k"],
            data["mean_mi_1"],
            yerr=data["sigma_mi_1"],
            fmt="o",
            capsize=5,
            label=os.path.basename(file),
        )
    plt.title("Figure 2")
    plt.xlabel("k")
    plt.ylabel("Mean MI")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

# Funzione per generare la figura 4 (MI sommata)
def generate_figure_4(file_paths, output_path):
    ensure_directory(os.path.dirname(output_path))
    plt.figure()
    for file in file_paths:
        data = load_data_csv(file)
        if not check_required_columns(data, ["k", "mean_mi_sum", "sigma_mi_sum"]):
            print(f"[WARNING] Skipping file {file} for Figure 4: required columns missing.")
            continue
        plt.errorbar(
            data["k"],
            data["mean_mi_sum"],
            yerr=data["sigma_mi_sum"],
            fmt="s",
            capsize=5,
            label=os.path.basename(file),
        )
    plt.title("Figure 4")
    plt.xlabel("k")
    plt.ylabel("Summed MI")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

# Funzione per generare la figura 6 (MI con binning)
def generate_figure_6(file_paths, output_path):
    ensure_directory(os.path.dirname(output_path))
    plt.figure()
    for file in file_paths:
        data = load_data_csv(file)
        if not check_required_columns(data, ["bin_size", "mean_mi_binning", "sigma_mi_binning"]):
            print(f"[WARNING] Skipping file {file} for Figure 6: required columns missing.")
            continue
        plt.errorbar(
            data["bin_size"],
            data["mean_mi_binning"],
            yerr=data["sigma_mi_binning"],
            fmt="d",
            capsize=5,
            label=os.path.basename(file),
        )
    plt.title("Figure 6")
    plt.xlabel("Bin Size")
    plt.ylabel("Mean MI")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

# Funzione per chiedere all'utente quali figure e tipi di MI generare
def user_select_figures_and_types():
    mi_types = {
        "1": "mi_1",
        "2": "mi_sum",
        "3": "mi_binning"
    }
    figures = {
        "2": "Figure 2",
        "4": "Figure 4",
        "6": "Figure 6"
    }

    selected_combinations = []

    print("\nSelect MI types to generate figures for:")
    for key, desc in mi_types.items():
        print(f"  {key}. {desc}")

    selected_mi_types = input("Enter MI type numbers separated by commas (e.g., 1,2): ").strip()
    selected_mi_types = [mi_types[key.strip()] for key in selected_mi_types.split(",") if key.strip() in mi_types]

    if not selected_mi_types:
        print("No MI types selected. Exiting.")
        sys.exit(0)

    print("\nSelect figures to generate:")
    for key, desc in figures.items():
        print(f"  {key}. {desc}")

    selected_figures = input("Enter figure numbers separated by commas (e.g., 2,4): ").strip()
    selected_figures = [int(key.strip()) for key in selected_figures.split(",") if key.strip() in figures]

    if not selected_figures:
        print("No figures selected. Exiting.")
        sys.exit(0)

    for mi_type in selected_mi_types:
        for fig in selected_figures:
            selected_combinations.append((mi_type, fig))

    return selected_combinations

if __name__ == "__main__":
    output_dir = "./plots/"

    # Naviga e seleziona file .csv
    selected_files = navigate_directories(start_path='./data/', file_extension=".csv")

    if not selected_files:
        print("No files selected. Exiting.")
        sys.exit(0)

    # Chiedi all'utente quali tipi di MI e figure generare
    selected_combinations = user_select_figures_and_types()

    # Genera le figure per le combinazioni selezionate
    for mi_type, fig in selected_combinations:
        if fig == 2:
            generate_figure_2(selected_files, os.path.join(output_dir, f"figure_2_{mi_type}.png"))
        elif fig == 4:
            generate_figure_4(selected_files, os.path.join(output_dir, f"figure_4_{mi_type}.png"))
        elif fig == 6:
            generate_figure_6(selected_files, os.path.join(output_dir, f"figure_6_{mi_type}.png"))

    print("Figures generated successfully.")
