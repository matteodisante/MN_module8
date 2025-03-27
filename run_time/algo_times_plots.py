import os
import sys
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

# Aggiungi il percorso per i moduli utility
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/')))
from interface_utils import navigate_directories, setup_logger
from io_utils import extract_file_details, ensure_directory

def calcola_pendenza(x, y, num_punti):
    """Calcola la pendenza del fit lineare sui dati forniti."""
    if len(x) >= num_punti:
        log_x = np.log(x[-num_punti:])
        log_y = np.log(y[-num_punti:])
        pendenza, _ = np.polyfit(log_x, log_y, 1)
        return pendenza
    return np.nan

if __name__ == '__main__':
    setup_logger(stdout_printing=True)

    logging.info("Seleziona il file contenente i valori centrali dei tempi di calcolo per una data stima di mi\n")
    central_value_file_path = navigate_directories(
        start_path="../run_time/computation_times/",
        multi_select=False,
        file_extension=".txt"
    )[0]

    logging.info("Seleziona il file contenente le incertezze per la stessa stima di mi\n")
    error_value_file_path = navigate_directories(
        start_path="../run_time/computation_times/",
        multi_select=False,
        file_extension=".txt"
    )[0]

    # Mappa dei nomi dei file ai tipi
    file_name_map = {
        "mi1.txt": "mi1",
        "d_mi1.txt": "mi1",
        "misum.txt": "misum",
        "d_misum.txt": "misum",
        "mi_binned.txt": "mi_binned",
        "d_mibinned.txt": "mi_binned",
    }
    
    central_file_name = os.path.basename(central_value_file_path)
    error_file_name = os.path.basename(error_value_file_path)

    # Determina il tipo di file (mi1, misum, mi_binned)
    mi_type = file_name_map.get(central_file_name, "sconosciuto")

    if mi_type == "sconosciuto":
        logging.error("Tipo di file non riconosciuto. Uscita.")
        sys.exit(1)

    # Estrai dettagli dal percorso del file (nome distribuzione, parametri, dimensione, indice file)
    details = extract_file_details(central_value_file_path)

    # Leggi i file con i tempi e le incertezze
    run_times_df = pd.read_csv(central_value_file_path, sep='\t', header=0)
    run_errors_df = pd.read_csv(error_value_file_path, sep='\t', header=0)

    # Estrai i valori di N dalla prima colonna
    N_values = run_times_df.iloc[:, 0].values
    # Estrai i valori di k dagli header (escludendo la prima colonna)
    k_headers = run_times_df.columns[1:].astype(float)

    # Estrai le matrici dei tempi e delle incertezze (escludendo la prima colonna)
    times_matrix = run_times_df.iloc[:, 1:].values
    errors_matrix = run_errors_df.iloc[:, 1:].values

    # Imposta la directory per salvare i plot
    plots_dir = os.path.join("..", "plots", "time_estimates")
    if not ensure_directory(plots_dir):
        sys.exit("Operazione annullata dall'utente.")

    ###############################################################
    # Primo plot: Curve per N fisso (x = k/N)
    ###############################################################
    plt.figure(figsize=(9, 6))
    plt.xscale('log')
    plt.yscale('log')

    for i, N in enumerate(N_values):
        row_times = times_matrix[i, :]
        row_errors = errors_matrix[i, :]

        valid_mask = row_times > 0
        if not np.any(valid_mask):
            continue

        x_vals = k_headers[valid_mask] / N
        y_vals = row_times[valid_mask]
        y_errs = row_errors[valid_mask]

        num_punti_fit = min(2, len(x_vals))
        pendenza = calcola_pendenza(x_vals, y_vals, num_punti_fit)

        plt.errorbar(
            x_vals, y_vals, yerr=y_errs,
            fmt='.-', capsize=0, label=f'N={int(N)} (slope ultimi {num_punti_fit} pts={pendenza:.2f})'
        )

    plt.xlabel('k/N')
    plt.ylabel('CPU time')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Salva il primo plot con un nome file che include il tipo di MI
    plot_filename_N_fixed = f"{mi_type}_{details['distribution_name']}_{details['params']}_{details['size']}_{details['file_index']}_N_fixed.pdf"
    plt.savefig(os.path.join(plots_dir, plot_filename_N_fixed), bbox_inches='tight')
    plt.show()

    ###############################################################
    # Secondo plot: Curve per k fisso (x = k/N, con N variabile)
    ###############################################################
    plt.figure(figsize=(9, 6))
    plt.xscale('log')
    plt.yscale('log')

    for j, k in enumerate(k_headers):
        col_times = times_matrix[:, j]
        col_errors = errors_matrix[:, j]

        x_vals = k / N_values
        valid_mask = col_times > 0
        if not np.any(valid_mask):
            continue

        x_vals_valid = x_vals[valid_mask]
        y_vals_valid = col_times[valid_mask]
        y_errs_valid = col_errors[valid_mask]

        num_punti_fit = 3 if k >= 100 else 2
        pendenza = calcola_pendenza(x_vals_valid, y_vals_valid, num_punti_fit)

        plt.errorbar(
            x_vals_valid, y_vals_valid, yerr=y_errs_valid,
            fmt='.-', capsize=0, label=f'k={int(k)} (slope primi {num_punti_fit} pts={pendenza:.2f})'
        )

    plt.xlabel('k/N')
    plt.ylabel('CPU time')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Salva il secondo plot con un nome file che include il tipo di MI
    plot_filename_k_fixed = f"{mi_type}_{details['distribution_name']}_{details['params']}_{details['size']}_{details['file_index']}_k_fixed.pdf"
    plt.savefig(os.path.join(plots_dir, plot_filename_k_fixed), bbox_inches='tight')
    plt.show()
    