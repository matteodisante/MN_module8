import os
import sys
import time
import numpy as np
import pandas as pd
import logging
from scipy.special import digamma

# Import dei moduli di utilità
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/')))
from interface_utils import navigate_directories, setup_logger
from io_utils import extract_file_details

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/')))
from mutual_information_1 import mutual_information_1
from mutual_information_1_entropies_sum import mutual_information_1_entropies_sum
from mutual_information_binning import mutual_information_binning_adaptive 

def measure_execution_time(func, *args, n_runs=10, warm_up_runs=3):
    """
    Misura il tempo di esecuzione di una funzione.
    
    Restituisce:
        (tempo medio, errore standard del tempo medio)
    """
    for _ in range(warm_up_runs):
        func(*args)
    
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        func(*args)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
    mean_time = np.mean(times)
    std_dev_mean_time = np.std(times, ddof=1) / np.sqrt(n_runs)
    
    return mean_time, std_dev_mean_time

def load_or_init_dataframe(filepath, index_values):
    """
    Se il file esiste, lo carica. Altrimenti, crea un DataFrame vuoto
    con gli indici specificati.
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, sep="\t", index_col=0)
        df.index = df.index.astype(int)
    else:
        df = pd.DataFrame(index=index_values)
    return df

def update_results(algorithm_name, param_list, func, files_detail, all_sizes_int, output_folder):
    """
    Aggiorna (o crea) i DataFrame dei tempi di esecuzione e delle incertezze per
    l'algoritmo indicato, calcolando solo i nuovi parametri non presenti nei file esistenti.
    
    Parameters:
        algorithm_name (str): "mi1", "misum" o "mi_binned"
        param_list (list): Lista dei valori di k (per mi1 e misum) oppure dei bins (per mi_binned)
        func (callable): Funzione da misurare.
        files_detail (list): Lista dei dettagli dei file.
        all_sizes_int (list): Lista degli size dei dataset.
        output_folder (str): Cartella in cui salvare i risultati.
    """
    # Determina i nomi dei file in base all'algoritmo
    time_filename = {
        "mi1": "mi1.txt",
        "misum": "misum.txt",
        "mi_binned": "mi_binned.txt"
    }[algorithm_name]
    
    uncert_filename = {
        "mi1": "d_mi1.txt",
        "misum": "d_misum.txt",
        "mi_binned": "d_mibinned.txt"
    }[algorithm_name]
    
    time_filepath = os.path.join(output_folder, time_filename)
    uncert_filepath = os.path.join(output_folder, uncert_filename)
    
    # Carica o inizializza i DataFrame
    df_time = load_or_init_dataframe(time_filepath, all_sizes_int)
    df_uncert = load_or_init_dataframe(uncert_filepath, all_sizes_int)
    
    # Se il DataFrame non è vuoto, converto le colonne in int
    if not df_time.empty:
        df_time.columns = list(map(int, df_time.columns))
    if not df_uncert.empty:
        df_uncert.columns = list(map(int, df_uncert.columns))
        
    # Trova i parametri (k o bins) già calcolati e quelli nuovi
    new_params = [p for p in param_list if p not in df_time.columns]
    if new_params:
        logging.info(f"Per l'algoritmo {algorithm_name}, verranno calcolati i parametri: {new_params}")
    else:
        logging.info(f"Tutti i parametri per l'algoritmo {algorithm_name} sono già presenti.")
    
    # Per ciascun dataset, calcola i tempi per i nuovi parametri
    for size in all_sizes_int:
        # Trova il percorso del dataset (assumiamo file_index "01")
        dataset_path = next(d['file_path'] for d in files_detail if d['size'] == size and d['file_index'] == "01")
        print(f"Elaborazione dataset: {dataset_path}")
        dataset = np.loadtxt(dataset_path)
        dataset_len = dataset.shape[0]
        
        for p in new_params:
            # Per mi1 e misum, se k >= lunghezza del dataset si assegna NaN
            if algorithm_name in ["mi1", "misum"] and p >= dataset_len:
                mean_time, std_time = np.nan, np.nan
            else:
                mean_time, std_time = measure_execution_time(func, dataset, p)
            df_time.loc[size, p] = mean_time
            df_uncert.loc[size, p] = std_time
    
    # Riordina le colonne in modo crescente
    df_time = df_time.reindex(sorted(df_time.columns), axis=1)
    df_uncert = df_uncert.reindex(sorted(df_uncert.columns), axis=1)
    
    # Salva i DataFrame aggiornati
    df_time.to_csv(time_filepath, sep="\t")
    df_uncert.to_csv(uncert_filepath, sep="\t")
    logging.info(f"Risultati per {algorithm_name} salvati in {output_folder}")

if __name__ == "__main__":
    setup_logger()
    logging.getLogger('core.mutual_information_1').setLevel(logging.WARNING)
    logging.getLogger('utils.core_utils').setLevel(logging.WARNING)

    # Selezione dei file da analizzare
    logging.info("Seleziona il file da analizzare")
    files_path = navigate_directories(start_path="../data/synthetic_data/", multi_select=True, file_extension=".txt")
    if not files_path:
        logging.warning("Nessun file selezionato. Uscita.")
        sys.exit(0)
    
    files_path_detail = []
    for single_path in files_path:
        details = extract_file_details(single_path)
        details["size"] = int(details["size"])
        files_path_detail.append(details)
        
    # Lista degli size univoci
    all_sizes_int = sorted({d["size"] for d in files_path_detail})
    # Informazioni di base dal primo file (si assume stessa distribuzione)
    dist_name = files_path_detail[0]["distribution_name"]
    params = files_path_detail[0]["params"]
    
    # Cartella di output: computation_times/<dist_name>/<params>/
    output_folder = os.path.join("computation_times", dist_name, params)
    os.makedirs(output_folder, exist_ok=True)
    
    # Selezione degli algoritmi da eseguire
    print("Seleziona gli algoritmi da eseguire:")
    print("1: mutual_information_1")
    print("2: mutual_information_1_entropies_sum")
    print("3: mutual_information_binning_adaptive")
    print("Esempio: per eseguire 1 e 3, inserisci '1,3'")
    selected_algorithms_input = input("Inserisci le scelte separate da virgola (lascia vuoto per eseguire tutti): ").strip()
    if selected_algorithms_input == "":
        selected_algorithms = ["1", "2", "3"]
    else:
        # Prende in considerazione solo le scelte valide
        selected_algorithms = [s.strip() for s in selected_algorithms_input.split(",") if s.strip() in ["1", "2", "3"]]
    
    # Inserimento dei valori di k
    k_input = input("Inserisci i valori di k separati da virgola (default: 1,5,25,100,500,2500,5000): ").strip()
    if k_input == "":
        k_list = [1, 5, 25, 100, 500, 2500, 5000]
    else:
        try:
            k_list = [int(val.strip()) for val in k_input.split(",")]
        except Exception:
            logging.error("Valori di k non validi. Utilizzo i default.")
            k_list = [1, 5, 25, 100, 500, 2500, 5000]
    
    # Inserimento dei valori di bins
    bins_input = input("Inserisci i valori di bins separati da virgola (default: 2,4,8,16,32,64,128,256): ").strip()
    if bins_input == "":
        bins_list = [2, 4, 8, 16, 32, 64, 128, 256]
    else:
        try:
            bins_list = [int(val.strip()) for val in bins_input.split(",")]
        except Exception:
            logging.error("Valori di bins non validi. Utilizzo i default.")
            bins_list = [2, 4, 8, 16, 32, 64, 128, 256]
    
    # Esecuzione degli algoritmi scelti
    if "1" in selected_algorithms:
        update_results("mi1", k_list, mutual_information_1, files_path_detail, all_sizes_int, output_folder)
    if "2" in selected_algorithms:
        update_results("misum", k_list, mutual_information_1_entropies_sum, files_path_detail, all_sizes_int, output_folder)
    if "3" in selected_algorithms:
        update_results("mi_binned", bins_list, mutual_information_binning_adaptive, files_path_detail, all_sizes_int, output_folder)
    
    logging.info("Elaborazione completata.")