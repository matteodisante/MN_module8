import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import scipy.interpolate as spi
import inspect

from utils.config_utils import load_config
from utils.math_utils import circular_mi_theoretical, gamma_exponential_mi_theoretical, \
    correlated_gaussian_rv_mi_theoretical, independent_exponential_rv_mi_theoretical, \
        independent_gaussian_rv_mi_theoretical, independent_uniform_rv_mi_theoretical, \
            oredered_wienman_exponential_mi_theoretical



def load_config_distributions():
    """Carica il file config.json dalla stessa directory dello script e restituisce le distribuzioni disponibili."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    if not os.path.exists(config_path):
        print(f"Errore: Il file config.json non è stato trovato in {script_dir}.")
        exit(1)

    config = load_config(config_path)
    return [dist["name"] for dist in config["distributions"]]

def get_user_choice(options, prompt):
    """Chiede all'utente di scegliere un'opzione valida."""
    while True:
        print(prompt)
        for idx, option in enumerate(options, 1):
            print(f"{idx}. {option}")
        choice = input("Inserisci il numero corrispondente: ")
        
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(options):
                return options[choice - 1]
        print("Scelta non valida. Riprova.")

def get_valid_directory():
    """Chiede all'utente di inserire un percorso valido alla cartella."""
    while True:
        directory_path = input("Inserisci il percorso della cartella da analizzare: ")
        if os.path.isdir(directory_path):
            return directory_path
        else:
            print("Errore: Il percorso inserito non è una cartella valida. Riprova.")



def select_files_by_extension(start_path=".", file_extension=".txt"):
    """Selects all files with a specified extension from the given directory and its subdirectories."""
    selected_paths = []

    # Walk through all directories and files starting from start_path
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.endswith(file_extension):
                selected_paths.append(os.path.join(root, file))
    return selected_paths

def find_matching_files(directory, distribution_name, mi_estimate):
    """Cerca file .txt nella cartella specificata e seleziona quelli corrispondenti alla 
    distribuzione e stima in input."""
    all_files = select_files_by_extension(directory, file_extension=".txt")
    pattern = re.compile(rf"summary_{distribution_name}_(.*?)_size_\d+_{mi_estimate}.txt")
    matching_files = [file for file in all_files if pattern.match(os.path.basename(file))]
    return matching_files


# Estrae la stringa {params_values} dal nome del file
def extract_parameters_from_filename(filename, distribution_name, mi_estimate):
    """
    Estrae la stringa dei parametri dal nome del file, dato il formato:
    'summary_{distribution_name}_{params_values}_size_{value_size}_{mi_estimate}.txt'
    """
    pattern = rf"summary_{distribution_name}_(.*?)_size_\d+_{mi_estimate}.txt"
    match = re.match(pattern, filename)
    return match.group(1) if match else None


def extract_parameters_from_paths(paths, distribution_name, mi_estimate):
    """
    Estrae i parametri e i loro valori da tutti i file nella lista paths e
    crea un dizionario che associa ogni file ai suoi parametri estratti.
    """
    # Sono due dizionari, uno per parametro-valore, uno per file-(parametro-valore)
    all_parameters = {}
    extracted_params_per_file = {}

    for path in paths:
        param_string = extract_parameters_from_filename(os.path.basename(path), distribution_name, mi_estimate)
        if param_string:
            param_pairs = param_string.split('_')

            if len(param_pairs) % 2 != 0:
                print(f"Errore nel file: {path}. Il numero di parametro-valore è dispari.")
                continue

            file_params = {param_pairs[i]: param_pairs[i + 1] for i in range(0, len(param_pairs), 2)}
            extracted_params_per_file[path] = file_params  # Salva i parametri del file

            for param_name, param_value in file_params.items():
                if param_name not in all_parameters:
                    all_parameters[param_name] = set()
                all_parameters[param_name].add(param_value)

    return all_parameters, extracted_params_per_file


def filter_files_by_parameters(extracted_params_per_file, selected_params):
    """
    Filtra i file in base ai parametri selezionati dall'utente.

    :param extracted_params_per_file: Dizionario {file_path: {param_name: param_value}}.
    :param selected_params: Dizionario {param_name: param_value} scelti dall'utente.
    :return: Lista di file filtrati.
    """
    return [
        path for path, file_params in extracted_params_per_file.items()
        if all(file_params.get(param) == value for param, value in selected_params.items())
    ]


# Estrae i valori dei k possibili dai files
def extract_k_values_from_files(files):
    """Estrae i valori di k dalla prima colonna dei file. La prima riga ha l'intestazione, 
    e i valori di k sono nella prima colonna dalla seconda riga in poi."""
    k_values = set()
    
    for file in files:
        with open(file, 'r') as f:
            # Saltare la prima riga (intestazione)
            next(f)
            for line in f:
                # Split della riga per separare le colonne
                columns = line.strip().split()
                if columns:  # Verifica che la riga non sia vuota
                    try:
                        k_value = int(columns[0])  # Prima colonna contiene il valore di k
                        k_values.add(k_value)
                    except ValueError:
                        # Se il valore nella prima colonna non è un intero, ignora quella riga
                        continue
    
    # Restituisci i valori di k ordinati
    return sorted(k_values)





def plot_figure_4(files, distribution_name, mi_estimate, theoretical_mi):
    """
    Genera il plot per la Figura 4 (un solo file per ogni N).
    L'asse y rappresenta la mutua informazione stimata.
    """
    plt.figure(figsize=(8, 6))

    for file in sorted(files):
        match = re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_({mi_estimate}).txt", os.path.basename(file))
        if match:
            N = int(match.group(2))

            # Caricamento dati
            k_vals, means, sigmas = np.loadtxt(file, skiprows=1, unpack=True)
            k_over_N = k_vals / N  # Normalizzazione di k rispetto a N

            # Ordinamento per garantire la connessione dei punti
            sorted_indices = np.argsort(k_over_N)
            k_over_N_sorted = k_over_N[sorted_indices]
            means_sorted = means[sorted_indices]
            sigmas_sorted = sigmas[sorted_indices]

            # Tracciare una linea diretta tra i punti
            plt.plot(k_over_N_sorted, means_sorted, linestyle='--', label=f'N={N}')

            # Punti con barre di errore
            plt.errorbar(k_over_N_sorted, means_sorted, yerr=sigmas_sorted, fmt='.')

    # Aggiunta della mutua informazione teorica
    plt.axhline(y=theoretical_mi, color='r', linestyle='-', linewidth=2, label=f'I theoretical = {theoretical_mi:.4f}')
    
    # Personalizzazione del grafico
    plt.xlabel("k/N", fontsize=14)
    plt.ylabel(f"I {mi_estimate}", fontsize=14)
    plt.legend(title="Sample Size (N)", fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def process_figure_4(files, distribution_name, mi_estimate):
    """
    Gestisce la selezione dei parametri per la Figura 4, filtra i file corrispondenti e fa il plot.
    """
    # Ottiene il dizionario di parametri-valori e quello per l'associazione con i paths
    all_parameters, extracted_params_per_file = extract_parameters_from_paths(files, distribution_name, mi_estimate)
    # Definisce il dizionario di parametri selezionati e chiede all'utente di selezionare i valori dei 
    # parametri per il plot
    selected_params = {}

    for param_name, param_values in all_parameters.items():
        chosen_value = get_user_choice(list(param_values), f"Scegli il valore per il parametro {param_name}:")
        selected_params[param_name] = chosen_value

    # Filtra i files con i valori dei parametri selezionati dall'utente
    filtered_files = filter_files_by_parameters(extracted_params_per_file, selected_params)

    print("File selezionati dopo il filtro:")
    for file in filtered_files:
        print(file)

    # Calcolo della mutua informazione teorica
    theoretical_mi_function = globals().get(f"{distribution_name}_mi_theoretical")
    if theoretical_mi_function:
        required_params = inspect.signature(theoretical_mi_function).parameters
        filtered_params = {key: float(value) for key, value in selected_params.items() if key in required_params}
        theoretical_mi = theoretical_mi_function(**filtered_params)
        print(f"Mutua informazione teorica per {distribution_name}: {theoretical_mi}")
    else:
        print(f"Nessuna funzione teorica trovata per {distribution_name}.")
        theoretical_mi = None

    # Fa il plot
    plot_figure_4(filtered_files, distribution_name, mi_estimate, theoretical_mi)


def process_figure_7_9(files, distribution_name, mi_estimate, figure):
    """
    Gestisce la selezione dei parametri e del k per la Figura 7 e 9, filtra i file corrispondenti e fa il plot.
    """
    # Estrai i k disponibili nei file
    k_values = extract_k_values_from_files(files)
    if not k_values:
        print("Nessun valore di k trovato nei file. Uscita.")
        exit()
    
    # L'utente sceglie il valore di k
    k_choice = get_user_choice(k_values, "Scegli il valore di k da utilizzare per il plot:")
    print(f"Hai scelto il valore di k: {k_choice}")

    # Estrai i parametri dai file
    all_parameters, extracted_params_per_file = extract_parameters_from_paths(files, distribution_name, mi_estimate)
    selected_param = get_user_choice(list(all_parameters.keys()), "Scegli il parametro da analizzare:")

    # Raggruppa i file per il valore del parametro selezionato
    grouped_files = {}
    for param_value in all_parameters[selected_param]:
        selected_files = filter_files_by_parameters(extracted_params_per_file, {selected_param: param_value})
        if selected_files:
            grouped_files[param_value] = selected_files

    plt.figure(figsize=(8, 6))

    # Ciclo sui valori del parametro
    for param_value, file_list in grouped_files.items():
        x_vals, y_vals, y_errs = [], [], []

        # Calcolo della mutua informazione teorica
        theoretical_mi_function = globals().get(f"{distribution_name}_mi_theoretical")
        theoretical_mi = None

        if theoretical_mi_function:
            required_params = inspect.signature(theoretical_mi_function).parameters

            # Creazione del dizionario dei parametri necessari
            filtered_params = {
            key: (float(param_value) if key == selected_param else float(next(iter(value))))
            for key, value in all_parameters.items() if key in required_params
            }

            # Calcolo della mutua informazione teorica
            theoretical_mi = theoretical_mi_function(**filtered_params)
            print(f"Mutua informazione teorica per {distribution_name}, {selected_param}={param_value}: {theoretical_mi}")

        if figure == "7":
            # Se la mutua informazione teorica è nulla, escludi dalla legenda
            if theoretical_mi is None or theoretical_mi == 0:
                print(f"Attenzione: mutua informazione teorica nulla per {selected_param}={param_value}. Punto escluso dalla legenda.")
                continue  # Salta il resto del ciclo per questo valore di parametro

            for file in file_list:
                with open(file, 'r') as f:
                    next(f)  # Salta l'intestazione
                    for line in f:
                        columns = line.strip().split()
                        if columns and int(columns[0]) == k_choice:
                            match = re.search(r"size_(\d+)", file)
                            if match:
                                N = int(match.group(1))
                                # Se la mutua informazione teorica è valida, aggiungi i punti
                                x_vals.append(k_choice / N)
                                y_vals.append(float(columns[1]) / theoretical_mi)
                                y_errs.append(float(columns[2]) / theoretical_mi)
                            else:
                                print(f"Attenzione: impossibile estrarre N dal nome file {file}")
        else:
            if figure == "9":
                 for file in file_list:
                    with open(file, 'r') as f:
                        next(f)  # Salta l'intestazione
                        for line in f:
                            columns = line.strip().split()
                            if columns and int(columns[0]) == k_choice:
                                match = re.search(r"size_(\d+)", file)
                                if match:
                                    N = int(match.group(1))
                                    # Se la mutua informazione teorica è valida, aggiungi i punti
                                    x_vals.append(k_choice / N)
                                    # Sottrarre la mutua informazione teorica dalla media (y_vals)
                                    y_vals.append(float(columns[1]) - theoretical_mi)
                                    # Mantieni invariato l'errore (y_errs)
                                    y_errs.append(float(columns[2]))
                                else:
                                    print(f"Attenzione: impossibile estrarre N dal nome file {file}")

        # Ordina i dati per x
        sorted_indices = np.argsort(x_vals)
        x_vals = np.array(x_vals)[sorted_indices]
        y_vals = np.array(y_vals)[sorted_indices]
        y_errs = np.array(y_errs)[sorted_indices]

        # Collegare i punti con una retta
        plt.plot(x_vals, y_vals, linestyle='--', label=f"{selected_param}={param_value}")

        # Grafico con barre di errore
        plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='.')

    plt.xlabel(f"{k_choice}/N", fontsize=14)
    if figure == "7":
        plt.ylabel(f"I {mi_estimate} / I theoretical", fontsize=14)
    else:
        if figure == "9":
            plt.ylabel(f"I {mi_estimate} - I theoretical", fontsize=14)
    plt.legend(title=selected_param, fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def process_figure_8(files, distribution_name, mi_estimate):
    """
    Gestisce la selezione dei parametri per la Figura 8, filtra i file corrispondenti e fa il plot.
    """
    # Estrai i k disponibili nei file
    k_values = extract_k_values_from_files(files)
    if not k_values:
        print("Nessun valore di k trovato nei file. Uscita.")
        exit()
    
    # L'utente sceglie il valore di k
    k_choice = get_user_choice(k_values, "Scegli il valore di k da utilizzare per il plot:")
    print(f"Hai scelto il valore di k: {k_choice}")

    # Ottiene il dizionario di parametri-valori e quello per l'associazione con i paths
    all_parameters, extracted_params_per_file = extract_parameters_from_paths(files, distribution_name, mi_estimate)
    
    # Definisce il dizionario di parametri selezionati e chiede all'utente di selezionare i valori dei 
    # parametri per il plot
    selected_params = {}

    for param_name, param_values in all_parameters.items():
        chosen_value = get_user_choice(list(param_values), f"Scegli il valore per il parametro {param_name}:")
        selected_params[param_name] = chosen_value

    # Filtra i files con i valori dei parametri selezionati dall'utente
    filtered_files = filter_files_by_parameters(extracted_params_per_file, selected_params)

    print("File selezionati dopo il filtro:")
    for file in filtered_files:
        print(file)

    # Lista per raccogliere i valori di N e le deviazioni standard corrispondenti al k scelto
    N_values = []
    sigma_values = []

    # Ciclo sui file filtrati per estrarre la deviazione standard corrispondente a k_choice
    for file in filtered_files:
        match = re.search(rf"summary_{distribution_name}_(.*?)_size_(\d+)_({mi_estimate}).txt", os.path.basename(file))
        if match:
            N = int(match.group(2))  # Estrae N dal nome del file

            # Caricamento dei dati
            k_vals, means, sigmas = np.loadtxt(file, skiprows=1, unpack=True)

            # Cerca il valore di k scelto nei dati e ottieni la corrispondente deviazione standard
            if k_choice in k_vals:
                k_index = np.where(k_vals == k_choice)[0][0]  # Trova l'indice del k scelto
                sigma_values.append(sigmas[k_index])  # Aggiungi la deviazione standard
                N_values.append(N)  # Aggiungi N

    # Verifica se sono stati trovati valori di sigma per il k scelto
    if not sigma_values:
        print(f"Nessun valore di deviazione standard trovato per k = {k_choice}.")
        return

    # Ordinamento dei valori in funzione di N
    sorted_indices = np.argsort(N_values)
    N_values_sorted = np.array(N_values)[sorted_indices]
    sigma_values_sorted = np.array(sigma_values)[sorted_indices]

    # Plot della deviazione standard in funzione di N
    plt.figure(figsize=(8, 6))
    plt.plot(N_values_sorted, sigma_values_sorted, linestyle='--', marker='.', color='b', label=f'Sigma per k={k_choice}')
    plt.xlabel('N', fontsize=14)
    plt.ylabel('Standard Deviation', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



def main():
    # Cerca le distribuzioni disponibili, mi_estimators sono gli stimatori disponibili
    available_distributions = load_config_distributions()
    mi_estimators = ["mi_1", "mi_sum", "mi_binning"]
    
    # Scelta della figura
    figure_choice = get_user_choice(["4", "7", "8", "9"], "Scegli la figura da riprodurre:")
    print(f"Hai scelto la figura {figure_choice}.")
    
    # Scelta della stima di mutua informazione
    mi_choice = get_user_choice(mi_estimators, "Scegli la stima di mutua informazione da utilizzare:")
    print(f"Hai scelto la stima {mi_choice}.")
    
    # Scelta della distribuzione
    distribution_choice = get_user_choice(available_distributions, "Scegli la distribuzione da usare per il plot:")
    print(f"Hai scelto la distribuzione {distribution_choice}.")
    
    # Scelta della cartella con controllo
    directory_path = get_valid_directory()
    print(f"Hai scelto la cartella: {directory_path}")
    
    # Ricerca dei file corrispondenti (distribuzione e stima scelti)
    matching_files = find_matching_files(directory_path, distribution_choice, mi_choice)

    if matching_files:
        print("File trovati:")
        for file in matching_files:
            print(file)
    else:
        print("Nessun file corrispondente trovato nella cartella selezionata.")

    # Se la figura scelta è la 4
    if figure_choice == "4":
        process_figure_4(matching_files, distribution_choice, mi_choice)

    # Se la figura scelta è la 7
    if figure_choice == "7":
        process_figure_7_9(matching_files, distribution_choice, mi_choice, figure_choice)

    # Se la figura scelta è la 8
    if figure_choice == "8":
        process_figure_8(matching_files, distribution_choice, mi_choice)

    # Se la figura scelta è la 9
    if figure_choice == "9":
        process_figure_7_9(matching_files, distribution_choice, mi_choice, figure_choice)

if __name__ == "__main__":
    main()
