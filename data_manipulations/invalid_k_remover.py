import os
import sys
import pandas as pd
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))
from interface_utils import navigate_directories
from io_utils import extract_file_details



def process_files():
    file_paths = navigate_directories(start_path="data/", multi_select=True, file_extension=".txt")
    
    for file_path in file_paths:
        details = extract_file_details(file_path)
        if details is None:
            logging.error(f"Non è stato possibile estrarre i dettagli da: {file_path}")
            continue
        
        try:
            # Estrae la dimensione (size) dal path e la converte in intero
            size = int(details["size"])
        except Exception as e:
            logging.error(f"Errore nella conversione della size per il file {file_path}: {e}")
            continue

        try:
            # Legge il file in un DataFrame. Si assume che il file sia formattato con separatori di spazi
            df = pd.read_csv(file_path, sep='\s+')
        except Exception as e:
            logging.error(f"Errore nella lettura del file {file_path}: {e}")
            continue
        
        # Filtra le righe dove il valore di k è minore della size (elimina quelle con k >= size)
        df_filtered = df[df["k"] < size]
        
        # Sovrascrive il file originale con i dati filtrati
        df_filtered.to_csv(file_path, index=False, sep=" ")
        print(f"File processato e sovrascritto: {file_path}")

if __name__ == "__main__":
    process_files()