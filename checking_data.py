import sys
import os
import argparse
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))

from plot_utils import plot_histograms, plot_density
from math_utils import log_likelihood, calculate_statistics
from io_utils import load_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analizza due serie temporali da un file di testo.")
    parser.add_argument("file_path", type=str, help="Percorso del file di testo contenente le serie temporali.")
    args = parser.parse_args()

    data = load_data(args.file_path)
    series1, series2 = data[:, 0], data[:, 1]
    bins = int(np.sqrt(len(series1)))
    
    # Calcolo delle statistiche
    stats1 = calculate_statistics(series1)
    stats2 = calculate_statistics(series2)
    correlation = pearsonr(series1, series2)[0]
    
    # Stampa delle statistiche
    print(f"Statistiche Serie 1: {stats1}")
    print(f"Statistiche Serie 2: {stats2}")
    print(f"Correlazione tra le serie: {correlation:.7f}")
    

    # Disegno degli istogrammi
    plot_histograms(series1, series2, bins)
    
    # Funzione teorica per test (normale standard in questo esempio)
    theoretical_pdf = norm.pdf
    plot_density(series1, series2, bins, theoretical_pdf)
    
    # Calcolo della log-likelihood
    ll1 = log_likelihood(series1, theoretical_pdf)
    ll2 = log_likelihood(series2, theoretical_pdf)
    print(f"Log-Likelihood Serie 1: {ll1}")
    print(f"Log-Likelihood Serie 2: {ll2}")
    
    
    mi_val_empirical = mutual_information_1(data, k = 15)
    mi_th = -0.5*np.log(1-correlation**2)
    print(f"Th. val = {mi_th}, Emp. val = {mi_val_empirical}")

