import argparse
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kurtosis, skew, norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from functions import *

def load_data(file_path):
    """Carica i dati da un file .txt con due colonne."""
    return np.loadtxt(file_path, delimiter=',')  # Assumiamo tab come separatore

def calculate_statistics(series):
    """Calcola le statistiche per una serie temporale."""
    return {
        'mean': np.mean(series),
        'std_dev': np.std(series),
        'excess_kurtosis': kurtosis(series, fisher=True),
        'skewness': skew(series)
    }

def plot_histograms(series1, series2, bins):
    """Disegna istogrammi sovrapposti per due serie temporali."""
    plt.figure(figsize=(10, 5))
    plt.hist(series1, bins=bins, alpha=0.5, label='Serie 1', color='blue', density=True)
    plt.hist(series2, bins=bins, alpha=0.5, label='Serie 2', color='green', density=True)
    plt.title('Istogrammi Sovrapposti')
    plt.xlabel('Valore')
    plt.ylabel('Densità')
    plt.legend()
    plt.show()

def plot_density(series1, series2, bins, theoretical_pdf):
    """Disegna le curve di densità sovrapposte con la funzione teorica."""
    fig, ax = plt.subplots(figsize=(10, 5))
    counts1, bin_edges1 = np.histogram(series1, bins=bins, density=True)
    counts2, bin_edges2 = np.histogram(series2, bins=bins, density=True)
    bin_centers1 = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])
    bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])
    
    ax.plot(bin_centers1, counts1, label='Densità Serie 1', color='blue')
    ax.plot(bin_centers2, counts2, label='Densità Serie 2', color='green')
    x = np.linspace(min(np.min(series1), np.min(series2)), max(np.max(series1), np.max(series2)), 1000)
    y = theoretical_pdf(x)
    ax.plot(x, y, 'r--', label='Funzione Teorica')
    
    ax.set_xlabel('Valore')
    ax.set_ylabel('Densità')
    ax.legend()
    plt.title('Curve di Densità con Funzione Teorica')
    plt.show()

def log_likelihood(series, theoretical_pdf):
    """Calcola la log-likelihood di una serie temporale per una data distribuzione teorica."""
    return np.sum(np.log(theoretical_pdf(series)))

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
    
    mi_val_empirical = mutual_information_1(data, 2)
    mi_th = -0.5*np.log(1-correlation**2)
    print(f"Th. val = {mi_th}, Emp. val = {mi_val_empirical}")
    
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
    
    

    
    
    
    
    
    
    
    
    