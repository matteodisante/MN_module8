import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import KBinsDiscretizer


def mutual_information_binning_adaptive(data, num_bins):
    """
    Estimate mutual information using adaptive binning and first-order correction.

    :param data: 2D NumPy array (n_samples, 2), where each row is a sample (x, y).
    :param num_bins: Number of bins alog each axis for discretizing the data. Total number of bins = num_bins^2
    :return: Estimated mutual information (MI).
    """

    # Adaptive binning using quantile strategy
    discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='quantile')
    
    try:
        binned_data = discretizer.fit_transform(data)
        x_bin_edges = discretizer.bin_edges_[0]
        y_bin_edges = discretizer.bin_edges_[1]
    except Exception as e:
        logging.error(f"Error in KBinsDiscretizer: {e}")
        return np.nan, None, None, None, None, None, None, None  # If binning fails

    # Compute joint histogram
    joint_hist, x_edges, y_edges = np.histogram2d(binned_data[:, 0], binned_data[:, 1], bins=[num_bins, num_bins])


    # Compute total number of cells
    total_cells =  (len(x_bin_edges) - 1) * (len(y_bin_edges) - 1)  # Subtract 1 from each to get the number of intervals
    
    # Compute how many cells contain at least one point
    non_empty_cells = np.count_nonzero(joint_hist)
    
    # Normalize joint histogram to get probabilities
    joint_prob = joint_hist / np.sum(joint_hist)
    
    # Compute marginal probabilities
    p_x = np.sum(joint_prob, axis=1, keepdims=True)  # Shape (max_bins, 1)
    p_y = np.sum(joint_prob, axis=0, keepdims=True)  # Shape (1, max_bins)
    

    # Compute mutual information
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            # Usa il broadcasting per ottenere il prodotto esterno
            outer_prob = p_x * p_y  
            valid_mask = (joint_prob > 0) & (outer_prob > 0)
            mi = np.sum(np.where(valid_mask, joint_prob * np.log(joint_prob / outer_prob), 0))
    except Exception as e:
        logging.error(f"Error computing mutual information: {e}")
        return np.nan, None, None, None, None, None, None, None   # If computation fails, return MI=0
    
    return mi, total_cells, non_empty_cells, joint_hist, x_edges, y_edges, x_bin_edges, y_bin_edges




if __name__ == '__main__':
    

    # Controlla se è stato passato un file.txt come argomento
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        try:
            data = np.loadtxt(filename)
        except Exception as e:
            logging.error(f"Errore nel caricamento del file {filename}: {e}")
            sys.exit(1)
        # Verifica che il file abbia due colonne
        if data.ndim != 2 or data.shape[1] != 2:
            logging.error(f"Il file deve contenere due colonne, ma ne sono state trovate {data.shape[1]}")
            sys.exit(1)
        print(f"Caricati {data.shape[0]} punti dal file {filename}")
    else:
        # Nessun file passato: genera 1000 punti casuali
        np.random.seed(42)
        n_points = 100000
        # Genera dati gaussiani:
        x = np.random.normal(0, 0.8, n_points)
        y = np.random.normal(0, 1.0, n_points)
        # Limita (clip) i dati al rettangolo [-2,2] x [-3,3]
        x = np.clip(x, -2, 2)
        y = np.clip(y, -3, 3)
        data = np.column_stack((x, y))
        print(f"Generati {n_points} punti casuali")

    # Imposta il numero di bin per ciascun asse
    num_bins = 1000  

    # Calcola la mutual information e recupera anche il joint histogram e i bordi
    mi, total_cells, non_empty_cells, joint_hist, x_edges, y_edges, real_x_edges, real_y_edges = mutual_information_binning_adaptive(data, num_bins)

    print(f"\nMutual Information: {mi:.4f}")
    print(f"Total number of cells: {total_cells}")
    #print("\nJoint Histogram (popolazione delle celle):")
    print(joint_hist)
    
    
    #print(f"real_x_edges_diff: {np.diff(real_x_edges)}")
    #print(f"real_y_edges_diff: {np.diff(real_y_edges)}")
    print(f"Actual number of cells of the partition:{total_cells}")
    print(f"Non empy cells: {non_empty_cells}")
    
    
    # Plot the heatmap of discretized cells with accurate cell proportions
    plt.figure(figsize=(10, 7))
    plt.pcolormesh(real_x_edges, real_y_edges, joint_hist.T, cmap='viridis')  # Nota: .T perché pcolormesh richiede una trasposta per allineare con gli edge corretti
    plt.colorbar(label='Number of points per cell')
    plt.xlabel('X edge')
    plt.ylabel('Y edge')
    title_string = f'Partitions of {total_cells} total cells - Non empy cells= {non_empty_cells}'
    plt.title(title_string)
    plt.savefig(title_string + '.png', dpi = 300)
    plt.show()
        
