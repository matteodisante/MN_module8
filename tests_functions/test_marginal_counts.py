import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_marginal_counts_vectorized(data: np.ndarray, epsilon: np.ndarray, tol: float = 1e-12, sorted_data: np.ndarray = None) -> np.ndarray:
    """
    Calcola in modo vettoriale i conteggi marginali per un array 1D.
    Per ogni campione, conta quanti punti cadono nell'intervallo:
      [x_i - (epsilon_i/2 - tol),  x_i + (epsilon_i/2 - tol)]
      
    Se sorted_data viene fornito, lo utilizza invece di ricalcolare np.sort(data).
    
    Parameters:
        data (np.ndarray): Array 1D dei dati di una variabile.
        epsilon (np.ndarray): Array 1D delle soglie per ciascun campione (deve avere la stessa lunghezza di data).
        tol (float): Tolleranza per la riduzione del raggio.
        sorted_data (np.ndarray, opzionale): Se fornito, l'array ordinato dei dati.
    
    Returns:
        np.ndarray: Array 1D dei conteggi marginali per ciascun campione.
    """
    data = np.asarray(data)
    if data.shape[0] != epsilon.shape[0]:
        raise ValueError("La dimensione di 'epsilon' deve essere uguale a quella di 'data'.")
    
    # Calcolo del raggio marginale per ciascun campione
    r = (epsilon / 2) - tol

    # Se non è fornito, ordiniamo i dati
    if sorted_data is None:
        sorted_data = np.sort(data)
    
    # Calcolo degli estremi dell'intervallo per ciascun campione
    left_bounds = data - r
    right_bounds = data + r
    
    # Ricerca degli indici negli array ordinati
    left_indices = np.searchsorted(sorted_data, left_bounds, side='left')
    right_indices = np.searchsorted(sorted_data, right_bounds, side='right')
    
    # Il conteggio è la differenza degli indici, sottraendo 1 per escludere il punto stesso
    counts = right_indices - left_indices - 1
    return counts

def compute_marginal_counts(matrix, epsilon):
    """
    Computes the marginal counts for a single variable (1D) using NearestNeighbors for efficiency.
    For each sample, counts how many points are within a specified distance threshold (epsilon/2).

    Parameters:
        matrix (1D array-like): Input data for a single variable, reshaped as a column vector of shape (n_samples, 1).
        epsilon (1D array-like): Distance thresholds for each sample, provided as a vector.

    Returns:
        np.ndarray: Array of shape (n_samples,) containing the marginal counts for each sample.
    """
    try:
        logger.info(f"Computing marginal counts with epsilon values: {epsilon}")

        # Ensure the input matrix is in (n_samples, 1) format
        matrix = np.asarray(matrix).reshape(-1, 1)
        n_samples = matrix.shape[0]
        marginal_counts = np.zeros(n_samples)

        # Initialize NearestNeighbors using KD-tree (ottimale per 1D)
        nbrs = NearestNeighbors(metric='euclidean', algorithm='kd_tree').fit(matrix)

        # Per ogni campione, query dei vicini entro il raggio: (epsilon[i]/2) - 1e-12
        #Nel caso: neighbors = nbrs.radius_neighbors(radius=(epsilon[i] / 2) - 1e-12)[0][0] non è considerato il query point tra i suoi vicini!!!!!!!
        #Nel caso: neighbors = nbrs.radius_neighbors(matrix[i].reshape(1, -1), radius=(epsilon[i] / 2) - 1e-12)[0][0] il querypoint è contato tra i suoi vicini !!!!
        for i in range(n_samples):
            neighbors = nbrs.radius_neighbors(matrix[i].reshape(1, -1), radius=(epsilon[i] / 2) - 1e-12)[0][0]
            print(neighbors)
            marginal_counts[i] = len(neighbors) -1
        
        logger.info(f"Marginal counts computed for {n_samples} samples.")
        return marginal_counts
    except Exception as e:
        logger.error(f"Error computing marginal counts: {e}")
        return None

if __name__ == '__main__':
    # Definizione di dati di prova (non necessariamente ordinati)
    data = np.array([1, 2, 3, 7, 9, 6, 4], dtype=float)
    # Impostiamo epsilon uguale a 4 per ogni punto, quindi raggio = 4/2 - tol ≈ 2
    epsilon = np.full(data.shape, 4.0)

    print("=== TEST DELLE FUNZIONI DI COMPUTAZIONE DEI CONTEGGI MARGINALI ===\n")
    
    # Test con il metodo vectorizzato (np.searchsorted)
    counts_vectorized = compute_marginal_counts_vectorized(data, epsilon)
    
    # Stampiamo i risultati intermedi per il metodo vectorizzato
    r = (epsilon / 2) - 1e-12
    sorted_data = np.sort(data)
    left_bounds = data - r
    right_bounds = data + r
    left_indices = np.searchsorted(sorted_data, left_bounds, side='left')
    right_indices = np.searchsorted(sorted_data, right_bounds, side='right')
    
    print("Metodo vectorizzato (np.searchsorted):")
    print("Data:", data)
    print("Epsilon:", epsilon)
    print("r (epsilon/2 - tol):", r)
    print("Sorted data:", sorted_data)
    print("Left bounds:", left_bounds)
    print("Right bounds:", right_bounds)
    print("Left indices:", left_indices)
    print("Right indices:", right_indices)
    print("Marginal counts (vectorizzato):", counts_vectorized)
    print("\n-----------------------------------------------\n")
    
    # Test con il metodo NearestNeighbors
    counts_nn = compute_marginal_counts(data, epsilon)
    
    print("Metodo basato su NearestNeighbors:")
    print("Data:", data)
    print("Epsilon:", epsilon)
    print("Marginal counts (NearestNeighbors):", counts_nn)
    
    print("\nConfronto dei risultati:")
    print("Vectorizzato:", counts_vectorized)
    print("NearestNeighbors:", counts_nn)