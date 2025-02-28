import numpy as np
import logging
logger = logging.getLogger(__name__)
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree


def kth_nearest_distance_1d(data: np.ndarray, k: int) -> np.ndarray:
    """
    Calcola in maniera vettoriale la distanza al k-esimo nearest neighbor per ciascun punto 
    di un array 1D, restituendo i risultati nell'ordine originale dei dati di input.
    
    L'algoritmo prevede:
      1. Ordinare l'array e salvare l'ordine originale.
      2. Per ogni punto:
         - Se si trova in una posizione interna (con almeno k elementi sia a sinistra che a destra),
           la distanza è il minimo tra la differenza con il k-esimo elemento a destra e quella con 
           il k-esimo elemento a sinistra.
         - Se il punto è vicino al bordo, si usa il candidato disponibile: 
           a destra per i primi k elementi e a sinistra per gli ultimi k elementi.
      3. Riposizionare le distanze calcolate affinché corrispondano all'ordine originale dell'array.
    
    Parameters:
        data (np.ndarray): Array 1D contenente i dati.
        k (int): Numero di vicini da considerare (escludendo il punto stesso).
    
    Returns:
        np.ndarray: Array 1D delle distanze al k-esimo nearest neighbor, ordinate secondo l'array 
                    di input originale.
    
    Raises:
        ValueError: Se k <= 0 o k >= numero di campioni.
    """
    # Ottieni l'ordine di ordinamento dell'array di input
    order = np.argsort(data)
    # Ordina i dati secondo l'ordine calcolato
    sorted_data = data[order]
    n = sorted_data.shape[0]
    
    # Inizializza un array per salvare le distanze calcolate sui dati ordinati
    kth_dists_sorted = np.empty(n)
    
    # Controlla che k sia un valore valido
    if k <= 0 or k >= n:
        raise ValueError("k deve essere maggiore di 0 e minore del numero di campioni.")
    
    # Calcolo per i primi k elementi (non hanno k elementi a sinistra)
    # Per questi punti consideriamo solo il k-esimo elemento a destra
    idx = np.arange(0, k)
    kth_dists_sorted[idx] = sorted_data[idx + k] - sorted_data[idx]
    
    # Calcolo per gli elementi interni (con almeno k elementi sia a sinistra che a destra)
    idx = np.arange(k, n - k)
    # Distanza verso il k-esimo elemento a sinistra
    left_candidates = sorted_data[idx] - sorted_data[idx - k]
    # Distanza verso il k-esimo elemento a destra
    right_candidates = sorted_data[idx + k] - sorted_data[idx]
    # Seleziona il minimo tra le due distanze
    kth_dists_sorted[idx] = np.minimum(left_candidates, right_candidates)
    
    # Calcolo per gli ultimi k elementi (non hanno k elementi a destra)
    # Per questi punti consideriamo solo il k-esimo elemento a sinistra
    idx = np.arange(n - k, n)
    kth_dists_sorted[idx] = sorted_data[idx] - sorted_data[idx - k]
    
    # Ripristina l'ordine originale: per ogni indice originale, assegna la distanza calcolata 
    kth_dists = np.empty(n)
    kth_dists[order] = kth_dists_sorted
    
    return kth_dists





def find_k_nearest_neighbors(matrix, k, workers=3):
    """
    Finds the k-th nearest neighbor for each point in a dataset using the Chebyshev (max) metric,
    optimized for 2D using scipy.spatial.cKDTree, con supporto alla parallelizzazione (SciPy >= 1.9.0).

    Parameters:
        matrix (2D array-like): Input data where each row is a point.
        k (int): Number of nearest neighbors to consider (excluding the point itself).
        workers (int, optional): Number of workers (threads) to use for parallel processing.
                                 If -1 is given, all CPU threads are used. Default is -1.

    Returns:
        np.ndarray: 1D array containing the distance to the k-th nearest neighbor for each point.
    """
    try:
        # Costruisce l'albero KD per i dati
        tree = cKDTree(matrix)
        # p=np.inf indica la distanza Chebyshev
        # workers consente la parallelizzazione (disponibile da SciPy 1.9.0 in poi)
        distances, indices = tree.query(matrix, k=k+1, p=np.inf, workers=workers)
        # Restituisce la distanza del k-esimo vicino (escludendo il self-neighbor)
        return distances[:, k]
    except Exception as e:
        return None






def compute_marginal_counts(matrix, epsilon, tol: float = 1e-13) -> np.ndarray:
    """
    Computes the marginal counts for a single variable (1D) using a vectorized NumPy approach.
    For each sample, counts how many points fall within the interval:
        [x_i - (epsilon_i/2 - tol), x_i + (epsilon_i/2 - tol)]
    (the sample itself is not counted).
    
    Parameters:
        matrix (1D array-like): Input data for a single variable. Should be of shape (n_samples,).
        epsilon (1D array-like): Distance thresholds for each sample (same length as matrix).
        tol (float): Tolerance to subtract from the radius (default 1e-12).
    
    Returns:
        np.ndarray: 1D array of marginal counts for each sample.
    """
    try:
        # Ensure that the input arrays are 1D
        data = np.asarray(matrix).flatten()
        epsilon = np.asarray(epsilon).flatten()
        if data.shape[0] != epsilon.shape[0]:
            raise ValueError("The dimensions of 'matrix' and 'epsilon' must match.")
        
        # Effective radius for each sample
        r = (epsilon / 2) - tol
        
        # Sort data for efficient interval search
        sorted_data = np.sort(data)
        
        # Calculate the left and right bounds of the interval for each sample
        left_bounds = data - r
        right_bounds = data + r
        
        # Use np.searchsorted to find insertion indices in the sorted array.
        left_indices = np.searchsorted(sorted_data, left_bounds, side='left')
        right_indices = np.searchsorted(sorted_data, right_bounds, side='right')
        
        # The count is the difference in indices, subtracting 1 to exclude the sample itself.
        counts = right_indices - left_indices - 1
        
        return counts
    except Exception as e:
        logger.error(f"Error computing marginal counts with numpy: {e}")
        return None
