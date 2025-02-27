import numpy as np
import logging
logger = logging.getLogger(__name__)
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree


def kth_nearest_distance_1d(data: np.ndarray, k: int) -> np.ndarray:
    """
    Calcola in modo vettoriale il k-esimo nearest neighbor per un array 1D in maniera efficiente in memoria.
    
    L'algoritmo prevede:
      1. Ordinare l'array.
      2. Per ogni punto, se è in posizione interna (cioè con almeno k elementi a sinistra e a destra),
         il k-esimo nearest neighbor è il min tra la differenza con il k-esimo elemento a destra
         e la differenza con il k-esimo elemento a sinistra.
      3. Per i punti vicini ai bordi, si usa il candidato disponibile (a destra per gli inizi, a sinistra per le code).
    
    Parameters:
        data (np.ndarray): Array 1D dei dati della variabile.
        k (int): Numero di vicini da considerare (escludendo il punto stesso).
    
    Returns:
        np.ndarray: Array 1D contenente per ogni campione la distanza al k-esimo nearest neighbor.
                  L'ordinamento viene fatto internamente; il risultato è in ordine crescente (ma l'ordinamento
                  non influenza la media logaritmica usata successivamente).
    """
    sorted_data = np.sort(data)
    n = sorted_data.shape[0]
    kth_dists = np.empty(n)
    
    if k <= 0 or k >= n:
        raise ValueError("k deve essere maggiore di 0 e minore del numero di campioni.")
    
    # Per i primi k punti, usiamo solo il candidato a destra
    idx = np.arange(0, k)
    kth_dists[idx] = sorted_data[idx + k] - sorted_data[idx]
    
    # Per i punti interni, disponibili sia a sinistra che a destra:
    idx = np.arange(k, n - k)
    left_candidates = sorted_data[idx] - sorted_data[idx - k]
    right_candidates = sorted_data[idx + k] - sorted_data[idx]
    kth_dists[idx] = np.minimum(left_candidates, right_candidates)
    
    # Per gli ultimi k punti, usiamo solo il candidato a sinistra
    idx = np.arange(n - k, n)
    kth_dists[idx] = sorted_data[idx] - sorted_data[idx - k]
    
    return kth_dists






#def find_k_nearest_neighbors(matrix, k):
#    """
#    Finds the k-nearest neighbors for each point in a dataset based on the max metric.
#
#    Parameters:
#        matrix (2D array-like): Input data where each row is a point and each column is a coordinate.
#        k (int): Number of nearest neighbors to find for each point.
#
#    Returns:
#        indices (2D array): Indices of the k-nearest neighbors for each point.
#        distances (2D array): Distances to the k-nearest neighbors for each point.
#    """
#    try:
#        logger.info(f"Finding {k} nearest neighbors using Chebyshev metric.")
#
#        # Use sklearn's NearestNeighbors with the Chebyshev (max) metric
#        nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev', algorithm='kd_tree').fit(matrix)
#        distances = nbrs.kneighbors(matrix)[0]
#
#        # Exclude self-neighbor and return
#        logger.info(f"Found nearest neighbors for {matrix.shape[0]} samples.")
#        return distances[:,k]  # Remove self-neighbor
#    except Exception as e:
#        logger.error(f"Error finding k-nearest neighbors with k={k}: {e}")
#        return None



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
        logger.info(f"Finding {k} nearest neighbors using Chebyshev distance (p=np.inf) with cKDTree.")
        # Costruisce l'albero KD per i dati
        tree = cKDTree(matrix)
        # p=np.inf indica la distanza Chebyshev
        # workers consente la parallelizzazione (disponibile da SciPy 1.9.0 in poi)
        distances, indices = tree.query(matrix, k=k+1, p=np.inf, workers=workers)
        logger.info(f"Found nearest neighbors for {matrix.shape[0]} samples.")
        # Restituisce la distanza del k-esimo vicino (escludendo il self-neighbor)
        return distances[:, k]
    except Exception as e:
        logger.error(f"Error finding k-nearest neighbors with k={k}: {e}")
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
