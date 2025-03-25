import numpy as np
import logging
logger = logging.getLogger(__name__)
from numba import njit
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree


@njit
def _kth_nearest_distance_numba(sorted_data, k):
    """
    Funzione compilata con Numba che, dato un array già ordinato, 
    calcola per ciascun elemento la distanza al k-esimo nearest neighbor.
    
    Per ogni elemento sorted_data[i]:
      - Le distanze verso i punti a sinistra sono: 
          sorted_data[i] - sorted_data[i-1], sorted_data[i] - sorted_data[i-2], ..., 
          fino a sorted_data[0] (in ordine crescente se consideriamo l'indice in modo inverso).
      - Le distanze verso i punti a destra sono: 
          sorted_data[i+1] - sorted_data[i], sorted_data[i+2] - sorted_data[i], ...,
          già in ordine crescente.
    
    Si procede "mergendo" queste due sequenze per ottenere il k-esimo valore.
    
    Parameters:
        sorted_data (np.ndarray): Array 1D ordinato.
        k (int): Numero di vicini da considerare (escludendo il punto stesso).
    
    Returns:
        np.ndarray: Array 1D delle distanze al k-esimo nearest neighbor (nell'ordine dell'array ordinato).
    """
    n = sorted_data.shape[0]
    kth_dists = np.empty(n)
    
    for i in range(n):
        # Punteri per la parte sinistra e destra
        l = 0  # contatore per i vicini a sinistra (da 1 a i)
        r = 0  # contatore per i vicini a destra (da 1 a n-i-1)
        count = 0
        kth = 0.0
        
        # Finché non abbiamo selezionato k elementi dalla fusione
        while count < k:
            # Se entrambi i lati hanno candidati disponibili
            if l < i and r < (n - i - 1):
                left_val = sorted_data[i] - sorted_data[i - l - 1]
                right_val = sorted_data[i + r + 1] - sorted_data[i]
                if left_val <= right_val:
                    kth = left_val
                    l += 1
                else:
                    kth = right_val
                    r += 1
            elif l < i:  # Solo lato sinistro disponibile
                kth = sorted_data[i] - sorted_data[i - l - 1]
                l += 1
            else:  # Solo lato destro disponibile
                kth = sorted_data[i + r + 1] - sorted_data[i]
                r += 1
            count += 1
        
        kth_dists[i] = kth
        
    return kth_dists



def kth_nearest_distance_1d(data: np.ndarray, k: int) -> np.ndarray:
    """
    Calcola la distanza al k-esimo nearest neighbor per ciascun punto di un array 1D,
    in modo efficiente e chiaro, sfruttando Numba per compilare il loop critico.
    
    La procedura è la seguente:
      1. Ordina l'array e salva l'ordine originale.
      2. Per ciascun elemento, "merga" le distanze verso i vicini a sinistra e a destra (calcolate 
         sull'array ordinato) per trovare il k-esimo valore.
      3. Ripristina l'ordine originale dei risultati.
    
    Parameters:
        data (np.ndarray): Array 1D contenente i dati.
        k (int): Numero di vicini (escludendo il punto stesso) da considerare.
    
    Returns:
        np.ndarray: Array 1D delle distanze al k-esimo nearest neighbor, riportato 
                    nell'ordine originale dei dati di input.
    
    Raises:
        ValueError: Se k <= 0 o k è maggiore o uguale al numero di campioni.
    """
    n = data.shape[0]
    if k <= 0 or k >= n:
        raise ValueError("k deve essere maggiore di 0 e minore del numero di campioni.")
    
    # Ordina i dati e salva l'ordine originale
    order = np.argsort(data)
    sorted_data = data[order]
    
    # Calcola le distanze al k-esimo nearest neighbor sull'array ordinato
    kth_dists_sorted = _kth_nearest_distance_numba(sorted_data, k)
    
    # Ripristina l'ordine originale
    kth_dists = np.empty(n)
    kth_dists[order] = kth_dists_sorted
    
    return kth_dists



def find_k_nearest_neighbors(matrix, k, workers=2):
    """
    Finds the k-th nearest neighbor for each point in a dataset using the Chebyshev (max) metric,
    optimized for 2D using scipy.spatial.cKDTree, con supporto alla parallelizzazione (SciPy >= 1.9.0).

    Parameters:
        matrix (2D array-like): Input data where each row is a point.
        k (int): Number of nearest neighbors to consider (excluding the point itself).
        workers (int, optional): Number of workers (threads) to use for parallel processing.


    Returns:
        np.ndarray: 1D array containing the distance to the k-th nearest neighbor for each point.
    """
    try:
        # Costruisce l'albero KD per i dati
        tree = cKDTree(matrix)
        # p=np.inf indica la distanza Chebyshev
        # workers consente la parallelizzazione (disponibile da SciPy 1.9.0 in poi)
        distances, indices = tree.query(matrix, k=k+1, p=np.inf, workers=workers)
        return distances[:, k]
    except Exception as e:
        raise ValueError(f'Error finding k-non distances in the joint space: {e}')



def compute_marginal_counts(matrix, epsilon, workers = 2):
    data = np.asarray(matrix).flatten()
    epsilon = np.asarray(epsilon).flatten()
    
    if data.shape[0] != epsilon.shape[0]:
        raise ValueError("Le dimensioni di 'matrix' ed 'epsilon' devono coincidere.")
    
    try:
        tol = 10*np.finfo(np.float64).eps
        r = (epsilon / 2) - tol
        # Costruisci l'albero KD
        tree = cKDTree(data[:, None])  # KDTree richiede un input 2D
        # Trova tutti i punti vicini per ogni punto in un'unica chiamata vettorizzata
        n_points_array = tree.query_ball_point(data[:, None], r, return_length=True, workers = workers)
        return n_points_array - 1
        
    except Exception as e:
        logger.error(f"Error computing marginal counts with numpy: {e}")
        return None

