import numpy as np


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




def main():
    # Test 1: Array ordinato
    data1 = np.array([10,25,70,120], dtype=float)
    print("Test 1: Array ordinato:", data1)
    for k in [1, 2]:
        try:
            kth_dists = kth_nearest_distance_1d(data1, k)
            print(f"k = {k}: kth_dists =", kth_dists)
        except Exception as e:
            print(f"k = {k} ha generato un errore: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 2: Array non ordinato
    data2 = np.array([4, 1, 5, 2, 3], dtype=float)
    print("Test 2: Array non ordinato:", data2)
    print("Test 2: Array ordinato:", np.sort(data2))
    for k in [1, 2]:
        try:
            kth_dists = kth_nearest_distance_1d(data2, k)
            print(f"k = {k}: kth_dists =", kth_dists)
        except Exception as e:
            print(f"k = {k} ha generato un errore: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 3: Array casuale di 20 elementi
    np.random.seed(0)
    data3 = np.random.rand(20) * 100
    print("Test 3: Array casuale (20 elementi):", data3)
    print("Test 3: Array ordinato (20 elementi):", np.sort(data3))
    for k in [1, 5]:
        try:
            kth_dists = kth_nearest_distance_1d(data3, k)
            print(f"k = {k}: kth_dists =", kth_dists)
        except Exception as e:
            print(f"k = {k} ha generato un errore: {e}")

if __name__ == '__main__':
    main()