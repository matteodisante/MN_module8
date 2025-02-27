import numpy as np

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
    for k in [1, 5]:
        try:
            kth_dists = kth_nearest_distance_1d(data3, k)
            print(f"k = {k}: kth_dists =", kth_dists)
        except Exception as e:
            print(f"k = {k} ha generato un errore: {e}")

if __name__ == '__main__':
    main()