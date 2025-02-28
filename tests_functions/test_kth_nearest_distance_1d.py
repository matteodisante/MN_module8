import numpy as np
from numba import njit

@njit
def _kth_nearest_distance_numba(sorted_data, k):
    """
    Funzione compilata con Numba che, dato un array già ordinato,
    calcola per ciascun elemento la distanza al k-esimo nearest neighbor.
    
    Per ogni elemento sorted_data[i]:
      - Calcola le distanze verso i punti a sinistra (se esistono) e verso i punti a destra.
      - Effettua una fusione (merge) delle due sequenze ordinate per selezionare il k-esimo valore.
    
    Parameters:
        sorted_data (np.ndarray): Array 1D ordinato.
        k (int): Numero di vicini (escludendo il punto stesso).
    
    Returns:
        np.ndarray: Array 1D delle distanze al k-esimo nearest neighbor (nell'ordine dell'array ordinato).
    """
    n = sorted_data.shape[0]
    kth_dists = np.empty(n)
    
    for i in range(n):
        # l e r sono i contatori per i vicini a sinistra e a destra, rispettivamente.
        l = 0
        r = 0
        count = 0
        kth = 0.0  # variabile per memorizzare la distanza corrente selezionata
        while count < k:
            # Se entrambi i lati hanno candidati disponibili:
            if l < i and r < (n - i - 1):
                left_val = sorted_data[i] - sorted_data[i - l - 1]
                right_val = sorted_data[i + r + 1] - sorted_data[i]
                if left_val <= right_val:
                    kth = left_val
                    l += 1
                else:
                    kth = right_val
                    r += 1
            elif l < i:  # Solo i punti a sinistra sono disponibili
                kth = sorted_data[i] - sorted_data[i - l - 1]
                l += 1
            else:  # Solo i punti a destra sono disponibili
                kth = sorted_data[i + r + 1] - sorted_data[i]
                r += 1
            count += 1
        kth_dists[i] = kth
    return kth_dists

def kth_nearest_distance_1d(data: np.ndarray, k: int) -> np.ndarray:
    """
    Calcola la distanza al k-esimo nearest neighbor per ciascun punto di un array 1D,
    in modo efficiente sfruttando Numba per compilare la parte iterativa.
    
    La procedura è la seguente:
      1. Ordina l'array e salva l'ordine originale.
      2. Per ciascun elemento, "merga" le distanze verso i vicini a sinistra e a destra
         (calcolate sull'array ordinato) per trovare il k-esimo valore.
      3. Ripristina l'ordine originale dei risultati.
    
    Parameters:
        data (np.ndarray): Array 1D contenente i dati.
        k (int): Numero di vicini (escludendo il punto stesso) da considerare.
    
    Returns:
        np.ndarray: Array 1D delle distanze al k-esimo nearest neighbor, riportato 
                    nell'ordine originale dei dati di input.
    
    Raises:
        ValueError: Se k <= 0 o se k è maggiore o uguale al numero di campioni.
    """
    n = data.shape[0]
    if k <= 0 or k >= n:
        raise ValueError("k deve essere maggiore di 0 e minore del numero di campioni.")
    
    # Salva l'ordine originale e ordina i dati
    order = np.argsort(data)
    sorted_data = data[order]
    
    # Calcola le distanze al k-esimo nearest neighbor sull'array ordinato
    kth_dists_sorted = _kth_nearest_distance_numba(sorted_data, k)
    
    # Ripristina l'ordine originale dei risultati
    kth_dists = np.empty(n)
    kth_dists[order] = kth_dists_sorted
    
    return kth_dists

def main():
    """
    Esegue una serie di test sulla funzione kth_nearest_distance_1d:
      - Test 1: Array ordinato.
      - Test 2: Array non ordinato.
      - Test 3: Array casuale di 20 elementi.
    
    Per ogni test vengono eseguite le verifiche per diversi valori di k.
    """
    # Test 1: Array ordinato
    print("Test 1: Array ordinato")
    data1 = np.array([1, 2, 4, 8], dtype=np.float64)
    print("Input:", data1)
    for k in [1, 2]:
        try:
            result = kth_nearest_distance_1d(data1, k)
            print(f"Con k = {k}, kth nearest distances: {result}")
        except Exception as e:
            print(f"Errore con k = {k}: {e}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Test 2: Array non ordinato
    print("Test 2: Array non ordinato")
    data2 = np.array([8, 1, 4, 2], dtype=np.float64)
    print("Input:", data2)
    for k in [1, 2]:
        try:
            result = kth_nearest_distance_1d(data2, k)
            print(f"Con k = {k}, kth nearest distances: {result}")
        except Exception as e:
            print(f"Errore con k = {k}: {e}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Test 3: Array casuale di 20 elementi
    print("Test 3: Array casuale (20 elementi)")
    np.random.seed(42)
    data3 = np.random.rand(20) * 100
    print(f"Test 3: Array casuale (20 elementi) ORDINATO {np.sort(data3)}")
    print("Input:", data3)
    for k in [1, 5]:
        try:
            result = kth_nearest_distance_1d(data3, k)
            print(f"Con k = {k}, kth nearest distances: {result}")
        except Exception as e:
            print(f"Errore con k = {k}: {e}")

if __name__ == '__main__':
    main()