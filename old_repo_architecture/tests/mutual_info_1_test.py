import sys
import os
import numpy as np
from scipy.special import digamma
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from functions import find_k_nearest_neighbors, compute_mutual_information

def test_dummy():
    assert True


def test_find_k_nearest_neighbors():
    # Dataset di esempio con punti in uno spazio bidimensionale più variegato
    matrix = np.array([
        [0, 10],
        [10, 0],
        [5, 5],
        [3, 7],
        [7, 3]
    ])
    k = 2

    # Chiamata alla funzione
    indices, distances = find_k_nearest_neighbors(matrix, k)

    # Valori attesi
    expected_indices = np.array([
        [3, 2],  # Vicini di [0, 10]
        [4, 2],  # Vicini di [10, 0]
        [3, 4],  # Vicini di [5, 5]
        [2, 0],  # Vicini di [3, 7]
        [2, 1]   # Vicini di [7, 3]
    ])
    expected_distances = np.array([
        [3.0, 5.0],  # Distanze per [0, 10]
        [3.0, 5.0],  # Distanze per [10, 0]
        [2.0, 2.0],  # Distanze per [5, 5]
        [2.0, 3.0],  # Distanze per [3, 7]
        [2.0, 3.0]   # Distanze per [7, 3]
    ])

    # Verifica che gli indici siano corretti
    np.testing.assert_array_equal(indices, expected_indices)

    # Verifica che le distanze siano corrette
    np.testing.assert_almost_equal(distances, expected_distances)


def test_compute_mutual_information():
    # Dataset di esempio con numeri più variati
    dataset = np.array([
        [1.0, 100.0, 0.1],
        [50.0, 200.0, 0.2],
        [25.0, 150.0, 0.15],
        [10.0, 50.0, 0.05],
        [30.0, 120.0, 0.12]
    ])
    k = 2

    # Calcola la MI
    mi, marginal_counts = compute_mutual_information(dataset, k)

    # Verifica dei conteggi marginali
    expected_counts = np.array([
        [2, 2, 3, 2, 3],  # Marginal counts for the first variable
        [3, 2, 3, 2, 2],  # Marginal counts for the second variable
        [2, 2, 3, 2, 2]   # Marginal counts for the third variable
    ])
    np.testing.assert_array_equal(marginal_counts, expected_counts)

    # Verifica del valore di MI
    # Valore atteso calcolato manualmente
    expected_mi = (
        digamma(k)
        - 1 / k
        + (dataset.shape[1] - 1) * digamma(dataset.shape[0])
        - np.mean(np.sum(digamma(expected_counts + 1), axis=0))
    )
    assert np.isclose(mi, expected_mi, atol=1e-6), f"Expected MI: {expected_mi}, got {mi}"