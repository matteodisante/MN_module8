import sys
import os
import numpy as np
from scipy.stats import norm, expon, uniform, pearsonr
from scipy.optimize import brentq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from decorators import time_it

# Funzione per generare campioni da una densità triangolare
def generate_triangular(a, b, c, size):
    
    u = np.random.uniform(0, 1, size)  # Campiona U uniformemente
    r = np.zeros(size)  # Pre-alloca i valori di R

    # Maschere per i due intervalli
    mask1 = u <= (b - a) / (c - a)
    mask2 = ~mask1

    # Primo intervallo (a <= r <= b)
    r[mask1] = a + np.sqrt((c - a) * (b - a) * u[mask1])

    # Secondo intervallo (b <= r <= c)
    r[mask2] = c - np.sqrt((c - a) * (c - b) * (1 - u[mask2]))

    return r

