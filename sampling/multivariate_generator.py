import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from math_utils import correlate_data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sampling')))
from univariate_generator import generate_triangular


def independent_gaussian_rv(mu, sigma, size):

    # Generazione di due variabili gaussiane indipendenti
    gauss_var1 = np.random.normal(mu, sigma, size)
    gauss_var2 = np.random.normal(mu, sigma, size)

    # Creazione di un array con le due variabili come colonne
    combined_array_gauss = np.column_stack((gauss_var1, gauss_var2))

    return combined_array_gauss

def correlated_gaussian_rv(mu, sigma, corr, size):

    combined_array_gauss = independent_gaussian_rv(mu, sigma, size)
    correlated_array_gauss = correlate_data(combined_array_gauss, corr)

    return correlated_array_gauss

def independent_uniform_rv(low, high, size):

    # Generazione di due variabili uniformi indipendenti
    uniform_var1 = np.random.uniform(low, high, size)
    uniform_var2 = np.random.uniform(low, high, size)

    # Creazione di un array con le due variabili come colonne
    combined_array_uniform = np.column_stack((uniform_var1, uniform_var2))

    return combined_array_uniform

def independent_exponential_rv(lam, size):

    scale = 1 / lam
    # Generazione di due variabili esponenziali indipendenti
    exp_var1 = np.random.exponential(scale, size)
    exp_var2 = np.random.exponential(scale, size)

    # Creazione di un array con le due variabili come colonne
    combined_array_exp = np.column_stack((exp_var1, exp_var2))

    return combined_array_exp

def circular(a, b, c, size):

    # Generazione delle variabili
    phi = np.random.uniform(0, 2 * np.pi, size)  # Φ distribuito uniformemente
    r = generate_triangular(a, b, c, size)  # R distribuito secondo la densità triangolare

    # Calcolo delle variabili X e Y
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Creazione di un array con le due variabili come colonne
    combined_array_circular = np.column_stack((x, y))

    return combined_array_circular

def gamma_exponential(theta, size):

    # Step 1: Generazione di X dalla distribuzione Gamma
    x = np.random.gamma(shape=theta, scale=1.0, size=size)

    # Step 2: Generazione di Y | X dalla distribuzione esponenziale
    y = np.random.exponential(scale=(1.0 / x))

    # Creazione di un array con le due variabili come colonne
    combined_array = np.column_stack((x, y))

    return combined_array

def ordered_wienman_exponential(theta, size):

    # Step 1: Generazione di X dalla distribuzione esponenziale con lambda = 2
    x = np.random.exponential(scale=1/2, size=size)

    # Step 2: Generazione di Y | X = x
    z = np.random.exponential(scale=theta, size=size)  # Z ~ Exp(1/theta)
    y = x + z  # Y = X + Z

    # Creazione di un array con le due variabili come colonne
    combined_array = np.column_stack((x, y))

    return combined_array
    
