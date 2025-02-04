import numpy as np
from scipy.stats import kurtosis, skew
from functools import partial
from scipy.special import digamma
from scipy.integrate import quad


def correlate_data(independent_series, correlation):
    """
    Applica correlazione tra due serie indipendenti.

    Parameters:
        data (np.ndarray): Dati indipendenti.
        correlation (float): Correlazione desiderata.

    Returns:
        np.ndarray: Dati correlati.
    """
    
    correlation_matrix = np.array([[1, correlation], [correlation, 1]])
    L = np.linalg.cholesky(correlation_matrix)
    correlated_series = np.dot(L, independent_series.T)
    return correlated_series.T  

def calculate_statistics(series):
    """Calcola le statistiche per una serie temporale."""
    return {
        'mean': np.mean(series),
        'std_dev': np.std(series, ddof=1),
        'excess_kurtosis': kurtosis(series, fisher=True),
        'skewness': skew(series)
    }

def log_likelihood(series, theoretical_pdf):
    """Calcola la log-likelihood di una serie temporale per una data distribuzione teorica."""
    return np.sum(np.log(theoretical_pdf(series)))


def configure_target_function(target_function, **kwargs):
    """
    Create a picklable, pre-configured version of a target function.
    
    Parameters:
        target_function (callable): The bivariate target function.
        **kwargs: Parameters to pre-configure the target function.
        
    Returns:
        callable: A function ready to use, with parameters pre-configured.
    """
    return partial(target_function, **kwargs)



# Functions to compute theoretical mi

def circular_mi_theoretical(a, b, c):
    """
    Calculate the theoretical mutual information for a circular distribution.

    :param a: Inner radius of the circular distribution.
    :param b: Outer radius of the circular distribution.
    :param c: Middle radius where the triangular distribution switches.
    :return: Theoretical mutual information.
    """
    def marginal_density(x):
        if abs(x) < a:
            return 0
        elif a <= abs(x) <= c:
            return (2 / (np.pi * (b - a) * (c - a))) * (
                np.sqrt(c**2 - x**2) - np.sqrt(a**2 - x**2) -
                a * np.log((np.sqrt(c**2 - x**2) + c) / (np.sqrt(a**2 - x**2) + a))
            )
        elif c < abs(x) <= b:
            return (2 / (np.pi * (b - a) * (b - c))) * (
                b * np.log((np.sqrt(b**2 - x**2) + b) / (np.sqrt(c**2 - x**2) + c)) -
                np.sqrt(b**2 - x**2) + np.sqrt(c**2 - x**2)
            )
        else:
            return 0

    def marginal_entropy():
        result, _ = quad(lambda x: -marginal_density(x) * np.log(marginal_density(x) + 1e-12), -b, b)
        return result

    h_x = marginal_entropy()

    h_xy = (
        0.5 + np.log10(np.pi * (b - a))
        - c**2 / ((c - a) * (b - c)) * (np.log(c) - 1.5)
        + a**2 / ((b - a) * (c - a)) * (np.log(a) - 1.5)
        + b**2 / ((b - a) * (b - c)) * (np.log(b) - 1.5)
    )

    return 2 * h_x - h_xy


def ordered_wienman_exponential_mi_theoretical(theta):
    """
    :param theta: Parameter of the distribution (0 < theta < 1 for the valid range)
    """
    if theta < 0.5:
        mi_exact = (
            np.log((2 * theta) / (1 - 2 * theta)) +
            digamma(1 / (1 - 2 * theta)) -
            digamma(1)
        )
    elif theta > 0.5:
        mi_exact = (
            np.log((2 * theta - 1) / theta) +
            digamma(2 * theta / (2 * theta - 1)) -
            digamma(1)
        )
    else:
        mi_exact = - digamma(1)
    return mi_exact


def gamma_exponential_mi_theoretical(theta):
    if theta <= 0:
        raise ValueError("The parameter theta must be greater than 0.")
    mi_exact = digamma(theta + 1) - np.log(theta)
    return mi_exact

def correlated_gaussian_rv_mi_theoretical(corr):
    mi_exact = - 0.5 * np.log(1-corr**2)
    return mi_exact

def independent_exponential_rv_mi_theoretical():
    mi_exact = 0
    return mi_exact

def independent_gaussian_rv_mi_theoretical():
    mi_exact = 0
    return mi_exact

def independent_uniform_rv_mi_theoretical():
    mi_exact = 0
    return mi_exact