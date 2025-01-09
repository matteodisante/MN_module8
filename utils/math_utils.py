import numpy as np
from scipy.stats import kurtosis, skew
from functools import partial


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
