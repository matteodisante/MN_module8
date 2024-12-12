import numpy as np
from scipy.stats import pearsonr, kurtosis, skew, norm


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

# Gaussiana bivariata
def gaussian_target_distribution(x, y):
    pos = np.stack((x, y), axis=-1)  # Combina x e y in coordinate (N, 2) in modo vettorializzato
    rv = multivariate_normal(mean, cov_matrix)
    return rv.pdf(pos)  # Restituisce direttamente la densità
    
# Gamma-exponential multivariate distribution 
def gamma_exponential_target_distribution(x, y, theta = 2):
    return 1/gamma(x) * x**theta*np.exp(-x-x*y)
    

# Weinman order exponential distribution    
def weinman_ordered_target_distribution(x, y, theta = 1):
    return 2/theta * np.exp(-2*x - (y-x)/theta)    
    
    
# Weinman order exponential distribution    
def circle_target_distribution(x, y, theta = 1):
    prefactor = 1/(np.pi* (b-a) * (c-a))
    array_norm = np.linalg.norm(np.array([x,y]))
    
    if (a <= array_norm <= c):
        return prefactor * (array_norm - a)/array_norm
        
    if (c <= array_norm <= b):
        return prefactor * (b - array_norm)/array_norm
    
    else:
        return 0
        
        
    return     
    
    

