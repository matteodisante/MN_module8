import numpy as np
from scipy.stats import pearsonr, kurtosis, skew, norm, multivariate_normal
from scipy.special import gamma
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



def gaussian_target_distribution(x, y, mean_x=0, mean_y=0, std_dev_x=1, std_dev_y=1, correlation=0):
    mean = [mean_x, mean_y]
    cov_matrix = [
        [std_dev_x**2, correlation * std_dev_x * std_dev_y],
        [correlation * std_dev_x * std_dev_y, std_dev_y**2]
    ]
    pos = np.stack((x, y), axis=-1)
    rv = multivariate_normal(mean, cov_matrix)
    return rv.pdf(pos)
    
               
    
# Gamma-exponential multivariate distribution 
def gamma_exponential_target_distribution(x, y, theta=2):
    """
    Compute the gamma-exponential target distribution.

    Parameters:
        x (float or array): x-coordinate(s)
        y (float or array): y-coordinate(s)
        theta (float): Parameter of the distribution.

    Returns:
        float or array: Value(s) of the gamma-exponential distribution.
    """
    return (1 / gamma(x)) * (x**theta) * np.exp(-x - x * y)
    

# Weinman order exponential distribution    
def weinman_ordered_target_distribution(x, y, theta = 1):
    return 2/theta * np.exp(-2*x - (y-x)/theta)    
    

    
    
def bivariate_uniform_target_distribution(x, y, a=2, b=1, correlation=0):
    """
    Compute the uniform PDF over a rotated elliptical domain, given axes and correlation.
    
    Parameters:
        x (float or array): x-coordinate(s).
        y (float or array): y-coordinate(s).
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        correlation (float): Desired correlation coefficient (-1 <= correlation <= 1).
    
    Returns:
        float or array: Value(s) of the PDF.
    """
    # Check valid correlation range
    if not (-1 <= correlation <= 1):
        raise ValueError("Correlation must be between -1 and 1.")
    
    # Compute the rotation angle theta from the correlation
    if a != b:
        theta = 0.5 * np.arcsin((correlation * (a**2 + b**2)) / (a**2 - b**2))
    else:
        theta = 0  # For a circle, no rotation needed
    
    # Apply rotation transformation
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    
    # Ellipse equation
    ellipse = (x_rot / a)**2 + (y_rot / b)**2
    
    # Constant density within the ellipse
    pdf = np.where(ellipse <= 1, 1 / (np.pi * a * b), 0)
    return pdf
    
    
    
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
