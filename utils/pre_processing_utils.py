import numpy as np

def add_noise(matrix,  magnitude = 1e-14, seed = None):
	# Crea un generatore di numeri casuali
	rng = np.random.default_rng(seed = seed)
	
	# Genera un array di rumore uniforme
	noise_matrix = rng.uniform(low=1*magnitude, high=5*magnitude, size=matrix.shape)
	
	noisy_matrix = matrix + noise_matrix
	return noisy_matrix
	
	
	
def sample_data(series1, series2, sample_size=10000, seed=42):
    """
    Randomly sample a subset of the data from two series.

    Parameters:
        series1 (array-like): First data series.
        series2 (array-like): Second data series.
        sample_size (int): Number of points to sample.
        seed (int): Seed for reproducibility.

    Returns:
        tuple: Sampled series1 and series2.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(series1), size=sample_size, replace=False)
    return series1[indices], series2[indices]	