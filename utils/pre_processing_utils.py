import numpy as np

def add_noise(matrix,  magnitude = 1e-14, seed = None):
	# Crea un generatore di numeri casuali
	rng = np.random.default_rng(seed = seed)
	
	# Genera un array di rumore uniforme
	noise_matrix = rng.uniform(low=1*magnitude, high=5*magnitude, size=matrix.shape)
	
	noisy_matrix = matrix + noise_matrix
	return noisy_matrix
	
	
