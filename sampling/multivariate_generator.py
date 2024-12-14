import numpy as np
from concurrent.futures import ProcessPoolExecutor



def proposal_sampler(x, y, delta=1.0):
    """Sampler uniforme entro una distanza delta."""
    angle = np.random.uniform(0, 2 * np.pi)  # Angolo casuale
    radius = np.random.uniform(0, delta)  # Distanza casuale entro delta
    x_new = x + radius * np.cos(angle)
    y_new = y + radius * np.sin(angle)
    return x_new, y_new


def proposal_probability(x, y, x_new, y_new, delta=1.0):
    """Probabilità della distribuzione uniforme entro un cerchio di raggio delta."""
    dist = np.sqrt((x_new - x)**2 + (y_new - y)**2)
    if dist < delta:
        return 1 / (np.pi * delta**2)
    return 0  # Fuori dal cerchio di raggio delta

# Algoritmo di Metropolis-Hastings
def metropolis_hastings(f, q_sampler, q_probability, x0, y0, steps, delta=1.0, check = True):
    points = []
    x, y = x0, y0
    current_pdf = f(x, y)  # Calcola la PDF una volta per il punto iniziale
    
    if current_pdf == 0 or np.isnan(current_pdf):
        raise ValueError(f"Initial point ({x0}, {y0}) has zero or invalid probability.")
    
    for _ in range(steps):
        # Generazione del punto proposto
        if check:
            valid_point = False
            while not valid_point: 
                x_new, y_new = q_sampler(x, y, delta)
                # Assicura che entrambi siano positivi
                valid_point = x_new > 1e-50 and y_new > 1e-50
            
        else:
            x_new, y_new = q_sampler(x, y, delta)

        # Calcolo del rapporto di accettazione
        if f(x, y) < 1e-50:
            print(f(x, y))
        
        acceptance_ratio = (
            f(x_new, y_new) * q_probability(x_new, y_new, x, y, delta)
        ) / (
            f(x, y) * q_probability(x, y, x_new, y_new, delta)
        )
        # Debug: stampa il rapporto di accettazione (opzionale, per il debug)
        #print(acceptance_ratio)

        # Accetta o rifiuta il punto
        if np.random.uniform(0, 1) < acceptance_ratio:
            x, y = x_new, y_new
        points.append((x, y))
    return np.array(points)
    

    
# Funzione worker da usare nei processi paralleli
def worker(seed, f, q_sampler, q_probability, x0, y0, steps_per_core, delta, burn_in, check):
    np.random.seed(seed)  # Imposta un seed unico per ogni processo
    samples = metropolis_hastings(f, q_sampler, q_probability, x0, y0, steps_per_core, delta, check = check)
    return samples[burn_in:]  # Scarta i primi `burn_in` punti



# Funzione per parallelizzare Metropolis-Hastings
def run_parallel_metropolis_hastings(f, q_sampler, q_probability, x0, y0, steps, delta, n_cores, burn_in=0, check = False):
    steps_per_core = steps // n_cores
    seeds = [np.random.randint(1e6) for _ in range(n_cores)]  # Genera semi per i processi

    # Usa ProcessPoolExecutor per parallelizzare
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(
            executor.map(
                worker,
                seeds,
                [f] * n_cores,
                [q_sampler] * n_cores,
                [q_probability] * n_cores,
                [x0] * n_cores,
                [y0] * n_cores,
                [steps_per_core] * n_cores,
                [delta] * n_cores,
                [burn_in] * n_cores,
                [check] * n_cores
            )
        )
    return np.vstack(results)