import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Define N and k values
N_values = np.array([100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000])
k_values = np.array([1, 25, 50, 100, 500, 2500, 5000])

# Funzione per calcolare y
def compute_y(N, k):
    return N*np.log2(N) + N*k

# Funzione lineare per il fit
def linear_fit(x, a, b):
    return a * x + b

# === Primo grafico: Raggruppamento per k === #
plt.figure(figsize=(8, 6))
plt.xscale('log')
plt.yscale('log')

for k in k_values:
    valid_N = N_values[N_values > k]  # Considera solo N > k
    if len(valid_N) < 3:  # Serve almeno 3 punti per il fit
        continue

    x_vals = k / valid_N
    y_vals = compute_y(valid_N, k)

    # Fit lineare sui primi 3 punti
    num_fit_points = 2
    log_x_fit = np.log(x_vals[-num_fit_points:])
    log_y_fit = np.log(y_vals[-num_fit_points:])
    params, _ = curve_fit(linear_fit, log_x_fit, log_y_fit)
    slope, intercept = params

    # Plot dati
    plt.plot(x_vals, y_vals, 'o-', label=f'k={k}, slope primi {num_fit_points} pts={slope:.2f}')
    
    # Plot fit su tutti i punti
    plt.plot(x_vals, np.exp(linear_fit(np.log(x_vals), *params)), '--')

plt.xlabel('k/N',fontsize=16)
plt.ylabel(rf'$N \log_{2}N + Nk$', fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Salvataggio in PDF con margini tight
plt.savefig("plot_k_grouped.pdf", bbox_inches='tight', dpi = 300)
plt.show()

# === Secondo grafico: Raggruppamento per N === #
plt.figure(figsize=(8, 6))
plt.xscale('log')
plt.yscale('log')

for N in N_values:
    valid_k = k_values[k_values < N]  # Considera solo k < N
    if len(valid_k) < 2:  # Serve almeno 2 punti per il fit
        continue

    x_vals = valid_k / N
    y_vals = compute_y(N, valid_k) #- N*(np.log2(N))**2

    # Scegli il numero di punti per il fit
    num_fit_points = 2 if valid_k[-1] >= 500 else 2
    if len(x_vals) < num_fit_points:
        continue  # Se non ci sono abbastanza punti, salta

    # Fit sugli ultimi num_fit_points punti
    log_x_fit = np.log(x_vals[-num_fit_points:])
    log_y_fit = np.log(y_vals[-num_fit_points:])
    params, _ = curve_fit(linear_fit, log_x_fit, log_y_fit)
    slope, intercept = params

    # Plot dati
    plt.plot(x_vals, y_vals, 'o-', label=f'N={N}, slope ultimi {num_fit_points} pts={slope:.2f}')
    
    # Plot fit su tutti i punti
    plt.plot(x_vals, np.exp(linear_fit(np.log(x_vals), *params)), '--')

plt.xlabel('k/N',fontsize=16)
plt.ylabel(rf'$N \log_{2} N + Nk$', fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Salvataggio in PDF con margini tight
plt.savefig("plot_N_grouped.pdf", bbox_inches='tight', dpi = 300)
plt.show()