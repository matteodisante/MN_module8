import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sampling/')))
from univariate_generator import data_generator
from multivariate_generator import run_parallel_metropolis_hastings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))
from io_utils import save_data
from plot_utils import plot_3d_histogram, plot_marginals
from config_utils import load_config


def generate_all_data(config_file):
    config = load_config(config_file)
    output_dir = config['output_dir']

    for dist in config['distributions']:
        dist_name = dist['name']
        params = dist['params']
        sizes = dist['sizes']
        correlation = dist.get('correlation', 0.7)

        for size in sizes:
            print(f"Generazione dati: {dist_name}, size={size}")
            if dist_name in ["gaussian", "uniform", "exponential"]:
                data = data_generator(dist_name, size, params, correlation)
            else:
                data = generate_mcmc_data(dist_name, size, params)

            data_path = os.path.join(output_dir, "generated_data", dist_name, f"size_{size}.csv")
            save_data(data, data_path)

            # Plotting
            plot_dir = os.path.join(output_dir, "plots", dist_name, f"size_{size}")
            os.makedirs(plot_dir, exist_ok=True)
            plot_3d_histogram(data, f"3D Histogram: {dist_name}", os.path.join(plot_dir, "hist3d.png"))
            plot_marginals(data, os.path.join(plot_dir, "marginals.png"))

if __name__ == "__main__":
    generate_all_data("config.json")