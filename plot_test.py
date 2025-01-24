import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './utils/')))
from interface_utils import user_select_mi_types, user_select_figures, user_select_combination, user_select_k_or_bins
from plot_utils import generate_plot
from io_utils import load_data_csv, ensure_directory_and_handle_file_conflicts
from interface_utils import navigate_directories

if __name__ == "__main__":
    output_dir = "./plots/"
    selected_mi_types = user_select_mi_types()
    selected_figures = user_select_figures()
    selected_files = navigate_directories(start_path='./data/', file_extension=".csv")
    if not selected_files:
        print("No files selected. Exiting.")
        sys.exit(0)
    selected_k_or_bins = user_select_k_or_bins()
    combine = user_select_combination()

    for fig in selected_figures:
        for x_col, y_col, yerr_col in selected_mi_types:
            generate_plot(
                file_paths=selected_files,
                x_col=x_col,
                y_col=y_col,
                yerr_col=yerr_col,
                title=f"Figure {fig}",
                xlabel=r"$\frac{1}{N}$" if fig == 2 else "N",
                ylabel=y_col,
                output_dir=output_dir,
                combine=combine,
                theoretical_value=0,
                k_or_bins=selected_k_or_bins,
                x_transform=(lambda x: x / selected_k_or_bins[0]) if fig != 2 else None
            )
    print("Figures generated successfully.")


