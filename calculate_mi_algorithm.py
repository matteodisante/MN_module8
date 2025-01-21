import sys
import os
import numpy as np

from utils.mutual_information_utils import analyze_and_save_mi_values


def main():
    # Define the input and output directories
    input_dir = "data/synthetic_data"
    output_dir = "mi_results"

    # Number of bins for mutual information estimation
    num_bins = 10

    # Run the analysis
    print(f"[INFO] Starting analysis of files in directory: {input_dir}")
    analyze_and_save_mi_values(input_dir=input_dir, output_dir=output_dir, num_bins=num_bins)
    print(f"[INFO] Analysis completed. Results saved in: {output_dir}")

# Entry point of the script
if __name__ == "__main__":
    main()

