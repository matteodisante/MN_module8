import os
import numpy as np
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from interface_utils import navigate_directories


def read_mi_file(file_path):
    """
    Reads a mutual information file and returns its content as a DataFrame.

    :param file_path: Path to the MI file.
    :return: DataFrame with columns ['k', 'mi_1'] or corresponding columns.
    """
    try:
        df = pd.read_csv(file_path, delim_whitespace=True)
        if "k" not in df.columns or len(df.columns) < 2:
            raise ValueError(f"Invalid file format in {file_path}. Expected 'k' and 'mi_*' columns.")
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def calculate_mi_statistics(file_paths):
    """
    Calculates mean and unbiased standard deviation for mutual information values across multiple files.

    :param file_paths: List of file paths.
    :return: DataFrame with columns ['k', 'mean', 'std'].
    """
    combined_data = []

    for file_path in file_paths:
        df = read_mi_file(file_path)
        if df is not None:
            combined_data.append(df)

    if not combined_data:
        print("No valid data found.")
        return None

    # Merge all DataFrames on the 'k' column
    merged_df = combined_data[0][['k']].copy()
    for idx, df in enumerate(combined_data):
        merged_df = merged_df.merge(df, on='k', how='outer', suffixes=('', f'_file{idx}'))

    # Extract only the columns with MI values
    mi_columns = [col for col in merged_df.columns if col.startswith("mi")]
    stats_df = merged_df[['k']].copy()

    # Calculate mean and unbiased standard deviation
    stats_df['mean'] = merged_df[mi_columns].mean(axis=1)
    stats_df['std'] = merged_df[mi_columns].std(axis=1, ddof=1)  # ddof=1 for unbiased std

    return stats_df


def save_mi_statistics(stats_df, file_paths, output_base_dir="data/mi_summaries"):
    """
    Saves the statistics DataFrame to a file, determining the output path based on input file paths.

    :param stats_df: DataFrame containing statistics.
    :param file_paths: List of input file paths.
    :param output_base_dir: Base directory for saving the summary files.
    """
    # Determine common base path and relative structure
    first_file_path = file_paths[0]
    relative_path = os.path.relpath(first_file_path, start="data/mi_numerical_results")
    parts = relative_path.split(os.sep)

    # Extract the directory structure and size from the input path
    size_dir = None
    mi_type = None

    # Extract size and MI type (mi_1, mi_sum, mi_binning)
    for part in parts:
        if part.startswith("size_"):
            size_dir = part
        if part.startswith("mi_"):
            mi_type = part

    if size_dir is None or mi_type is None:
        print(f"[ERROR] Could not determine size or MI type from input path structure: {relative_path}")
        return

    # Adjust column names based on the MI type
    if mi_type == "mi_1":
        stats_df.rename(columns={"mean": "mean_mi_1", "std": "sigma_mi_1"}, inplace=True)
    elif mi_type == "mi_sum":
        stats_df.rename(columns={"mean": "mean_mi_sum", "std": "sigma_mi_sum"}, inplace=True)
    elif mi_type == "mi_binning":
        stats_df.rename(columns={"mean": "mean_mi_binning", "std": "sigma_mi_binning"}, inplace=True)
    else:
        print(f"[ERROR] Unsupported MI type: {mi_type}")
        return

    # Construct the output directory and file path
    output_dir = os.path.join(output_base_dir, *parts[:parts.index(size_dir)])
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, f"{size_dir}_summary.txt")
    stats_df.to_csv(output_file_path, index=False, sep='\t')

    print(f"Statistics saved to {output_file_path}")
    

if __name__ == "__main__":
    # Navigate and select files
    print("Navigate and select mutual information files.")
    selected_files = navigate_directories(start_path='.', multi_select=True, file_extension=".txt")

    if not selected_files:
        print("No files selected. Exiting.")
        exit()

    # Calculate statistics
    stats_df = calculate_mi_statistics(selected_files)

    if stats_df is not None:
        print("Statistics calculated:")
        print(stats_df)

        # Save the statistics to the appropriate directory
        save_mi_statistics(stats_df, selected_files)