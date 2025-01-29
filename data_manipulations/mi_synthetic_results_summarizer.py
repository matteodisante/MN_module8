import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from interface_utils import navigate_directories

def read_mi_file(file_path):
    """
    Reads a mutual information file and returns its content as a DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep=r'\s+')  # Use regex separator for whitespace
        if "k" not in df.columns or len(df.columns) < 2:
            raise ValueError(f"Invalid file format in {file_path}. Expected 'k' and 'mi_*' columns.")
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def calculate_mi_statistics(file_paths):
    """
    Calculates mean and unbiased standard deviation for mutual information values across multiple files.
    """
    combined_data = []
    for file_path in file_paths:
        df = read_mi_file(file_path)
        if df is not None:
            combined_data.append(df)
    
    if not combined_data:
        print("No valid data found.")
        return None
    
    merged_df = combined_data[0][['k']].copy()
    for idx, df in enumerate(combined_data):
        merged_df = merged_df.merge(df, on='k', how='outer', suffixes=('', f'_file{idx}'))
    
    mi_columns = [col for col in merged_df.columns if col.startswith("mi")]
    stats_df = merged_df[['k']].copy()
    stats_df['mean'] = merged_df[mi_columns].mean(axis=1)
    stats_df['std'] = merged_df[mi_columns].std(axis=1, ddof=1)  # Unbiased std
    
    return stats_df

def save_mi_statistics(stats_df, file_paths, output_base_dir="../data/mi_summaries"):
    """
    Saves the statistics DataFrame to a file with a descriptive name.
    """
    grouped_files = {}
    for file_path in file_paths:
        key = os.path.dirname(file_path)
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append(file_path)
    
    for group_path, group_files in grouped_files.items():
        stats_df = calculate_mi_statistics(group_files)
        if stats_df is None:
            continue
        
        parts = group_path.split(os.sep)
        mi_type = parts[parts.index("mi_numerical_results") + 1]  # mi_1 or mi_sum
        distribution_name = parts[parts.index("mi_numerical_results") + 2]  # Distribution name
        param_parts = parts[parts.index("mi_numerical_results") + 3:-1]  # All parameters before size
        size_dir = parts[-1]  # size_*
        
        param_str = "_".join(param_parts) if param_parts else "no_params"
        
        if mi_type == "mi_1":
            stats_df.rename(columns={"mean": "mean_mi_1", "std": "sigma_mi_1"}, inplace=True)
        elif mi_type == "mi_sum":
            stats_df.rename(columns={"mean": "mean_mi_sum", "std": "sigma_mi_sum"}, inplace=True)
        elif mi_type == "mi_binning":
            stats_df.rename(columns={"mean": "mean_mi_binning", "std": "sigma_mi_binning"}, inplace=True)
        else:
            print(f"[ERROR] Unsupported MI type: {mi_type}")
            continue
        
        output_dir = os.path.join(output_base_dir, mi_type, distribution_name, *param_parts, size_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file_name = f"summary_{distribution_name}_{param_str}_{size_dir}_{mi_type}.txt"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        stats_df.to_csv(output_file_path, index=False, sep='\t')
        
        print(f"Statistics saved to {output_file_path}")

if __name__ == "__main__":
    print("Navigate and select mutual information files.")
    selected_files = navigate_directories(start_path='../data', multi_select=True, file_extension=".txt")
    
    if not selected_files:
        print("No files selected. Exiting.")
        exit()
    
    save_mi_statistics(None, selected_files)