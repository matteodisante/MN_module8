import os
import sys
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from interface_utils import navigate_directories

def read_mi_file(file_path):
    """
    Reads a mutual information file and returns its content as a DataFrame.
    Automatically detects if the key column is 'k' or 'bins_number'.
    """
    try:
        df = pd.read_csv(file_path, sep=r'\s+')  # Use regex separator for whitespace
        
        # Detect column type ('k' or 'bins_number')
        if "k" in df.columns and df.shape[1] > 1:
            key_col = "k"
        elif "bins_number" in df.columns and df.shape[1] > 1:
            key_col = "bins_number"
        else:
            raise ValueError(f"Invalid file format in {file_path}. Expected 'k' or 'bins_number' with 'mi_*' columns.")
        
        return df, key_col
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None, None

def calculate_mi_statistics(file_paths):
    """
    Calculates mean and unbiased standard deviation for MI values across multiple files.
    """
    combined_data = []
    key_col = None  # Determines whether to use 'k' or 'bins_number'
    
    for file_path in file_paths:
        df, detected_key_col = read_mi_file(file_path)
        if df is not None:
            if key_col is None:
                key_col = detected_key_col  # Set key column based on first valid file
            elif key_col != detected_key_col:
                logging.warning(f"Inconsistent key columns detected: {key_col} vs {detected_key_col}. Skipping {file_path}.")
                continue
            combined_data.append(df)
    
    if not combined_data:
        logging.error("No valid data found.")
        return None, key_col
    
    merged_df = combined_data[0][[key_col]].copy()
    for idx, df in enumerate(combined_data):
        merged_df = merged_df.merge(df, on=key_col, how='outer', suffixes=('', f'_file{idx}'))
    
    mi_columns = [col for col in merged_df.columns if col.startswith("mi")]
    stats_df = merged_df[[key_col]].copy()
    stats_df['mean'] = merged_df[mi_columns].mean(axis=1)
    stats_df['std'] = merged_df[mi_columns].std(axis=1, ddof=1)  # Unbiased std
    
    return stats_df, key_col

def save_mi_statistics(stats_df, file_paths, key_col, output_base_dir="../data/mi_summaries"):
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
        stats_df, detected_key_col = calculate_mi_statistics(group_files)
        if stats_df is None:
            continue
        
        parts = group_path.split(os.sep)
        mi_type = parts[parts.index("mi_numerical_results") + 1]  # mi_1, mi_sum, mi_binning
        distribution_name = parts[parts.index("mi_numerical_results") + 2]  # Distribution name
        param_parts = parts[parts.index("mi_numerical_results") + 3:-1]  # All parameters before size
        size_dir = parts[-1]  # size_*
        
        param_str = "_".join(param_parts) if param_parts else "no_params"
        output_dir = os.path.join(output_base_dir, mi_type, distribution_name, *param_parts, size_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file_name = f"summary_{distribution_name}_{param_str}_{size_dir}_{mi_type}.txt"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        stats_df.rename(columns={"mean": f"mean_{mi_type}", "std": f"sigma_{mi_type}"}, inplace=True)
        stats_df.to_csv(output_file_path, index=False, sep='\t')
        
        logging.info(f"Statistics saved to {output_file_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Navigate and select mutual information files.")

    # Ask user for file type
    file_type_choice = input("Vuoi selezionare file log (_log.txt) o file normali (.txt)? [log/norm]: ").strip().lower()
    file_extension = "_log.txt" if file_type_choice == "log" else ".txt"
    
    if file_type_choice not in ["log", "norm"]:
        print("Scelta non valida. Uscita.")
        exit()
    
    selected_files = navigate_directories(
        start_path='../data/mi_numerical_results', 
        multi_select=True, 
        file_extension=file_extension
    )
    
    if not selected_files:
        print("No files selected. Exiting.")
        exit()
    
    save_mi_statistics(None, selected_files, key_col=None)
