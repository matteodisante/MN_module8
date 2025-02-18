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
    Automatically detects the key column:
      - For mi_binning files, expects the column 'bins_asked_per_axis'
      - For mi_1 and mi_sum files, expects 'k'     
    :param file_path: Path to the file.
    :return: Tuple (DataFrame, key_column) or (None, None) on error.
    """
    try:
        df = pd.read_csv(file_path, sep=r'\s+')
        # If the file contains the mi_binning column, use it.
        if "bins_asked_per_axis" in df.columns:
            key_col = "bins_asked_per_axis"
        elif "k" in df.columns and df.shape[1] > 1:
            key_col = "k"
        else:
            raise ValueError(f"Invalid file format in {file_path}. Expected one of 'bins_asked_per_axis', 'k', or 'bins_number'.")
        return df, key_col
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None, None

def calculate_mi_statistics(file_paths):
    """
    Calculates summary statistics (mean, standard deviation, and standard error)
    for MI values across multiple files (for mi_1 and mi_sum files).
    
    :param file_paths: List of file paths.
    :return: Tuple (statistics DataFrame, key_column)
    """
    combined_data = []
    key_col = None  # Will be set to 'k'     
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
        logging.error("No valid data found for MI statistics.")
        return None, key_col
    
    # Merge on the key column across all files
    merged_df = combined_data[0][[key_col]].copy()
    for idx, df in enumerate(combined_data):
        merged_df = merged_df.merge(df, on=key_col, how='outer', suffixes=('', f'_file{idx}'))
    
    mi_columns = [col for col in merged_df.columns if col.startswith("mi")]
    stats_df = merged_df[[key_col]].copy()
    
    # Compute mean, ignoring NaNs
    stats_df['mean'] = merged_df[mi_columns].mean(axis=1, skipna=True)
    
    # Compute unbiased standard deviation (sample std with ddof=1)
    stats_df['std'] = merged_df[mi_columns].std(axis=1, ddof=1)
    
    # Count valid (non-NaN) MI values for each row
    n_values = merged_df[mi_columns].count(axis=1)
    
    # Compute the standard error of the mean: std / sqrt(n)
    stats_df['std_mean'] = stats_df['std'] / np.sqrt(n_values)
    
    # Set rows with no valid MI values to NaN
    stats_df.loc[n_values == 0, ['mean', 'std', 'std_mean']] = np.nan
 
    return stats_df, key_col
    
    
    
    

def calculate_mi_binning_statistics(file_paths):
    """
    Calculates summary statistics for mi_binning files.
    The summary DataFrame will have the following columns:
      - bins_asked_per_axis: the bin value (from the first column)
      - MIN_nonzero_cells: minimum of non_empty_cells across files for each bin value
      - MEDIAN_nonzero_cells: median of non_empty_cells across files for each bin value
      - MAX_nonzero_cells: maximum of non_empty_cells across files for each bin value
      - mean_mi_binning: mean of mi_binning across files for each bin value
      - sigma_mi_binning: sample standard deviation (ddof=1) of mi_binning
      - std_mean_mi_binning: standard error of the mean (sigma/sqrt(n)) of mi_binning
    
    :param file_paths: List of file paths.
    :return: Summary DataFrame.
    """
    all_data = []
    for file_path in file_paths:
        df, key_col = read_mi_file(file_path)
        if df is not None:
            # Ensure the file is mi_binning by checking the key column.
            if key_col != "bins_asked_per_axis":
                logging.warning(f"File {file_path} does not appear to be a mi_binning file. Skipping.")
                continue
            all_data.append(df)
    if not all_data:
        logging.error("No valid mi_binning data found.")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Group by the bins_asked_per_axis value.
    grouped = combined_df.groupby("bins_asked_per_axis")
    
    summary = grouped.agg(
        MIN_nonzero_cells=pd.NamedAgg(column="non_empty_cells", aggfunc="min"),
        MEDIAN_nonzero_cells=pd.NamedAgg(column="non_empty_cells", aggfunc="median"),
        MAX_nonzero_cells=pd.NamedAgg(column="non_empty_cells", aggfunc="max"),
        mean_mi_binning=pd.NamedAgg(column="mi_binning", aggfunc="mean"),
        sigma_mi_binning=pd.NamedAgg(column="mi_binning", aggfunc=lambda x: np.std(x, ddof=1))
    ).reset_index()
    
    # Get count for standard error calculation.
    count_values = grouped["mi_binning"].count().reset_index(name="count")
    summary = summary.merge(count_values, on="bins_asked_per_axis")
    summary["std_mean_mi_binning"] = summary["sigma_mi_binning"] / np.sqrt(summary["count"])
    summary = summary.drop(columns=["count"])
    
    # Reorder columns
    summary = summary[["bins_asked_per_axis", "MIN_nonzero_cells", "MEDIAN_nonzero_cells", "MAX_nonzero_cells",
                         "mean_mi_binning", "sigma_mi_binning", "std_mean_mi_binning"]]
    return summary    
    
    

    
    


def save_mi_statistics(output_base_dir, selected_files):
    """
    Saves the summary statistics for mutual information files.
    This function groups files by their directory and then, based on the MI type
    (mi_1, mi_sum, or mi_binning), calls the appropriate summary calculation function.
    
    For mi_1 and mi_sum:
      - Reads files with key 'k' or 'bins_number'
      - Computes mean, std, and standard error of MI values.
    For mi_binning:
      - Reads files with key 'bins_asked_per_axis'
      - Computes summary statistics as described in calculate_mi_binning_statistics.
    
    The output file name is constructed based on parts of the directory structure.
    
    :param output_base_dir: Base directory for saving summary files.
    :param selected_files: List of file paths selected for processing.
    """
    grouped_files = {}
    for file_path in selected_files:
        group_key = os.path.dirname(file_path)
        if group_key not in grouped_files:
            grouped_files[group_key] = []
        grouped_files[group_key].append(file_path)
    
    for group_path, group_files in grouped_files.items():
        parts = group_path.split(os.sep)
        try:
            mi_index = parts.index("mi_numerical_results")
            mi_type = parts[mi_index + 1]  # e.g., "mi_1", "mi_sum", or "mi_binning"
            distribution_name = parts[mi_index + 2]  # Distribution name
            param_parts = parts[mi_index + 3:-1]      # All parameters before the size directory
            size_dir = parts[-1]                      # Size directory (e.g., size_*)
        except Exception as e:
            logging.warning(f"Error parsing group path {group_path}: {e}")
            mi_type = "unknown"
            distribution_name = "unknown"
            param_parts = []
            size_dir = "unknown"
        
        # Choose summary calculation function based on mi_type.
        if mi_type == "mi_binning":
            stats_df = calculate_mi_binning_statistics(group_files)
        else:
            stats_df, detected_key_col = calculate_mi_statistics(group_files)
        
        if stats_df is None:
            continue
        
        param_str = "_".join(param_parts) if param_parts else "no_params"
        output_dir = os.path.join(output_base_dir, mi_type, distribution_name, *param_parts, size_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file_name = f"summary_{distribution_name}_{param_str}_{size_dir}_{mi_type}.txt"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        if mi_type != "mi_binning":
            # For mi_1 and mi_sum, rename columns to include the MI type in their names.
            stats_df.rename(columns={"mean": f"mean_{mi_type}", "std": f"sigma_{mi_type}"}, inplace=True)
        # Save the summary as a tab-separated file.
        stats_df.to_csv(output_file_path, index=False, sep='\t')
        logging.info(f"Statistics saved to {output_file_path}")
        
        


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Navigate and select mutual information files.")
    
    # Ask the user to select log files (_log.txt) or normal files (.txt)
    file_type_choice = input("Do you want to select log files (_log.txt) or normal files (.txt)? [log/norm]: ").strip().lower()
    file_extension = "_log.txt" if file_type_choice == "log" else ".txt"
    
    if file_type_choice not in ["log", "norm"]:
        print("Invalid choice. Exiting.")
        exit()
    
    selected_files = navigate_directories(
        start_path='../data/mi_numerical_results',
        multi_select=True,
        file_extension=file_extension
    )
    
    if not selected_files:
        print("No files selected. Exiting.")
        exit()
    
    # Define the base directory where summary files will be saved.
    output_base_dir = "../data/mi_summaries"
    save_mi_statistics(output_base_dir, selected_files)
