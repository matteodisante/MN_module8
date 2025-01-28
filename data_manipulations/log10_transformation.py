import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from interface_utils import navigate_directories  # Ensure this utility is properly implemented and available


def apply_log10_transformation_to_file(file_path):
    """
    Reads a file and applies a logarithmic transformation (log base 10) to its numeric values.

    :param file_path: Path to the input file.
    :return: A list of transformed rows as strings.
    """
    transformed_rows = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Assuming the file has rows of space-separated values
                values = line.strip().split()
                transformed_values = []
                for value in values:
                    try:
                        value = float(value)
                        if value > 0:
                            transformed_values.append(str(np.log10(value)))  # log base 10
                        else:
                            # Replace non-positive values with log(1e-5)
                            transformed_values.append(str(-15.0))
                    except ValueError:
                        # Preserve non-numeric values (e.g., headers)
                        transformed_values.append(value)
                transformed_rows.append(" ".join(transformed_values))
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return transformed_rows


def save_transformed_data_to_subfolder(file_path, transformed_data):
    """
    Saves the transformed data to a dedicated subfolder named 'log_transformed' within the file's directory.

    :param file_path: Original file path for determining the output subfolder and filename.
    :param transformed_data: A list of strings containing transformed rows.
    """
    try:
        # Define the output subfolder within the same directory as the original file
        parent_dir = os.path.dirname(file_path)
        output_folder = os.path.join(parent_dir, "log_transformed")
        os.makedirs(output_folder, exist_ok=True)

        # Generate the new file path
        original_filename = os.path.basename(file_path)
        new_filename = original_filename.replace('.txt', '_log.txt')
        output_path = os.path.join(output_folder, new_filename)

        # Save the transformed data
        with open(output_path, 'w') as file:
            file.write("\n".join(transformed_data))
        print(f"Transformed file saved as: {output_path}")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")



if __name__ == "__main__":
    print("Log10 Transformation Tool\n")

    # Use navigate_directories to select files
    selected_files = navigate_directories(
        start_path=".", multi_select=True, file_extension=".txt"
    )

    # Check if files are valid
    if not selected_files:
        print("No files selected. Exiting.")
        exit()

    # Apply transformations
    for file_path in selected_files:
        if file_path.endswith(".txt"):
            print(f"Applying log10 transformation to {file_path}")
            transformed_data = apply_log10_transformation_to_file(file_path)
            save_transformed_data_to_subfolder(file_path, transformed_data)
        else:
            print(f"Skipping non-txt file: {file_path}")

    print("Log10 Transformation completed.")