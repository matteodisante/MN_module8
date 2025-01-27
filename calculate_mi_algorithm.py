import os
from pathos.multiprocessing import ProcessingPool as Pool  # Usa pathos per evitare problemi di pickle


from utils.mutual_information_utils import process_and_save_mi_table, aggregate_mi_results
from utils.interface_utils import navigate_directories, parse_k_values, setup_logging
from utils.io_utils import save_transformed_file
from utils.math_utils import transform_to_bilog_scale

from core.mutual_information_1 import mutual_information_1
from core.mutual_information_1_entropies_sum import mutual_information_1_entropies_sum
from core.mutual_information_binning import mutual_information_binning



def process_single_file_wrapper(args):
    file_path, selected_functions, output_dir, k_values, num_bins = args
    for mi_func in selected_functions:
        process_and_save_mi_table(
            file_path=file_path,
            output_dir=output_dir,
            k_values=k_values,
            mi_estimate_function=mi_func,
            num_bins=num_bins if mi_func == mutual_information_binning else None,  # Only pass num_bins for binning
        )


def main():
    print("Welcome to the Mutual Information Analysis Tool\n")

    # Call the setup function at the beginning of the main script
    setup_logging() 

    # Get input directory
    input_dir = input("Enter the path to the directory containing the data files: ").strip()
    if not os.path.isdir(input_dir):
        print("Invalid input directory. Exiting.")
        return

    # Get output directory
    output_dir = input("Enter the path to the directory where results should be saved: ").strip()
    if not os.path.exists(output_dir):
        create_dir = input(f"The output directory '{output_dir}' does not exist. Do you want to create it? [y/n]: ").strip().lower()
        if create_dir == 'y':
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory '{output_dir}' created.")
        else:
            print("Output directory not created. Exiting.")
            return
    else:
        print(f"Using existing output directory: '{output_dir}'.")

    # Get k values
    print("Specify k values as a list of ranges and/or individual values (e.g., 1-15,17,30-35,45):")
    k_input = input("Enter k values: ").strip()
    k_values = parse_k_values(k_input)

    if not k_values:
        print("No valid k values provided. Exiting.")
        return

    # Get mutual information estimation functions
    mi_functions_map = {
        "1": mutual_information_1,
        "2": mutual_information_1_entropies_sum,
        "3": mutual_information_binning,
    }
    print("Select the MI estimation functions to use:")
    print("1: mutual_information_1")
    print("2: mutual_information_1_entropies_sum")
    print("3: mutual_information_binning")
    selected_functions = input("Enter the numbers corresponding to the desired functions (comma-separated): ").strip()
    try:
        selected_functions = [mi_functions_map[num.strip()] for num in selected_functions.split(",") if num.strip() in mi_functions_map]
    except KeyError:
        print("Invalid selection of MI functions. Exiting.")
        return

    if not selected_functions:
        print("No valid MI estimation functions selected. Exiting.")
        return

    # If mutual_information_binningadaptive is selected, ask for the number of bins
    num_bins = None
    if mutual_information_binning in selected_functions:
        try:
            num_bins = int(input("Enter the number of bins for mutual_information_binning: ").strip())
            if num_bins <= 0:
                raise ValueError("Number of bins must be greater than 0.")
        except ValueError as e:
            print(f"Invalid input for number of bins: {e}. Exiting.")
            return

    print(f"selectedddddddddddddddd: mi_estimate_function: {selected_functions}")
    # Use navigate_directories to select files or directories
    selected_files = navigate_directories(start_path=input_dir, multi_select=True, file_extension=".txt")

    if not selected_files:
        print("No files selected. Exiting.")
        return

    # Chiedere una sola volta per trasformare tutti i file
    apply_log = input("Do you want to apply a logarithmic (base 10) transformation to all the .txt files in the selected directory? [y/n]: ").strip().lower()
    if apply_log == 'y':
        for file in selected_files:
            if file.endswith('.txt'):
                print(f"Applying log10 transformation to {file}")
                transformed_data = transform_to_bilog_scale(file)
                save_file = input(f"Do you want to overwrite the original file {file} or save it as a new file? [overwrite/new]: ").strip().lower()
                if save_file == 'new':
                    new_file_path = file.replace('.txt', '_log10.txt')
                    save_transformed_file(new_file_path, transformed_data)
                    print(f"Saved transformed file as {new_file_path}")
                else:
                    save_transformed_file(file, transformed_data)
                    print(f"Overwritten the original file {file}")
    else:
        print("Skipping log10 transformation.")
        
        # Chiedere se l'utente vuole applicare la trasformazione a singoli file o a più file
        transform_files = input("Do you want to apply the transformation to a single file or multiple files? [single/multiple]: ").strip().lower()
        
        if transform_files == 'single':
            # Mostrare i file tra cui l'utente può scegliere
            print("Select a file to apply the transformation:")
            for i, file in enumerate(selected_files):
                if file.endswith('.txt'):
                    print(f"{i + 1}: {file}")
            file_index = int(input("Enter the number corresponding to the file: ").strip()) - 1
            file_to_transform = selected_files[file_index]
            print(f"Applying log10 transformation to {file_to_transform}")
            transformed_data = transform_to_bilog_scale(file_to_transform)
            save_file = input(f"Do you want to overwrite the original file {file_to_transform} or save it as a new file? [overwrite/new]: ").strip().lower()
            if save_file == 'new':
                new_file_path = file_to_transform.replace('.txt', '_log10.txt')
                save_transformed_file(new_file_path, transformed_data)
                print(f"Saved transformed file as {new_file_path}")
            else:
                save_transformed_file(file_to_transform, transformed_data)
                print(f"Overwritten the original file {file_to_transform}")
        
        elif transform_files == 'multiple':
            # Mostrare i file tra cui l'utente può scegliere
            print("Select multiple files to apply the transformation:")
            for i, file in enumerate(selected_files):
                if file.endswith('.txt'):
                    print(f"{i + 1}: {file}")
            selected_file_indices = input("Enter the numbers corresponding to the files (comma-separated): ").strip().split(",")
            for index_str in selected_file_indices:
                try:
                    file_index = int(index_str.strip()) - 1
                    file_to_transform = selected_files[file_index]
                    print(f"Applying log10 transformation to {file_to_transform}")
                    transformed_data = transform_to_bilog_scale(file_to_transform)
                    save_file = input(f"Do you want to overwrite the original file {file_to_transform} or save it as a new file? [overwrite/new]: ").strip().lower()
                    if save_file == 'new':
                        new_file_path = file_to_transform.replace('.txt', '_log10.txt')
                        save_transformed_file(new_file_path, transformed_data)
                        print(f"Saved transformed file as {new_file_path}")
                    else:
                        save_transformed_file(file_to_transform, transformed_data)
                        print(f"Overwritten the original file {file_to_transform}")
                except ValueError:
                    print(f"Invalid input for file index: {index_str}. Skipping.")

    # Prepare arguments for each file
    args_list = [
    (file_path, selected_functions, output_dir, k_values, num_bins)
    for file_path in selected_files
    ]

    # Process files in parallel
    with Pool() as pool:
        pool.map(process_single_file_wrapper, args_list)


    # Aggregate results
    print("Aggregating results...")
    aggregate_mi_results(
        base_dir=output_dir,
        output_dir=output_dir,
        mi_estimate_functions=selected_functions,
        k_values=k_values,  # Pass k_values for filtering
    )

    print("Mutual Information Analysis completed successfully!")

if __name__ == "__main__":
    main()