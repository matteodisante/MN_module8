from core.mutual_information_1 import *
from real_data_processing import (import_raw_data, real_data_processing_1, 
                                  real_data_processing_2, get_valid_integer)

def get_user_parameters3():
    """
    Prompt the user to input parameters for the process with input validation.

    Returns:
        tuple: (d, k) with valid integer values.
    """
    # Default values
    default_k = 1

    # Prompt for 'h' with options and validation
    while True:
        print("\nChoose the overlap type:")
        print("1 - Half overlapping")
        print("2 - No overlaps")
        d_choice = input("Enter your choice (1 or 2, default 1): ").strip()
        if not d_choice:
            d = 2
            break
        elif d_choice == "1":
            d = 2
            break
        elif d_choice == "2":
            d = 1
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # Prompt for 'k' with validation
    k = get_valid_integer("Enter the value for 'k'", default_k, min_value=1)

    return d, k


if __name__ == "__main__":
    # directory path
    directory_path = "data/real_data/raw_data"

    # import raw data
    data_dict_A, data_dict_C = import_raw_data(directory_path)
    raw_data = {'A': data_dict_A, 'C': data_dict_C}

    # choose the values of n to explore
    n_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    processing_choice = ['Processig A and C data', 'Processing every file A and C']

    while True:
        # Interactive selection of processing choice
        print("\nAvailable choices:")
        for i, choice_name in enumerate(processing_choice, start=1):
            print(f"{i}. {choice_name}")

        selected_choice = None
        while selected_choice is None:
            try:
                user_input = input("Select a choice by number: ").strip()
                selected_index = int(user_input) - 1
                if 0 <= selected_index < len(processing_choice):
                    selected_choice = processing_choice[selected_index]
                else:
                    print("Invalid selection. Please choose a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        print(f"\nSelected choice: {selected_choice}")

        if selected_index == 0:
                
            d, k = get_user_parameters3()
            for n in n_list:
                h = n // d
                print(f"\nParameters: n={n}, h={h}, k={k}")
                real_data_processing_1(raw_data['A'], 'A', n, h, k)
                real_data_processing_1(raw_data['C'], 'C', n, h, k)


        elif selected_index == 1:

            d, k = get_user_parameters3()
            for n in n_list:
                h = n // d
                print(f"\nParameters: n={n}, h={h}, k={k}")
                real_data_processing_2(raw_data['A'], 'A', n, h, k)
                real_data_processing_2(raw_data['C'], 'C', n, h, k)


        # Ask if the user wants to process more data or exit
        continue_choice = None
        while continue_choice not in ['y', 'n']:
            continue_choice = input("\nDo you want to process more data? (y/n): ").strip().lower()
        if continue_choice == 'n':
            print("Exiting program. Goodbye!")
            break