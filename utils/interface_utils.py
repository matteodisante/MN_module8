import os
import logging
import datetime

def navigate_directories(start_path=".", multi_select=False, file_extension=".bin"):
    """
    Allows users to navigate directories and select files or folders.

    Parameters:
        start_path (str): The starting directory for navigation.
        multi_select (bool): If True, allows selection of multiple files or folders.
        file_extension (str): File extension to filter files (e.g., '.bin').

    Returns:
        list[str]: List of selected file or folder paths.
    """
    current_dir = os.path.abspath(start_path)
    selected_paths = []

    while True:
        # Filter and sort contents (exclude hidden files/directories)
        contents = [item for item in sorted(os.listdir(current_dir)) if not item.startswith('.')]
        directories = [item for item in contents if os.path.isdir(os.path.join(current_dir, item))]
        files = [item for item in contents if os.path.isfile(os.path.join(current_dir, item)) and item.endswith(file_extension)]

        # Display current directory and contents
        print(f"\nCurrent directory: {current_dir}")
        print("Directories:")
        for idx, directory in enumerate(directories, 1):
            print(f"  D{idx}. {directory}")
        print("Files:")
        for idx, file in enumerate(files, len(directories) + 1):
            print(f"  F{idx}. {file}")

        # Display actions
        print("\nActions: '..' (up), '.' (list), 'done' (finish selection), 'exit' (quit), 'all' (select all)")

        user_input = input("Enter your choice (number, '..', '.', 'done', 'exit', 'all'): ").strip()

        if user_input == "exit":
            print("[INFO] Exiting navigation.")
            exit(0)
        elif user_input == "done":
            if selected_paths:
                break
            print("[INFO] No selections made. Returning to navigation.")
        elif user_input == "..":
            # Move up one directory
            current_dir = os.path.dirname(current_dir)
        elif user_input == ".":
            # Refresh listing
            continue
        elif user_input == "all":
            # Select all visible items
            for item in contents:
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path):
                    # Recursively collect files from directory
                    for root, _, files_in_dir in os.walk(item_path):
                        selected_paths.extend([os.path.join(root, f) for f in files_in_dir if f.endswith(file_extension)])
                elif os.path.isfile(item_path) and item_path.endswith(file_extension):
                    selected_paths.append(item_path)
            print(f"[INFO] Selected all items: {selected_paths}")
            break
        elif user_input.startswith("D") and user_input[1:].isdigit():
            # Handle directory selection
            choice_idx = int(user_input[1:]) - 1
            if 0 <= choice_idx < len(directories):
                current_dir = os.path.join(current_dir, directories[choice_idx])
            else:
                print("[ERROR] Invalid directory index.")
        elif user_input.startswith("F") and user_input[1:].isdigit():
            # Handle file selection
            choice_idx = int(user_input[1:]) - 1 - len(directories)
            if 0 <= choice_idx < len(files):
                selected_path = os.path.join(current_dir, files[choice_idx])
                if multi_select:
                    if selected_path not in selected_paths:
                        selected_paths.append(selected_path)
                        print(f"[INFO] Added to selection: {selected_path}")
                    else:
                        print("[INFO] File already selected.")
                else:
                    return [selected_path]
            else:
                print("[ERROR] Invalid file index.")
        else:
            print("[ERROR] Invalid command.")

    return selected_paths



def parse_k_values(k_input):
    """
    Parse a string input for k values and ranges into a sorted list of unique integers.
    Input format example: "1-15,17,30-35,45"

    :param k_input: String containing k ranges and values.
    :return: Sorted list of unique k values with overlapping values removed.
    """
    k_values = set()
    try:
        ranges = []
        singles = set()

        parts = k_input.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:  # Range
                k_min, k_max = map(int, part.split('-'))
                if k_min <= 0 or k_max <= 0 or k_min > k_max:
                    raise ValueError(f"Invalid range: {part}")
                ranges.append(range(k_min, k_max + 1))
            else:  # Single value
                k = int(part)
                if k <= 0:
                    raise ValueError(f"Invalid value: {part}")
                singles.add(k)

        # Expand ranges and remove overlaps
        for r in ranges:
            k_values.update(r)
        k_values.update(singles)

        return sorted(k_values)
    except Exception as e:
        raise ValueError(f"Error parsing k values: {e}")



# Logging setup function
def setup_logger():
    """Set up the logging configuration and create a log file dynamically."""
    log_dir = "logging"
    os.makedirs(log_dir, exist_ok=True)  # Ensure logging directory exists

    # Nome file log dinamico basato sullo script attuale
    script_name = os.path.basename(__file__).replace(".py", "")
    log_filename = f"{script_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.DEBUG,  # DEBUG mostra tutti i messaggi
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),  # Scrive su file
            logging.StreamHandler()  # Stampa in console
        ]
    )
    logging.info("Logging started. File: %s", log_path)
