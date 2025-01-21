import os

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
