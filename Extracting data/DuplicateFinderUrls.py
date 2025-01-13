import os
from collections import defaultdict

def find_and_remove_duplicates(file_path):
    """
    Reads the specified file, extracts lines starting with 'fixture:',
    prints out the fixtures that are duplicated,
    and modifies the file by deleting the first occurrence of each duplicate
    and everything after that line until a line starting with '#'.
    """
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    fixture_counts = defaultdict(int)
    fixtures_original_case = {}  # To store original case of fixtures

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Count fixtures and store original case
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.lower().startswith('fixture:'):
            fixture_lower = stripped_line.lower()
            fixture_counts[fixture_lower] += 1
            if fixture_lower not in fixtures_original_case:
                fixtures_original_case[fixture_lower] = stripped_line  # Store first occurrence with original case

    # Identify duplicates
    duplicates = [fixture for fixture, count in fixture_counts.items() if count > 1]

    if duplicates:
        print("Duplicate fixture lines found:")
        for fixture in duplicates:
            # Print the fixture in its original case
            print(fixtures_original_case[fixture])
    else:
        print("No duplicate fixture lines found.")
        return  # Exit since there are no duplicates to process

    if duplicates:
        # Create a set for faster lookup
        duplicates_set = set(duplicates)
        # Find indices to delete
        lines_to_delete = set()
        for fixture in duplicates_set:
            # Find first occurrence index
            for idx, line in enumerate(lines):
                stripped_line = line.strip().lower()
                if stripped_line == fixture:
                    # Start deleting from idx
                    delete_start = idx
                    # Find the index to stop deleting
                    delete_end = delete_start + 1
                    while delete_end < len(lines):
                        if lines[delete_end].strip().startswith('#'):
                            break
                        delete_end += 1
                    # Include the '#' line if found
                    if delete_end < len(lines) and lines[delete_end].strip().startswith('#'):
                        delete_end += 1
                    # Mark lines to delete
                    for i in range(delete_start, delete_end):
                        lines_to_delete.add(i)
                    break  # Only first occurrence

        # Create new list excluding lines to delete
        new_lines = [line for idx, line in enumerate(lines) if idx not in lines_to_delete]

        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(new_lines)
            print("File has been updated by removing duplicate fixtures and associated lines.")
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")

if __name__ == "__main__":
    # Specify the path to your file
    file_path = r"C:\Users\aaron\football-england-premier-league-2023-2024-Main.txt"
    find_and_remove_duplicates(file_path)
