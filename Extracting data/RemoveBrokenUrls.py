import re
from pathlib import Path
from LLMscraping import Crawler  # Ensure this module is correctly installed and accessible
import asyncio

# Path to the Redo.txt file
redo_file_path = Path("Redo.txt")

# Check if Redo.txt exists
if not redo_file_path.is_file():
    print(f"Redo.txt not found at {redo_file_path}")
    exit(1)

# Read all lines from Redo.txt
try:
    with redo_file_path.open("r", encoding="utf-8") as redo_file:
        redo_lines = redo_file.readlines()
except Exception as e:
    print(f"Failed to read Redo.txt: {e}")
    exit(1)

# List to store all fixture tuples
all_fixtures = []

# Process each line from Redo.txt
for line_number, line in enumerate(redo_lines, start=1):
    # Strip whitespace and ignore empty lines
    line = line.strip()
    if not line:
        continue

    # Example line format:
    # ('https://www.betexplorer.com/football/england/premier-league-2011-2012/liverpool-swansea/6i3mNChk/#1x2', 'Tab: 1X2')
    # Extract the URL part using regex
    url_match = re.search(r"'(https://[^'#]+)", line)
    if not url_match:
        print(f"Line {line_number}: Failed to extract URL.")
        continue

    cleaned_url = url_match.group(1)
    tab_label = "Main"  # As per your requirement

    # Create the tuple and add to the list
    fixture_tuple = (tab_label, cleaned_url)
    all_fixtures.append(fixture_tuple)
    print(f"Line {line_number}: Fixture Tuple created: {fixture_tuple}")

# If no valid fixtures found, exit
if not all_fixtures:
    print("No valid fixtures found in Redo.txt.")
    exit(0)

# Now, process each fixture tuple
# Function to process a single fixture
def process_fixture(fixture_tuple):
    tab_label, url = fixture_tuple

    # Define the regex pattern for extracting the desired part
    pattern = r"football/[a-z-]+/[a-z-]+-\d{4}-\d{4}"

    # Search for the pattern in the URL
    match = re.search(pattern, url)

    if not match:
        print(f"URL does not match the required pattern: {url}")
        return None

    # Extract the matched text
    extracted_text = match.group()

    # Construct the file path using pathlib for better portability
    base_path = Path("C:/Users/aaron")
    file_name = extracted_text.replace("/", "-") + "-main.txt"
    file_path = base_path / file_name

    # Print the file path
    print(f"Processing file: {file_path}")

    # Initialize error counter and error details
    error_counter = 0
    error_details = []

    # Read fixture lines from the generated file path
    try:
        with file_path.open("r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Extract team names
    teams = set()
    for line_content in lines:
        fixture_match = re.search(r"Fixture:\s*(.+?)\s+vs\s+(.+)", line_content, re.IGNORECASE)
        if fixture_match:
            teams.add(fixture_match.group(1).strip().lower())
            teams.add(fixture_match.group(2).strip().lower())

    # Convert set to list and print
    team_list = list(teams)
    print(f"Unique team names in {file_name}: {team_list}")
    print(f"Total number of unique teams: {len(team_list)}")

    # Extract fixture name from URL with enhanced regex pattern
    url_fixture_pattern = r"/football/(?:[^/]+/)*([a-z-]+)/"
    url_fixture_match = re.search(url_fixture_pattern, url)

    if not url_fixture_match:
        print(f"Fixture string not found in the URL: {url}")
        return None

    # Extract the entire fixture string without trailing slash
    fixture_str = url_fixture_match.group(1)  # e.g., "liverpool-swansea"

    print(f"Extracted fixture string: {fixture_str}")

    # Split the fixture string into parts
    parts = fixture_str.split('-')

    # Generate all possible split options
    options = []
    for i in range(1, len(parts)):
        team1 = ' '.join(parts[:i]).lower()
        team2 = ' '.join(parts[i:]).lower()
        option = f"{team1} vs {team2}"
        options.append(option)

    print(f"Possible fixture options: {options}")

    # Store valid fixture names
    valid_fixture_names = []

    # Check which options are valid based on team_list
    for option in options:
        # Split the option into potential team names
        potential_teams = option.split(" vs ")
        # Normalize team names for comparison
        potential_teams = [team.strip().lower() for team in potential_teams]
        # Check if both teams are in the team_list
        if all(team in team_list for team in potential_teams):
            print(f"Valid fixture name found: {option}")
            valid_fixture_names.append(option)

    # If there are valid fixture names, proceed to process related text
    if valid_fixture_names:
        # Initialize new_lines list
        new_lines = []
        idx = 0
        total_lines = len(lines)

        while idx < total_lines:
            line_content = lines[idx]
            # Check if current line matches any valid fixture
            fixture_found = False
            for valid_fixture in valid_fixture_names:
                fixture_search_str = f"Fixture: {valid_fixture}"
                if fixture_search_str.lower() in line_content.lower():
                    fixture_found = True
                    print(f"Found fixture '{valid_fixture}' at line {idx + 1}")
                    start_idx = idx
                    # Search for end marker
                    end_idx = None
                    for search_idx in range(idx + 1, total_lines):
                        if lines[search_idx].strip().startswith("#"):
                            end_idx = search_idx
                            print(f"End marker '#' found at line {search_idx + 1}")
                            break
                    if end_idx is not None:
                        print(f"Deleting lines {start_idx + 1} to {end_idx + 1}")
                        # Skip lines from start_idx to end_idx (inclusive)
                        idx = end_idx + 1
                    else:
                        # End marker not found; increment error counter and store details
                        error_counter += 1
                        error_info = {
                            'fixture': valid_fixture,
                            'start_line': start_idx + 1,
                            'message': "End marker '#' not found. Fixture details extend to end of file."
                        }
                        error_details.append(error_info)
                        print(f"Error: {error_info['message']} (Fixture '{valid_fixture}' at line {start_idx + 1})")
                        # Do not delete lines; include them
                        new_lines.append(line_content)
                        idx += 1
                    break  # Move to next line after processing fixture
            if not fixture_found:
                # Not a fixture line; include in new_lines
                new_lines.append(line_content)
                idx += 1

        # Write the updated lines back to the file
        try:
            with file_path.open("w", encoding="utf-8") as file:
                file.writelines(new_lines)
            print(f"Deletion of fixture details in {file_name} completed successfully.")
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            return None

        # After processing all fixtures, print the error summary if any
        if error_counter > 0:
            print(f"\nTotal Errors Encountered in {file_name}: {error_counter}")
            print("Error Details:")
            for error in error_details:
                print(f" - Fixture: {error['fixture']}, Start Line: {error['start_line']}, Message: {error['message']}")
        else:
            print("\nNo errors encountered. All fixtures have proper end markers.")

        # **Tuple Extraction and Crawler Integration**
        # Desired tuple: ("Main", "https://www.betexplorer.com/football/england/premier-league-2011-2012/liverpool-swansea/6i3mNChk")

        # Step 1: Extract the URL without the fragment (remove everything after '#')
        # Already extracted as 'cleaned_url'

        # Step 2: Set tab_label to "Main" directly
        # Already set as 'tab_label = "Main"'

        # Step 3: Create the tuple
        # Already created as 'fixture_tuple = (tab_label, cleaned_url)'

        # (Since we're processing one fixture at a time, this function returns the tuple)
        return fixture_tuple

    else:
        print(f"No valid fixture names found based on the team list in {file_name}.")
        return None

# List to hold all fixture tuples to be passed to Crawler
fixtures_to_crawl = []

# Process each fixture tuple and collect valid ones
for fixture in all_fixtures:
    fixture_tuple = process_fixture(fixture)
    if fixture_tuple:
        fixtures_to_crawl.append(fixture_tuple)

# If there are fixtures to crawl, invoke the Crawler
if fixtures_to_crawl:
    print("\nInvoking Crawler with the following fixtures:")
    for ft in fixtures_to_crawl:
        print(ft)

    try:
        crawler = Crawler(fixtures_to_crawl)  # Pass a list of tuples
        asyncio.run(crawler.run())  # Run the asynchronous crawler
        print("Crawler has been invoked successfully.")
    except Exception as e:
        print(f"An error occurred while invoking the Crawler: {e}")
else:
    print("No fixtures to crawl.")
