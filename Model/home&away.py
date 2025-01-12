import pandas as pd
import os
import re
import openpyxl

# Directory where the week's Excel files are stored
directory = 'c:/Users/aaron/Model'

# Get a list of all Excel files in the directory
week_files = [file for file in os.listdir(directory) if file.endswith('.xlsx')]

# Initialize a dictionary to store team data
team_home_data = {}

pattern = r'Week_(\d+)\.xlsx'

# Iterate through each week's Excel file
for week_file in week_files:
    # Extract the week number from the file name
    match = re.search(pattern, week_file)
    if match:
        week_played = int(match.group(1))  # Convert the extracted value to an integer
    else:
        continue

# Iterate through each week's Excel file
for week_file in week_files:
    # Load the week's data
    df = pd.read_excel(os.path.join(directory, week_file), engine='openpyxl')
    
    # Iterate through each row in the week's data
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        goals_scored_home = row['FTHG']
        goals_conceded_home = row['FTAG']
        
        # Skip rows with missing or inconsistent data
        if pd.isna(home_team) or pd.isna(week_played) or pd.isna(goals_scored_home) or pd.isna(goals_conceded_home):
            continue
        
        # Initialize or update data for the home team
        if home_team not in team_home_data:
            team_home_data[home_team] = {'Week Played': [], 'Goals Scored at Home': [], 'Goals Conceded at Home': []}
        team_home_data[home_team]['Week Played'].append(week_played)
        team_home_data[home_team]['Goals Scored at Home'].append(goals_scored_home)
        team_home_data[home_team]['Goals Conceded at Home'].append(goals_conceded_home)

# Create a separate Excel file for each home team
for team, data in team_home_data.items():
    home_file = f'{team} - Home.xlsx'
    home_df = pd.DataFrame(data)

    with pd.ExcelWriter(home_file, engine='openpyxl') as writer:
        writer.book = openpyxl.Workbook()
        writer.sheets = {ws.title: ws for ws in writer.book.worksheets}
        writer.book.active.title = 'VisibleSheet'


        
 
    
    # Save data to Excel
        home_df.to_excel(writer, sheet_name='Home Data', index=False)

# The Excel files with home data for each team will be saved in the current directory
