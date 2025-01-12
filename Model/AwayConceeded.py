
import pandas as pd

# Read the Excel file
df = pd.read_csv('C:/Users/aaron/Model/data.csv')

# Assuming your Excel file has a 'Week' and 'Home Goals Against' column
week_column = 'Week'
team_column = 'AwayTeam'
goals_column = 'FTHG'

goals_data = {}

unique_weeks = sorted(df[week_column].unique())

for index, row in df.iterrows():
    week = row[week_column]
    team = row[team_column]
    goals = row[goals_column]

    if pd.notna(week) and pd.notna(team) and pd.notna(goals):
        if team in goals_data:
            goals_data[team][week] = goals
        else:
            goals_data[team] = {week: goals}

# Create a DataFrame from the dictionary with sorted weeks
df = pd.DataFrame.from_dict(goals_data, orient='index')
df = df[unique_weeks]  # Reorder columns to be in chronological order

# Fill NaN values with empty cells
df = df.fillna('')

df.to_excel('AwayConceeded.xlsx')