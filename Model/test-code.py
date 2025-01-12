import pandas as pd
import os

# Load your existing Excel file
sheet_name = 'data'

# Read data from the CSV file
df = pd.read_csv('C:/Users/aaron/Model/data.csv')

# Group data by week
grouped = df.groupby('Week')

# Create a new Excel file for each week
for week, group in grouped:
    week_file = f'Week_{week}.xlsx'
    
    # Create a new DataFrame for the current week
    week_df = pd.DataFrame()
    
    # Check if 'HomeTeam' column exists in the group
    if 'HomeTeam' in group.columns:
        week_df['HomeTeam'] = group['HomeTeam']
        week_df['FTHG'] = group['FTHG']
    else:
        week_df['HomeTeam'] = '-'
        week_df['FTHG'] = '-'
    
    # Check if 'AwayTeam' column exists in the group
    if 'AwayTeam' in group.columns:
        week_df['AwayTeam'] = group['AwayTeam']
        week_df['FTAG'] = group['FTAG']
    else:
        week_df['AwayTeam'] = '-'
        week_df['FTAG'] = '-'
    
    # Save the current week's data to a new Excel file
    week_df.to_excel(week_file, index=False)




