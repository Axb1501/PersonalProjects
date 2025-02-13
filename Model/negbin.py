import pandas as pd
import numpy as np
import math
from scipy.stats import nbinom

# Read fixtures
fixtures_df = pd.read_csv('C:/Users/aaron/Model/data.csv')

# Read home and away goals and home and away goals conceded
home_goals_df = pd.read_excel('C:/Users/aaron/Model/HomeGoals.xlsx', index_col=0)
home_conceded_df = pd.read_excel('C:/Users/aaron/Model/HomeConceeded.xlsx', index_col=0)
away_goals_df = pd.read_excel('C:/Users/aaron/Model/AwayGoals.xlsx', index_col=0)
away_conceded_df = pd.read_excel('C:/Users/aaron/Model/AwayConceeded.xlsx', index_col=0)

# Assuming you have a list of possible scorelines
scorelines = [(i, j) for i in range(5) for j in range(5)]



# Function to predict the winner based on probabilities
def predict_winner(probabilities):
    max_prob = 0
    winner = 'D'
    for scoreline, probability in probabilities.items():
        if probability > max_prob:
            max_prob = probability
            winner = 'H' if scoreline[0] > scoreline[1] else 'A' if scoreline[1] > scoreline[0] else 'D'
    return winner

# Calculate accuracy for each week
weekly_accuracies = []
weekly_over_2_5_goals_accuracies = []
total_correct_prediction = []
total_prediction = []

for week in fixtures_df['Week'].unique():
    correct_predictions = 0
    correct_over_2_5_goals_predictions = 0
    total_match_predictions = 0
    total_score_predictions = 0

    week_fixtures = fixtures_df[fixtures_df['Week'] == week]

    if week < 6:
        continue

    # Calculate probabilities for each scoreline
    for index, fixture in week_fixtures.iterrows():
        home_team = fixture['HomeTeam']
        away_team = fixture['AwayTeam']
        fixture_week = fixture['Week']
        actual_result = fixture['HTR']

        # Filter data up to the current fixture week for the home team
        home_goals_up_to_week = home_goals_df.loc[home_team, :fixture_week -1 ].values
        home_conceded_up_to_week = home_conceded_df.loc[home_team, :fixture_week -1 ].values

        # Filter data up to the current fixture week for the away team
        away_goals_up_to_week = away_goals_df.loc[away_team, :fixture_week -1 ].values
        away_conceded_up_to_week = away_conceded_df.loc[away_team, :fixture_week -1 ].values

        # Calculate the average goals for the league up to the current fixture week
        league_home_goals = np.nanmean(home_goals_df.loc[:, :fixture_week -1 ].values)
        league_home_conceded = np.nanmean(home_conceded_df.loc[:, :fixture_week -1 ].values)

        league_away_goals = np.nanmean(away_goals_df.loc[:, :fixture_week -1 ].values)
        league_away_conceded = np.nanmean(away_conceded_df.loc[:, :fixture_week -1 ].values)

        # Get the average home goals and home goals conceded for the home team
        home_mu_goals = np.nanmean(home_goals_up_to_week)
        home_mu_conceded = np.nanmean(home_conceded_up_to_week)

        # Get the average away goals and away goals conceded for the away team
        away_mu_goals = np.nanmean(away_goals_up_to_week)
        away_mu_conceded = np.nanmean(away_conceded_up_to_week)

        Home_attack_rating = home_mu_goals / league_home_goals
        Home_defence_rating = home_mu_conceded / league_home_conceded
        Away_attack_rating = away_mu_goals / league_away_goals
        Away_defence_rating = away_mu_conceded / league_away_conceded

        # Calculate probabilities for each scoreline with adjusted team strengths
        probabilities = {}

        home_win_probability = 0
        away_win_probability = 0
        draw_probability = 0
        over_2_5_goals_probability = 0

        variance_home_goals = np.nanvar(home_goals_df.values)
        variance_away_goals = np.nanvar(away_goals_df.values)
        variance_goals = np.mean([variance_home_goals, variance_away_goals])

        r_value = 0.7 * variance_goals

        
        for scoreline in scorelines:
            home_goals, away_goals = scoreline

            home_p = nbinom.pmf(home_goals, r_value, Home_attack_rating * Away_defence_rating * league_home_goals)
            away_p = nbinom.pmf(away_goals, r_value, Away_attack_rating * Home_defence_rating * league_away_goals)

            probability = home_p * away_p
            probabilities[scoreline] = probability
            print(f"Probability of Scoreline {scoreline}: {probability:.4f}")


        # Ensure the sum of probabilities is 1
            total_probability = sum(probabilities.values())
            normalized_probabilities = {scoreline: prob / total_probability for scoreline, prob in probabilities.items()}

        
        # Update individual probabilities
            for scoreline, probability in normalized_probabilities.items():
                home_goals, away_goals = scoreline
                if home_goals > away_goals:
                    home_win_probability += probability
                elif away_goals > home_goals:
                    away_win_probability += probability
                else:
                    draw_probability += probability        

            

            # Update over 2.5 goals probability
                    over_2_5_goals_probability = sum(prob for scoreline, prob in normalized_probabilities.items() if sum(scoreline) > 2)

                

            print(f"Home Teams: {home_team}, Away Teams: {away_team} Probability: {home_win_probability:.4f}, Home Win Probability: {away_win_probability:.4f}")

        # Predict the winner based on probabilities
        # Check if the home team winning probability is at least 10% higher than each different probability
            if home_win_probability > away_win_probability:
        # Predict the winner based on probabilities
                predicted_result = 'H'
            elif away_win_probability > home_win_probability:
                predicted_result = 'A'
            else:
                predicted_result = 'D'

        # Check if the prediction matches the actual result
            

            if  predicted_result == actual_result:
                correct_predictions += 1
                total_match_predictions += 1
            
            elif actual_result == 'D':
                correct_predictions += 1
                total_match_predictions += 1
        
            else:
                total_match_predictions += 1

        

        # Check if the prediction for over 2.5 goals is correct
            if over_2_5_goals_probability > 0.5:  # Adjust this threshold as needed
                predicted_over_2_5_goals = 'O'
            else:
                predicted_over_2_5_goals = 'U'

            actual_over_2_5_goals = 'O' if fixture['FTHG'] + fixture['FTAG'] > 2.5 else 'U'

            if predicted_over_2_5_goals == actual_over_2_5_goals:
                correct_over_2_5_goals_predictions += 1

        
            total_score_predictions += 1

        

        if total_match_predictions > 0:
            accuracy = correct_predictions / total_match_predictions
            weekly_accuracies.append((week, accuracy))
        else:
            accuracy = 0 
            total_correct_prediction.append(correct_predictions)
            total_prediction.append(total_match_predictions)
    

    # Calculate accuracy for predicting over 2.5 goals for the week
        if total_score_predictions > 0:
            accuracy_over_2_5_goals = correct_over_2_5_goals_predictions / total_score_predictions
            weekly_over_2_5_goals_accuracies.append((week, accuracy_over_2_5_goals))
        else:
            accuracy_over_2_5_goals = 0

    # Calculate overall accuracy for match winner
overall_accuracy = sum(accuracy for _, accuracy in weekly_accuracies) / len(weekly_accuracies)

overall_correct_predictions = sum(total_correct_prediction)

# Calculate overall accuracy for over 2.5 goals
overall_over_2_5_goals_accuracy = sum(accuracy_over_2_5_goals for _, accuracy_over_2_5_goals in weekly_over_2_5_goals_accuracies) / len(weekly_over_2_5_goals_accuracies)


# Print overall accuracies
print(f"Overall Match Winner Accuracy: {overall_accuracy * 100:.2f}%")
print(f"Overall Over 2.5 Goals Accuracy: {overall_over_2_5_goals_accuracy * 100:.2f}%")
print(overall_correct_predictions)
print(total_prediction)
       
