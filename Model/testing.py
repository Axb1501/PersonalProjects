import pandas as pd
import numpy as np
import math
from scipy.stats import nbinom
from sklearn.model_selection import KFold

# Read fixtures
fixtures_df = pd.read_csv('C:/Users/aaron/Model/data.csv')

# Read home and away goals and home and away goals conceded
home_goals_df = pd.read_excel('HomeGoals.xlsx', index_col=0)
home_conceded_df = pd.read_excel('HomeConceeded.xlsx', index_col=0)
away_goals_df = pd.read_excel('AwayGoals.xlsx', index_col=0)
away_conceded_df = pd.read_excel('AwayConceeded.xlsx', index_col=0)

# Assuming you have a list of possible scorelines
scorelines = [(i, j) for i in range(5) for j in range(5)]

# Negative binomial distribution function
def calculate_negative_binomial_probability(p, r, k):
    return nbinom.pmf(k, r, p)

n_folds = 5

# Function to predict the winner based on probabilities
def predict_winner(probabilities):
    max_prob = 0
    winner = 'D'
    for scoreline, probability in probabilities.items():
        if probability > max_prob:
            max_prob = probability
            winner = 'H' if scoreline[0] > scoreline[1] else 'A' if scoreline[1] > scoreline[0] else 'D'
    return winner

def k_fold_cross_validation(fixtures_df, n_folds, a_value):
    kf = KFold(n_splits = n_folds, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(fixtures_df):
        train_data, test_data = fixtures_df.iloc[train_index], fixtures_df.iloc[test_index]

        # Estimate r_value based on historical goal differences
        goal_differences = train_data['FTHG'] - train_data['FTAG']
        variance = np.var(goal_differences)
        r_value = a_value * variance

# Calculate accuracy for each week
        weekly_accuracies = []
        weekly_over_2_5_goals_accuracies = []

        for week in fixtures_df['Week'].unique():   
            correct_predictions = 0
            correct_over_2_5_goals_predictions = 0
            total_predictions = 0

            week_fixtures = fixtures_df[fixtures_df['Week'] == week]

            if week < 5:
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
                over_2_5_goals_probability = 0

                for scoreline in scorelines:
                    home_goals, away_goals = scoreline
                    probability = (
                        calculate_negative_binomial_probability(Home_attack_rating * Away_defence_rating, r_value, home_goals) *
                        calculate_negative_binomial_probability(Away_attack_rating * Home_defence_rating, r_value, away_goals)
                    )
                    
                    probabilities[scoreline] = probability
                    
                total_probability = sum(probabilities.values())
                valid_probabilities = [prob for prob in probabilities.values() if not math.isnan(prob)]

                # Check if total probability is non-zero before normalization
                if total_probability != 0:
                    # Normalize probabilities
                    normalized_probabilities = {scoreline: prob / total_probability for scoreline, prob in probabilities.items()}

                    for scoreline, prob in normalized_probabilities.items():
                        home_win_prob = sum(p for (h, a), p in normalized_probabilities.items() if h > a)
                        away_win_prob = sum(p for (h, a), p in normalized_probabilities.items() if a > h)
                        draw_prob = sum(p for (h, a), p in normalized_probabilities.items() if h == a)

                        print(f"Home Teams: {home_team}, Away Teams: {away_team} Probability: {home_win_probability:.4f}, Home Win Probability: {away_win_probability:.4f}")
                        # Update over 2.5 goals probability
                    
                if home_goals + away_goals > 2:
                    over_2_5_goals_probability += probability

                    
               

        

        
        

                

            # Update over 2.5 goals probability
                if home_goals + away_goals > 2:
                    over_2_5_goals_probability += probability

        # Predict the winner based on probabilities
                predicted_result = predict_winner(probabilities)

        # Check if the prediction matches the actual result
                if predicted_result == actual_result:
                    correct_predictions += 1

        # Check if the prediction for over 2.5 goals is correct
                if over_2_5_goals_probability > 0.5:  # Adjust this threshold as needed
                    predicted_over_2_5_goals = 'O'
                else:
                    predicted_over_2_5_goals = 'U'

                actual_over_2_5_goals = 'O' if fixture['FTHG'] + fixture['FTAG'] > 2.5 else 'U'

                if predicted_over_2_5_goals == actual_over_2_5_goals:
                    correct_over_2_5_goals_predictions += 1

                total_predictions += 1

        

    # Calculate accuracy for the week
            accuracy = correct_predictions / total_predictions
            weekly_accuracies.append((week, accuracy))

    # Calculate accuracy for predicting over 2.5 goals for the week
            accuracy_over_2_5_goals = correct_over_2_5_goals_predictions / total_predictions
            weekly_over_2_5_goals_accuracies.append((week, accuracy_over_2_5_goals))

    # Calculate overall accuracy for match winner
        overall_accuracy = sum(correct_predictions for _, correct_predictions in weekly_accuracies) / sum(total_predictions for _, _ in weekly_accuracies)

# Calculate overall accuracy for over 2.5 goals
        overall_over_2_5_goals_accuracy = sum(correct_over_2_5_goals_predictions for _, correct_over_2_5_goals_predictions in weekly_over_2_5_goals_accuracies) / sum(total_predictions for _, _ in weekly_over_2_5_goals_accuracies)

        accuracies.append(overall_accuracy)

    average_accuracy = sum(accuracies) / n_folds
    return average_accuracy

# Define a range of a values to search for the optimal one
a_values_to_try = [0.7]  # Adjust as needed

# Initialize variables to store the best result and corresponding a value
best_accuracy = 0
best_a_value = None

# Loop through each a value and perform cross-validation
for a_value in a_values_to_try:
    average_accuracy = k_fold_cross_validation(fixtures_df, n_folds, a_value)

    # Update the best result if the current a value performs better
    if average_accuracy > best_accuracy:
        best_accuracy = average_accuracy
        best_a_value = a_value




# Print overall accuracies


# Print the optimal a value and corresponding accuracy
print(f"Optimal 'a' value: {best_a_value}")
print(f"Corresponding Average Accuracy: {best_accuracy * 1000:.2f}%")

