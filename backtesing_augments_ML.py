import pandas as pd

# Load the data from the Excel files
timeframe_accuracy_df = pd.read_excel('model_test_results.xlsx', sheet_name='Timeframe_Accuracy')
backtest_results_df = pd.read_excel('backtest_results_multi_symbol.xlsx', sheet_name='Summary')

# Initialize an empty list to store the new calculated win rates
calculated_win_rates = []

# Iterate over each row in the backtest results dataframe
for index, row in backtest_results_df.iterrows():
    symbol = row['Symbol']
    timeframe = row['Timeframe']
    original_win_rate = row['Win Rate']

    # Find the corresponding accuracy for the timeframe
    accuracy = timeframe_accuracy_df[timeframe_accuracy_df['Timeframe'] == timeframe]['Accuracy'].values[0]

    # Calculate the expected win rate after applying the ML model
    correct_wins = accuracy * original_win_rate
    incorrect_wins = (1 - accuracy) * (1 - original_win_rate)
    total_predicted_wins = correct_wins + incorrect_wins
    new_win_rate = correct_wins / total_predicted_wins if total_predicted_wins > 0 else 0

    # Append the calculated result to the list
    calculated_win_rates.append({
        'Symbol': symbol,
        'Timeframe': timeframe,
        'Original Win Rate': original_win_rate,
        'ML Model Accuracy': accuracy,
        'Calculated Win Rate': new_win_rate
    })

# Convert the results to a DataFrame
calculated_win_rates_df = pd.DataFrame(calculated_win_rates)

# Save the new calculated win rates to a new Excel file
calculated_win_rates_df.to_excel('calculated_win_rates.xlsx', index=False)

print("Calculated win rates saved to 'calculated_win_rates.xlsx'")
