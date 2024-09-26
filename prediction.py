import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained Gradient Boosting Regressor model
model = joblib.load('gradient_boosting_regressor.pkl')

# Load the data from the Excel file
df = pd.read_excel('backtesting_results for ml future.xlsx')

# Ensure the data columns match the model's expected input
# You may need to adjust the column names based on how they are named in your Excel file

# Manual mapping for categorical features
symbol_mapping = {
    'BTC/USDT': 0, 'ETH/USDT': 1, 'BNB/USDT': 2, 'SOL/USDT': 3, 'XRP/USDT': 4,
    'ADA/USDT': 5, 'DOGE/USDT': 6, 'DOT/USDT': 7, 'LINK/USDT': 8, 'IMX/USDT': 9, 'ICP/USDT': 10
}
timeframe_mapping = {
    '15m': 0, '30m': 1, '1h': 2, '2h': 3, '4h': 4
}
type_mapping = {
    'Long': 0, 'Short': 1
}

df['Symbol'] = df['Symbol'].map(symbol_mapping)
df['Timeframe'] = df['Timeframe'].map(timeframe_mapping)
df['Type'] = df['Type'].map(type_mapping)

# Normalize continuous features
continuous_features = ['Entry Price', 'TP Multiplier', 'RSI Line Slope (k)', 'RSI Line Intercept (b)',
                       'MACD Line Slope (k)', 'MACD Line Intercept (b)']
scaler = StandardScaler()
df[continuous_features] = scaler.fit_transform(df[continuous_features])

# Select features for prediction
features = df[['Symbol', 'Timeframe', 'Entry Price', 'Type', 'TP Multiplier',
               'RSI Line Slope (k)', 'RSI Line Intercept (b)', 'MACD Line Slope (k)', 'MACD Line Intercept (b)']]

# Perform predictions
df['Predicted Profit Change'] = model.predict(features)

# Optionally, save the results to a new Excel file
df.to_excel('predicted_results.xlsx', index=False)

print("Predictions completed and saved to predicted_results.xlsx.")
