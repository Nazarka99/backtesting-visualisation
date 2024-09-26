import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

"""training ml-model"""

# Load data from Excel
df = pd.read_excel('backtesting_results.xlsx')

# Manual mappings for categorical features
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

# Apply mappings
df['Symbol'] = df['Symbol'].map(symbol_mapping)
df['Timeframe'] = df['Timeframe'].map(timeframe_mapping)
df['Type'] = df['Type'].map(type_mapping)

# Selecting the necessary columns
features = ['Symbol', 'Timeframe', 'Entry Price', 'Type', 'TP Multiplier',
            'RSI Line Slope (k)', 'RSI Line Intercept (b)', 'MACD Line Slope (k)', 'MACD Line Intercept (b)']
X = df[features]

# Calculate Profit Change
df['Profit change'] = df['Entry Price'] / df['Optimum Closing']
y = df['Profit change']

# Normalizing continuous data
continuous_features = ['Entry Price', 'TP Multiplier', 'RSI Line Slope (k)', 'RSI Line Intercept (b)',
                       'MACD Line Slope (k)', 'MACD Line Intercept (b)']
# scaler = StandardScaler()
# X[continuous_features] = scaler.fit_transform(X[continuous_features])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

print(X_test)

# Calculate and print the performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# Save the trained model for later use
joblib.dump(model, 'gradient_boosting_regressor.pkl')
print("Model saved successfully!")
