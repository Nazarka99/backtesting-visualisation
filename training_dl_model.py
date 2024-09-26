import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data from Excel
df = pd.read_excel('backtesting_results.xlsx')

# Selecting the necessary columns
features = ['Symbol', 'Timeframe', 'Entry Price', 'Type', 'TP Multiplier',
            'RSI Line Slope (k)', 'RSI Line Intercept (b)', 'MACD Line Slope (k)', 'MACD Line Intercept (b)']
X = df[features]

# Calculate Profit Change
df['Profit change'] = df['Entry Price'] / df['Optimum Closing']
y = df['Profit change']

# Handling categorical data
categorical_features = ['Symbol', 'Timeframe', 'Type']
X = pd.get_dummies(X, columns=categorical_features)

# Normalizing continuous data
continuous_features = ['Entry Price', 'TP Multiplier', 'RSI Line Slope (k)', 'RSI Line Intercept (b)',
                       'MACD Line Slope (k)', 'MACD Line Intercept (b)']
scaler = StandardScaler()
X[continuous_features] = scaler.fit_transform(X[continuous_features])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Calculate and print the performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# Optional: Save the trained model for later use
import joblib
joblib.dump(model, 'gradient_boosting_regressor.pkl')
print("Model saved successfully!")

# If you need to load the model later:
# model = joblib.load('gradient_boosting_regressor.pkl')
