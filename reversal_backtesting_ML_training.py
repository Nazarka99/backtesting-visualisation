# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Import joblib for saving the model

# Step 1: Load the data
df = pd.read_excel('backtest_results_multi_symbol.xlsx', sheet_name='Trades')

# Step 2: Data Preprocessing

# 2.1 Custom encoding mappings
symbol_mapping = {
    'BTC/USDT': 0, 'ETH/USDT': 1, 'BNB/USDT': 2, 'SOL/USDT': 3, 'XRP/USDT': 4,
    'ADA/USDT': 5, 'DOGE/USDT': 6, 'MATIC/USDT': 7, 'DOT/USDT': 8, 'LINK/USDT': 9,
    'IMX/USDT': 10, 'ICP/USDT': 11
}
timeframe_mapping = {
    '15m': 0, '30m': 1, '1h': 2, '2h': 3, '4h': 4
}
type_mapping = {
    'Long': 0, 'Short': 1
}

# 2.2 Apply custom encoding to categorical variables
df['Symbol_encoded'] = df['Symbol'].map(symbol_mapping)
df['Timeframe_encoded'] = df['Timeframe'].map(timeframe_mapping)
df['Type_encoded'] = df['Type'].map(type_mapping)

# 2.3 Encode target variable 'Result'
df['Result_encoded'] = df['Result'].map({'Win': 1, 'Loss': 0})

# 2.4 Prepare feature matrix X and target vector y
feature_columns = [
    'Symbol_encoded', 'Timeframe_encoded', 'Type_encoded',
    'RSI_7', 'RSI_14', 'RSI_21',
    'KER_RSI_7', 'KER_RSI_14', 'KER_RSI_21',
    'MACD_Line', 'Signal_Line', 'MACD_Histogram',
    'Higher_RSI_7', 'Higher_RSI_14', 'Higher_RSI_21',
    'Higher_KER_RSI_7', 'Higher_KER_RSI_14', 'Higher_KER_RSI_21',
    'Higher_MACD_Line', 'Higher_Signal_Line', 'Higher_MACD_Histogram'
]

X = df[feature_columns]
y = df['Result_encoded']
df['Open_Time'] = df['Open_Time']  # Ensure Open_Time is retained for splitting

# Step 3: Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y  # Ensure proportional representation of classes
)

# Step 3.1: Save training and testing data into Excel sheets
train_data = df.loc[X_train.index].copy()
test_data = df.loc[X_test.index].copy()

with pd.ExcelWriter('backtest_results_multi_symbol.xlsx', mode='a', engine='openpyxl') as writer:
    train_data.to_excel(writer, sheet_name='train', index=False)
    test_data.to_excel(writer, sheet_name='test', index=False)

# Step 4: Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    class_weight='balanced'  # Handle class imbalance if present
)

rf_classifier.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Overall Accuracy: {accuracy:.2f}')

classification_rep = classification_report(y_test, y_pred)
print('Classification Report:')
print(classification_rep)

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Step 7: Feature Importance
feature_importances = pd.Series(
    rf_classifier.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

print('Feature Importances:')
print(feature_importances)

# Step 8: Save predictions to Excel
# Map encoded values back to original categories for readability
X_test_results = X_test.copy()
X_test_results['Actual_Result'] = y_test.map({1: 'Win', 0: 'Loss'})
X_test_results['Predicted_Result'] = y_pred
X_test_results['Predicted_Result'] = X_test_results['Predicted_Result'].map({1: 'Win', 0: 'Loss'})

# Decode categorical features using the reverse mapping
reverse_symbol_mapping = {v: k for k, v in symbol_mapping.items()}
reverse_timeframe_mapping = {v: k for k, v in timeframe_mapping.items()}
reverse_type_mapping = {v: k for k, v in type_mapping.items()}

X_test_results['Symbol'] = X_test_results['Symbol_encoded'].map(reverse_symbol_mapping)
X_test_results['Timeframe'] = X_test_results['Timeframe_encoded'].map(reverse_timeframe_mapping)
X_test_results['Type'] = X_test_results['Type_encoded'].map(reverse_type_mapping)

# Reorder columns for clarity
output_columns = [
    'Symbol', 'Timeframe', 'Type',
    'RSI_7', 'RSI_14', 'RSI_21',
    'KER_RSI_7', 'KER_RSI_14', 'KER_RSI_21',
    'MACD_Line', 'Signal_Line', 'MACD_Histogram',
    'Higher_RSI_7', 'Higher_RSI_14', 'Higher_RSI_21',
    'Higher_KER_RSI_7', 'Higher_KER_RSI_14', 'Higher_KER_RSI_21',
    'Higher_MACD_Line', 'Higher_Signal_Line', 'Higher_MACD_Histogram',
    'Actual_Result', 'Predicted_Result'
]

X_test_results = X_test_results[output_columns]

# Save to Excel
X_test_results.to_excel('model_test_results.xlsx', index=False)

# Optional Step 9: Visualization

# 9.1 Plot Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 9.2 Plot Feature Importances
plt.figure(figsize=(12,6))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title('Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Step 10: Calculate accuracy per timeframe

# Group by 'Timeframe' and calculate accuracy for each group, excluding the grouping column
timeframe_accuracies = X_test_results.groupby('Timeframe').apply(
    lambda group: accuracy_score(group['Actual_Result'], group['Predicted_Result'])
).reset_index(name='Accuracy')

print("\nAccuracy per Timeframe:")
print(timeframe_accuracies)

# Save accuracy per timeframe to Excel using openpyxl engine for append mode
with pd.ExcelWriter('model_test_results.xlsx', mode='a', engine='openpyxl') as writer:
    timeframe_accuracies.to_excel(writer, sheet_name='Timeframe_Accuracy', index=False)

# Step 11: Save the trained model with versioning

model_filename = 'random_forest_model_v1.0.pkl'
joblib.dump(rf_classifier, model_filename)
print(f"Model saved as {model_filename}")