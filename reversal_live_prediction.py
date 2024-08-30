import pandas as pd
from _managing_data import update_data
from common_functions import calculate_heikin_ashi
import pandas_ta as ta
import joblib

# Helper function to determine the higher timeframe
def get_higher_timeframe(timeframe):
    if timeframe == '15m':
        return '1h'
    elif timeframe == '30m':
        return '2h'
    elif timeframe == '1h':
        return '4h'
    elif timeframe == '2h':
        return '4h'
    elif timeframe == '4h':
        return '1d'  # Optional: adjust as necessary
    else:
        return None

# Calculate Kaufman Efficiency Ratio (KER)
def calculate_kaufman_efficiency_ratio(data_series, period):
    signal = data_series.diff(period).abs()
    noise = data_series.diff().abs().rolling(window=period).sum()
    ker = signal / noise
    return ker

# Calculate all necessary indicators
def calculate_indicators(df):
    df[['SUPERT', 'direction', 'long', 'short']] = ta.supertrend(df.HA_high, df.HA_low, df.HA_close, length=12, multiplier=3)
    macd = ta.macd(df['HA_close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    df['ATR'] = ta.atr(df['HA_high'], df['HA_low'], df['HA_close'], length=14)
    df['RSI_7'] = ta.rsi(df['HA_close'], length=7)
    df['RSI_14'] = ta.rsi(df['HA_close'], length=14)
    df['RSI_21'] = ta.rsi(df['HA_close'], length=21)
    df['KER_RSI_7'] = calculate_kaufman_efficiency_ratio(df['RSI_7'], period=10)
    df['KER_RSI_14'] = calculate_kaufman_efficiency_ratio(df['RSI_14'], period=14)
    df['KER_RSI_21'] = calculate_kaufman_efficiency_ratio(df['RSI_21'], period=20)
    return df

# Align higher timeframe data with the lower timeframe data
def align_higher_timeframe_data(df, higher_df):
    higher_df = higher_df.resample('1T').ffill().reindex(df.index).ffill()
    return higher_df

# Generate signals
def generate_signals(df):
    df['Type_encoded'] = None  # Initialize the column with None
    df.loc[((df['MACD_12_26_9'].shift(1) < df['MACDs_12_26_9'].shift(1)) &
            (df['MACD_12_26_9'] > df['MACDs_12_26_9']) &
            (df['SUPERT'] > df['HA_close'])), 'Type_encoded'] = 0  # Long Signal

    df.loc[((df['MACD_12_26_9'].shift(1) > df['MACDs_12_26_9'].shift(1)) &
            (df['MACD_12_26_9'] < df['MACDs_12_26_9']) &
            (df['SUPERT'] < df['HA_close'])), 'Type_encoded'] = 1  # Short Signal
    return df

# Load the trained model
model_filename = 'random_forest_model_v1.0.pkl'
rf_classifier = joblib.load(model_filename)

# Main process
if __name__ == "__main__":
    # Dictionary with symbols and their corresponding timeframes
    # symbols_timeframes = {
    #     'LINK/USDT': ['30m'],
    #     'BNB/USDT': ['30m'],
    #     'DOGE/USDT': ['30m'],
    #     'XRP/USDT': ['15m']
    # }

    symbols_timeframes = {
        'XRP/USDT': ['2h']
    }

    predictions = []

    # Encoding mappings
    symbol_mapping = {
        'BTC/USDT': 0, 'ETH/USDT': 1, 'BNB/USDT': 2, 'SOL/USDT': 3, 'XRP/USDT': 4,
        'ADA/USDT': 5, 'DOGE/USDT': 6, 'MATIC/USDT': 7, 'DOT/USDT': 8, 'LINK/USDT': 9,
        'IMX/USDT': 10, 'ICP/USDT': 11
    }
    timeframe_mapping = {
        '15m': 0, '30m': 1, '1h': 2, '2h': 3, '4h': 4
    }

    for symbol, timeframes in symbols_timeframes.items():
        for timeframe in timeframes:
            # Fetch and process data for the lower timeframe
            df = update_data(symbol, timeframe)
            df = calculate_heikin_ashi(df)

            # Fetch and process data for the higher timeframe
            higher_timeframe = get_higher_timeframe(timeframe)
            if higher_timeframe:
                higher_df = update_data(symbol, higher_timeframe)
                higher_df = calculate_heikin_ashi(higher_df)
                higher_df = calculate_indicators(higher_df)
                higher_df = align_higher_timeframe_data(df, higher_df)

                # Calculate indicators and generate signals for the lower timeframe
                df = calculate_indicators(df)
                df = generate_signals(df)

                # Create a DataFrame to store signals with all required columns
                signals_df = pd.DataFrame({
                    'Symbol_encoded': symbol_mapping[symbol],
                    'Timeframe_encoded': timeframe_mapping[timeframe],
                    'Type_encoded': df['Type_encoded'],
                    'RSI_7': df['RSI_7'],
                    'RSI_14': df['RSI_14'],
                    'RSI_21': df['RSI_21'],
                    'KER_RSI_7': df['KER_RSI_7'],
                    'KER_RSI_14': df['KER_RSI_14'],
                    'KER_RSI_21': df['KER_RSI_21'],
                    'MACD_Line': df['MACD_12_26_9'],
                    'Signal_Line': df['MACDs_12_26_9'],
                    'MACD_Histogram': df['MACDh_12_26_9'],
                    'Higher_RSI_7': higher_df['RSI_7'],
                    'Higher_RSI_14': higher_df['RSI_14'],
                    'Higher_RSI_21': higher_df['RSI_21'],
                    'Higher_KER_RSI_7': higher_df['KER_RSI_7'],
                    'Higher_KER_RSI_14': higher_df['KER_RSI_14'],
                    'Higher_KER_RSI_21': higher_df['KER_RSI_21'],
                    'Higher_MACD_Line': higher_df['MACD_12_26_9'],
                    'Higher_Signal_Line': higher_df['MACDs_12_26_9'],
                    'Higher_MACD_Histogram': higher_df['MACDh_12_26_9']
                })

                # Filter only rows where a signal was generated
                signals_df = signals_df.dropna(subset=['Type_encoded'])

                # Predict for the last signal generated
                if not signals_df.empty:
                    last_signal = signals_df.iloc[-1]  # Select the last signal
                    features = last_signal.values.reshape(1, -1)
                    prediction = rf_classifier.predict(features)[0]
                    result = 'Win' if prediction == 1 else 'Loss'
                    predictions.append({
                        'Time': last_signal.name,
                        'Symbol': symbol,
                        'Timeframe': timeframe,
                        'Type': 'Long' if last_signal['Type_encoded'] == 0 else 'Short',
                        'Predicted_Result': result
                    })
                    # Print last prediction
                    print(f"Last Prediction for {symbol} on {timeframe} timeframe: {result}")

    # Convert predictions to DataFrame and save to Excel
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_excel('predictions_results.xlsx', index=False)
    print("Last predictions saved to predictions_results.xlsx")
