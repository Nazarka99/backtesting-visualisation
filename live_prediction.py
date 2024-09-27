import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import joblib  # For loading the model
from sklearn.preprocessing import StandardScaler
from _managing_data import update_data

# Timeframe mapping for higher timeframes
higher_timeframes = {
    '15m': '1h',
    '30m': '1h',
    '1h': '4h',
    '2h': '4h',
    '4h': '1d'
}

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


def calculate_heikin_ashi(data):
    data['HA_close'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    data['HA_open'] = (data['open'].shift(1) + data['close'].shift(1)) / 2
    data['HA_open'].iloc[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2
    data['HA_high'] = data[['high', 'HA_open', 'HA_close']].max(axis=1)
    data['HA_low'] = data[['low', 'HA_open', 'HA_close']].min(axis=1)
    return data


def calculate_indicators(df):
    df['SMA200'] = ta.sma(df['HA_close'], length=200)
    macd = ta.macd(df['HA_close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    supertrend = ta.supertrend(df['HA_high'], df['HA_low'], df['HA_close'], length=12, multiplier=3)
    df['SuperTrend'] = supertrend['SUPERT_12_3.0']
    df['ATR'] = ta.atr(df['HA_high'], df['HA_low'], df['HA_close'], length=14)
    df['RSI'] = ta.rsi(df['HA_close'], length=14)
    return df


def macd_signals(df):
    df['potential_signal'] = 0
    last_signal = None  # Keeps track of the last signal type issued

    for i in range(1, len(df)):
        if df['macd'].iloc[i] > df['macd_signal'].iloc[i] and last_signal != 'long' and df['HA_close'].iloc[i] > \
                df['SuperTrend'].iloc[i] and df['HA_close'].iloc[i] > df['SMA200'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = 1  # Long signal
            last_signal = 'long'
        elif df['macd'].iloc[i] < df['macd_signal'].iloc[i] and last_signal != 'short' and df['HA_close'].iloc[i] < \
                df['SuperTrend'].iloc[i] and df['HA_close'].iloc[i] < df['SMA200'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = -1  # Short signal
            last_signal = 'short'

    return df


def get_previous_values(df_higher, signal_time, column):
    valid_times = df_higher[df_higher.index <= signal_time]
    if not valid_times.empty:
        last_times = valid_times.index[-5:]
        return df_higher.loc[last_times][column].values, last_times
    return None, None


def calculate_linear_regression(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b


def plot_data(df, symbol, timeframe):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['HA_open'], high=df['HA_high'],
                                 low=df['HA_low'], close=df['HA_close'],
                                 name='Heikin Ashi'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], line=dict(color='blue', width=1.5), name='SuperTrend'))
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], line=dict(color='green', width=1.5), name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], line=dict(color='red', width=1.5), name='MACD Signal'))

    # Get the last signal, either buy (1) or sell (-1)
    last_buy = df[df['potential_signal'] == 1].iloc[-1:] if (df['potential_signal'] == 1).any() else pd.DataFrame()
    last_sell = df[df['potential_signal'] == -1].iloc[-1:] if (df['potential_signal'] == -1).any() else pd.DataFrame()

    # Determine the most recent signal (either buy or sell)
    if not last_buy.empty and not last_sell.empty:
        last_signal = last_buy if last_buy.index[-1] > last_sell.index[-1] else last_sell
    elif not last_buy.empty:
        last_signal = last_buy
    elif not last_sell.empty:
        last_signal = last_sell
    else:
        last_signal = pd.DataFrame()  # No signal

    # Plot the last signal (either buy or sell)
    if not last_signal.empty:
        signal_type = 'triangle-up' if last_signal['potential_signal'].iloc[0] == 1 else 'triangle-down'
        signal_color = 'green' if last_signal['potential_signal'].iloc[0] == 1 else 'red'
        signal_name = 'Last Buy Signal' if last_signal['potential_signal'].iloc[0] == 1 else 'Last Sell Signal'
        fig.add_trace(go.Scatter(x=last_signal.index, y=last_signal['HA_close'], mode='markers',
                                 marker_symbol=signal_type, marker_color=signal_color, marker_size=10,
                                 name=signal_name))

    # fig.update_layout(title=f"Last Trading Signal for {symbol} - {timeframe}", xaxis_title="Date", yaxis_title="Price",
    #                   template="plotly_dark")
    # fig.show()

def main(symbol, timeframe):
    # Fetch data for the main timeframe
    df = update_data(symbol, timeframe)
    df = calculate_heikin_ashi(df)
    df = calculate_indicators(df)
    df = macd_signals(df)

    # Fetch data for the higher timeframe
    higher_timeframe = higher_timeframes[timeframe]
    df_higher = update_data(symbol, higher_timeframe)
    df_higher = calculate_heikin_ashi(df_higher)
    df_higher = calculate_indicators(df_higher)

    # Get the last signal
    last_signal_row = df[df['potential_signal'] != 0].iloc[-1]  # Last non-zero signal row

    # Get the previous 5 values of RSI and MACD from the higher timeframe
    previous_rsi_values, rsi_times = get_previous_values(df_higher, last_signal_row.name, 'RSI')
    previous_macd_values, macd_times = get_previous_values(df_higher, last_signal_row.name, 'macd')

    # Print the prediction and entry price
    if previous_rsi_values is not None and previous_macd_values is not None:
        # Calculate k and b for RSI and MACD
        x = np.arange(5)  # x values as [0, 1, 2, 3, 4]
        rsi_k, rsi_b = calculate_linear_regression(x, previous_rsi_values)
        macd_k, macd_b = calculate_linear_regression(x, previous_macd_values)

        # Prepare data for ML prediction
        features_dict = {
            'Symbol': symbol_mapping[symbol],
            'Timeframe': timeframe_mapping[timeframe],
            'Entry Price': last_signal_row['HA_close'],  # Example Entry Price
            'Type': type_mapping['Long'] if last_signal_row['potential_signal'] == 1 else type_mapping['Short'],  # Determine type
            'TP Multiplier': 1.0,  # You can replace this with your logic for TP multiplier
            'RSI Line Slope (k)': rsi_k,
            'RSI Line Intercept (b)': rsi_b,
            'MACD Line Slope (k)': macd_k,
            'MACD Line Intercept (b)': macd_b
        }

        # Prepare the last_features array
        last_features = np.array(list(features_dict.values())).reshape(1, -1)

        # Normalize the features
        scaler = StandardScaler()
        last_features = scaler.fit_transform(last_features)

        # Load the pre-trained model
        model = joblib.load("gradient_boosting_regressor.pkl")

        # Predict the next price
        prediction = model.predict(last_features)

        # Calculate the ratio of entry price to predicted price
        entry_price = last_signal_row['HA_close']
        predicted_price = prediction[0]
        ratio = entry_price / predicted_price

        # Print only the prediction, the entry price, and the ratio
        print(f"Entry Price: {entry_price}")
        print(f"Predicted Next Price: {predicted_price}")
        print(f"Entry Price / Predicted Price Ratio: {ratio}")
    else:
        print("Insufficient data to retrieve previous values.")

    # # Plotting the data
    # plot_data(df, symbol, timeframe)



# Example call to the main function
if __name__ == "__main__":
    main('BTC/USDT', '15m')

