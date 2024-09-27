import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from _managing_data import update_data

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
    return df

def macd_signals(df):
    df['potential_signal'] = 0
    last_signal = None  # Keeps track of the last signal type issued

    for i in range(1, len(df)):
        if df['macd'].iloc[i] > df['macd_signal'].iloc[i] and last_signal != 'long' and df['HA_close'].iloc[i] > df['SuperTrend'].iloc[i] and df['HA_close'].iloc[i] > df['SMA200'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = 1  # Long signal
            last_signal = 'long'
        elif df['macd'].iloc[i] < df['macd_signal'].iloc[i] and last_signal != 'short' and df['HA_close'].iloc[i] < df['SuperTrend'].iloc[i] and df['HA_close'].iloc[i] < df['SMA200'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = -1  # Short signal
            last_signal = 'short'

    return df

def backtest_strategy(df, df_higher):
    initial_capital = 10000
    risk_per_trade = 0.05  # 5% of capital risked per trade
    results = []
    capital = initial_capital
    max_drawdown = 0
    peak_capital = capital
    last_exit_time = None  # This will keep track of the last exit time

    tp_multipliers = [1, 1.5, 2]

    for index, row in df.iterrows():
        # Skip to next iteration if current row time is before last exit time
        if last_exit_time is not None and row.name <= last_exit_time:
            continue

        # Conditions for Long and Short positions based on MACD, SuperTrend, and SMA200
        if row['potential_signal'] == 1 and row['HA_close'] > row['SuperTrend'] and row['HA_close'] > row['SMA200']:
            signal = 1  # Long position
        elif row['potential_signal'] == -1 and row['HA_close'] < row['SuperTrend'] and row['HA_close'] < row['SMA200']:
            signal = -1  # Short position
        else:
            continue  # No valid signal, skip the iteration

        entry_price = row['open']
        atr = row['ATR']
        super_trend = row['SuperTrend']
        stop_loss = super_trend - atr if signal == 1 else super_trend + atr
        stop_loss_distance = abs(entry_price - stop_loss)
        position_size = (risk_per_trade * capital) / stop_loss_distance

        previous_rsi_values, rsi_times = get_previous_values(df_higher, row.name, 'RSI')
        previous_macd_values, macd_times = get_previous_values(df_higher, row.name, 'macd')  # Fetch the last five MACD values

        rsi_k, rsi_b = get_linear_coeffs(previous_rsi_values) if previous_rsi_values is not None else (None, None)  # Calculate regression line for RSI
        macd_k, macd_b = get_linear_coeffs(previous_macd_values) if previous_macd_values is not None else (None, None)  # Calculate regression line for MACD

        future_rows = df.iloc[df.index.get_loc(index) + 1:]  # Start checking from the next row

        for tp_multiplier in tp_multipliers:
            tp = entry_price + tp_multiplier * stop_loss_distance if signal == 1 else entry_price - tp_multiplier * stop_loss_distance
            exit_price = None
            highest_high = row['high']
            lowest_low = row['low']

            for j, future_row in future_rows.iterrows():
                if signal == 1:
                    highest_high = max(highest_high, future_row['high'])
                    if future_row['low'] <= stop_loss or future_row['high'] >= tp:
                        exit_price = stop_loss if future_row['low'] <= stop_loss else tp
                        break
                else:
                    lowest_low = min(lowest_low, future_row['low'])
                    if future_row['high'] >= stop_loss or future_row['low'] <= tp:
                        exit_price = stop_loss if future_row['high'] >= stop_loss else tp
                        break

            if exit_price is None:
                exit_price = row['close']  # Default exit if no stop/TP was hit

            last_exit_time = future_row.name  # Update last exit time with the end of the current trade

            optimum_closing = tp if exit_price == tp else (highest_high if signal == 1 else lowest_low)
            profit = (exit_price - entry_price) * position_size if signal == 1 else (entry_price - exit_price) * position_size

            results.append({
                'Symbol': row['Symbol'], 'Timeframe': row['Timeframe'],
                'Entry Price': entry_price, 'Exit Price': exit_price,
                'Profit': profit, 'Type': 'Long' if signal == 1 else 'Short',
                'Entry Date': index, 'Exit Date': j, 'TP Multiplier': tp_multiplier,
                'Optimum Closing': optimum_closing, 'Previous RSI Values': previous_rsi_values[-1] if previous_rsi_values is not None else [],
                'RSI Times': rsi_times.tolist() if rsi_times is not None else [], 'RSI Line Slope (k)': rsi_k, 'RSI Line Intercept (b)': rsi_b,
                'Previous MACD Values': previous_macd_values.tolist() if previous_macd_values is not None else [],
                'MACD Times': macd_times.tolist() if macd_times is not None else [], 'MACD Line Slope (k)': macd_k, 'MACD Line Intercept (b)': macd_b
            })

    return results

def plot_data(df, symbol, timeframe):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['HA_open'], high=df['HA_high'],
                                 low=df['HA_low'], close=df['HA_close'],
                                 name='Heikin Ashi'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], line=dict(color='blue', width=1.5), name='SuperTrend'))
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], line=dict(color='green', width=1.5), name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], line=dict(color='red', width=1.5), name='MACD Signal'))
    buys = df[df['potential_signal'] == 1]
    sells = df[df['potential_signal'] == -1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['HA_close'], mode='markers', marker_symbol='triangle-up', marker_color='green', marker_size=10, name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sells.index, y=sells['HA_close'], mode='markers', marker_symbol='triangle-down', marker_color='red', marker_size=10, name='Sell Signal'))
    fig.update_layout(title=f"Trading Signals for {symbol} - {timeframe}", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    fig.show()

def main(symbol, timeframe):
    df = update_data(symbol, timeframe)
    df = calculate_heikin_ashi(df)
    df = calculate_indicators(df)
    df = macd_signals(df)
    plot_data(df, symbol, timeframe)

if __name__ == "__main__":
    symbol = 'BTC/USDT'
    timeframe = '15m'
    main(symbol, timeframe)
