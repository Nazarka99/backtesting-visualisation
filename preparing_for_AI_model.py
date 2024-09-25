import pandas as pd
import pandas_ta as ta
import numpy as np
from openpyxl import Workbook
from sklearn.linear_model import LinearRegression

# Assuming these functions are defined in your local modules
from _managing_data import update_data
from common_functions import calculate_heikin_ashi, calculate_supertrend, calculate_macd

# Define symbols and timeframes
timeframes = ['30m', '1h']
symbols = ['BTC/USDT', 'ETH/USDT']
higher_timeframes = {
    '15m': '1h',
    '30m': '1h',
    '1h': '4h',
    '2h': '4h',
    '4h': '1d'
}

def get_linear_coeffs(values):
    """Calculate linear regression coefficients k (slope) and b (intercept)."""
    x = np.arange(len(values)).reshape(-1, 1)
    y = values
    model = LinearRegression().fit(x, y)
    return model.coef_[0], model.intercept_

def macd_signals(df):
    """Generate potential trading opportunities based on MACD crossover."""
    df['potential_signal'] = 0
    for i in range(1, len(df)):
        if df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = 1  # Potential buy
        elif df['macd'].iloc[i] < df['macd_signal'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = -1  # Potential sell
    return df

def get_previous_rsi(df_higher, signal_time):
    """Retrieve the RSI value just before the signal time from the higher timeframe data."""
    valid_times = df_higher[df_higher.index <= signal_time]
    if not valid_times.empty:
        last_time = valid_times.index[-1]
        return df_higher.loc[last_time]['RSI'], last_time
    return None, None

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

        previous_rsi, rsi_time = get_previous_rsi(df_higher, row.name)  # Fetch the previous RSI value

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
                'Optimum Closing': optimum_closing, 'Previous RSI': previous_rsi, 'RSI Time': rsi_time
            })

    return results

def calculate_indicators(df):
    df['SMA200'] = ta.sma(df['HA_close'], length=200)
    macd_values = ta.macd(df['HA_close'])
    df['macd'] = macd_values['MACD_12_26_9']
    df['macd_signal'] = macd_values['MACDs_12_26_9']
    df['macd_hist'] = macd_values['MACDh_12_26_9']
    supertrend_values = ta.supertrend(df['HA_high'], df['HA_low'], df['HA_close'], length=12, multiplier=3)
    df['SuperTrend'] = supertrend_values['SUPERT_12_3.0']
    df['ATR'] = ta.atr(df['HA_high'], df['HA_low'], df['HA_close'], length=14)
    df['RSI'] = ta.rsi(df['HA_close'], length=14)  # Calculate RSI
    return df

def save_results_to_excel(results, filename='backtesting_results.xlsx'):
    df = pd.DataFrame(results)
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Consolidated Results', index=False)

def main():
    results = []
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"Processing {symbol} on {timeframe} timeframe")
            df = update_data(symbol, timeframe)  # Fetch data
            df = calculate_heikin_ashi(df)  # Convert data to Heikin Ashi
            df = calculate_indicators(df)  # Calculate indicators like MACD, ATR, etc.
            df = macd_signals(df)  # Identify MACD buy/sell signals
            df['Symbol'] = symbol  # Add symbol column for reference
            df['Timeframe'] = timeframe  # Add timeframe column for reference

            # Fetch and prepare higher timeframe data
            df_higher = update_data(symbol, higher_timeframes[timeframe])
            df_higher = calculate_heikin_ashi(df_higher)  # Convert higher timeframe data to Heikin Ashi
            df_higher = calculate_indicators(df_higher)  # Calculate indicators for the higher timeframe

            trades = backtest_strategy(df, df_higher)  # Perform backtesting
            results.extend(trades)  # Collect all trades across symbols and timeframes

    save_results_to_excel(results, filename='backtesting_results.xlsx')
    print("Backtesting completed and results are saved.")

if __name__ == "__main__":
    main()
