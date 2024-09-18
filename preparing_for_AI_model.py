import pandas as pd
import pandas_ta as ta
from openpyxl import Workbook

# Assuming these functions are defined in your local modules
from _managing_data import update_data
from common_functions import calculate_heikin_ashi, calculate_supertrend, calculate_macd

# Define symbols and timeframes
timeframes = ['30m', '1h']
symbols = ['BTC/USDT', 'ETH/USDT']

def macd_signals(df):
    """Generate potential trading opportunities based on MACD crossover."""
    df['potential_signal'] = 0
    for i in range(1, len(df)):
        if df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = 1  # Potential buy
        elif df['macd'].iloc[i] < df['macd_signal'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = -1  # Potential sell
    return df

def backtest_strategy(df):
    initial_capital = 10000
    risk_per_trade = 0.05
    trades = []
    capital = initial_capital
    max_drawdown = 0
    peak_capital = capital
    trade_open = False

    tp_multipliers = [1, 1.5, 2]
    results = []

    for index, row in df.iterrows():
        if trade_open:
            continue

        if row['potential_signal'] == 1 and row['HA_close'] > row['SuperTrend'] and row['HA_close'] > row['SMA200']:
            signal = 1  # Long position
        elif row['potential_signal'] == -1 and row['HA_close'] < row['SuperTrend'] and row['HA_close'] < row['SMA200']:
            signal = -1  # Short position
        else:
            continue

        entry_price = row['open']
        atr = row['ATR']
        super_trend = row['SuperTrend']
        stop_loss = super_trend - atr if signal == 1 else super_trend + atr
        stop_loss_distance = abs(entry_price - stop_loss)

        for tp_multiplier in tp_multipliers:
            tp = entry_price + tp_multiplier * stop_loss_distance if signal == 1 else entry_price - tp_multiplier * stop_loss_distance
            exit_price = None
            highest_high = row['high']
            lowest_low = row['low']

            for j, future_row in df.loc[index:].iterrows():
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

            if exit_price is not None:
                optimum_closing = tp if exit_price == tp else (highest_high if signal == 1 else lowest_low)
                profit = (exit_price - entry_price) if signal == 1 else (entry_price - exit_price)
                results.append({
                    'Symbol': row['Symbol'], 'Timeframe': row['Timeframe'],
                    'Entry Price': entry_price, 'Exit Price': exit_price,
                    'Profit': profit, 'Type': 'Long' if signal == 1 else 'Short',
                    'Entry Date': index, 'Exit Date': j, 'TP Multiplier': tp_multiplier,
                    'Optimum Closing': optimum_closing
                })
                trade_open = False  # Close trade after processing

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
    return df

def save_results_to_excel(results, filename='backtesting_results.xlsx'):
    df = pd.DataFrame(results)
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Consolidated Results', index=False)

def main():
    results = []
    for symbol in symbols:
        for timeframe in timeframes:
            df = update_data(symbol, timeframe)  # Assume this fetches the data
            df = calculate_heikin_ashi(df)
            df = calculate_indicators(df)
            df = macd_signals(df)

            # Add 'Symbol' and 'Timeframe' columns before passing to backtesting
            df['Symbol'] = symbol
            df['Timeframe'] = timeframe

            trades = backtest_strategy(df)
            results.extend(trades)
    save_results_to_excel(results)

if __name__ == "__main__":
    main()
