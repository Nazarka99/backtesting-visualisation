import pandas as pd
import pandas_ta as ta
from openpyxl import Workbook

# Assuming these functions are defined in your local modules
from _managing_data import update_data
from common_functions import calculate_heikin_ashi, calculate_supertrend, calculate_macd

# Define symbols and timeframes
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT',
           'DOT/USDT', 'LINK/USDT', 'IMX/USDT', 'ICP/USDT']

timeframes = ['15m', '30m', '1h', '2h', '4h']

# symbols = ['BTC/USDT', 'ETH/USDT']

# timeframes = ['30m', '1h']
#
# symbols = ['BTC/USDT', 'ETH/USDT']



def macd_signals(df):
    """Generate potential trading opportunities based on MACD crossover."""
    df['potential_signal'] = 0
    for i in range(1, len(df)):
        if df['macd'].iloc[i] > df['Signal_Line'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = 1  # Potential buy
        elif df['macd'].iloc[i] < df['Signal_Line'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = -1  # Potential sell
    return df


def backtest_strategy(df):
    initial_capital = 10000
    risk_per_trade = 0.05
    trades = []
    capital = initial_capital
    max_drawdown = 0
    peak_capital = capital
    trade_open = False  # Indicates if a trade is currently open

    tp_multipliers = [1, 1.5, 2]
    results = {multiplier: [] for multiplier in tp_multipliers}

    for index, row in df.iterrows():
        if trade_open:
            continue  # Skip to next row if a trade is already open

        if row['potential_signal'] == 1 and row['HA_close'] > row['SuperTrend'] and row['HA_close'] > row['SMA200']:
            row['signal'] = 1  # Long position
        elif row['potential_signal'] == -1 and row['HA_close'] < row['SuperTrend'] and row['HA_close'] < row['SMA200']:
            row['signal'] = -1  # Short position

        if 'signal' in row:
            entry_price = row['open']
            atr = row['ATR']
            super_trend = row['SuperTrend']
            stop_loss = super_trend - atr if row['signal'] == 1 else super_trend + atr
            stop_loss_distance = abs(entry_price - stop_loss)

            position_size = (risk_per_trade * capital) / stop_loss_distance
            trade_open = True  # Mark that a trade is now open

            for tp_multiplier in tp_multipliers:
                tp = entry_price + tp_multiplier * stop_loss_distance - atr if row['signal'] == 1 else entry_price - tp_multiplier * stop_loss_distance + atr
                exit_price = None

                for j, future_row in df.loc[index:].iterrows():
                    if row['signal'] == 1 and (future_row['low'] <= stop_loss or future_row['high'] >= tp):
                        exit_price = stop_loss if future_row['low'] <= stop_loss else tp
                        break
                    elif row['signal'] == -1 and (future_row['high'] >= stop_loss or future_row['low'] <= tp):
                        exit_price = stop_loss if future_row['high'] >= stop_loss else tp
                        break

                if exit_price is None:
                    continue  # If no exit was triggered, evaluate next TP

                profit = (exit_price - entry_price) * position_size if row['signal'] == 1 else (entry_price - exit_price) * position_size
                capital += profit
                peak_capital = max(peak_capital, capital)
                current_drawdown = peak_capital - capital
                max_drawdown = max(max_drawdown, current_drawdown)

                trade = {
                    'Entry': entry_price,
                    'Exit': exit_price,
                    'Profit': profit,
                    'Type': 'Long' if row['signal'] == 1 else 'Short',
                    'Entry Date': index,
                    'Exit Date': j,
                    'TP Multiplier': tp_multiplier  # Track the TP multiplier for this trade
                }
                results[tp_multiplier].append(trade)
                trade_open = False  # Close the trade after exit

    final_results = {}
    for multiplier, trades in results.items():
        win_rate = sum(1 for trade in trades if trade['Profit'] > 0) / len(trades) if trades else 0
        total_profit = sum(trade['Profit'] for trade in trades)
        final_results[multiplier] = {
            'Trades': trades,
            'Win Rate': win_rate,
            'Total Profit': total_profit,
            'Max Drawdown': max_drawdown
        }
    return final_results


def calculate_indicators(df):
    df['SMA200'] = ta.sma(df['HA_close'], length=200)
    # macd_values = ta.macd(df['HA_close'])
    # df['macd'] = macd_values['MACD_12_26_9']
    # df['macd_signal'] = macd_values['MACDs_12_26_9']
    # df['macd_hist'] = macd_values['MACDh_12_26_9']

    # Calculation MACD
    # Calculate the 12-period EMA
    df['EMA12'] = df['HA_close'].ewm(span=12, adjust=False).mean()

    # Calculate the 26-period EMA
    df['EMA26'] = df['HA_close'].ewm(span=26, adjust=False).mean()

    # Calculate MACD (the difference between 12-period EMA and 26-period EMA)
    df['macd'] = df['EMA12'] - df['EMA26']

    # Calculate the 9-period EMA of MACD (Signal Line)
    df['Signal_Line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['macd'] - df['Signal_Line']

    supertrend_values = ta.supertrend(df['HA_high'], df['HA_low'], df['HA_close'], length=12, multiplier=3)
    df['SuperTrend'] = supertrend_values['SUPERT_12_3.0']
    df['ATR'] = ta.atr(df['HA_high'], df['HA_low'], df['HA_close'], length=14)
    df['RSI'] = ta.rsi(df['HA_close'], length=14)  # Calculate RSI

    return df


def save_results_to_excel(results, filename='backtesting_results.xlsx'):
    """Save the backtesting results to an Excel file in a single sheet."""
    # Create an empty DataFrame to collect all trade data
    consolidated_data = pd.DataFrame()

    # Iterate through all trade data and append to the consolidated DataFrame
    for (symbol, timeframe, tp_multiplier), trades in results.items():
        df = pd.DataFrame(trades)
        df['Symbol'] = symbol
        df['Timeframe'] = timeframe
        df['TP Multiplier'] = tp_multiplier
        consolidated_data = pd.concat([consolidated_data, df], ignore_index=True)

    # Write the consolidated DataFrame to an Excel file
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        consolidated_data.to_excel(writer, sheet_name='Consolidated Results', index=False)

def save_summary_to_excel(summary, filename='backtesting_summary.xlsx'):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # Create DataFrame from summary list including the 'Trades Count'
        df_summary = pd.DataFrame(summary)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

def main():
    results = {}
    summary = []

    for symbol in symbols:
        for timeframe in timeframes:
            df = update_data(symbol, timeframe)
            df = calculate_heikin_ashi(df)
            df = calculate_indicators(df)
            df = macd_signals(df)
            metrics = backtest_strategy(df)  # Returns dictionary keyed by TP multiplier
            for tp_multiplier, metric in metrics.items():
                results[(symbol, timeframe, tp_multiplier)] = metric['Trades']
                summary.append({
                    'Symbol': symbol,
                    'Timeframe': timeframe,
                    'TP Multiplier': tp_multiplier,
                    'Win Rate': metric['Win Rate'],
                    'Total Profit': metric['Total Profit'],
                    'Max Drawdown': metric['Max Drawdown'],
                    'Trades Count': len(metric['Trades'])
                })

            print(f'Done for {symbol} with the {timeframe} timeframe')

    save_results_to_excel(results)
    save_summary_to_excel(summary)
    print("Backtesting completed and results are saved.")


if __name__ == "__main__":
    main()

