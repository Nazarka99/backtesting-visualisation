import pandas as pd
import pandas_ta as ta
from openpyxl import Workbook

# Assuming these functions are defined in your local modules
from _managing_data import update_data
from common_functions import calculate_heikin_ashi, calculate_supertrend, calculate_macd

# Define symbols and timeframes
# symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT',
#            'DOT/USDT', 'LINK/USDT', 'IMX/USDT', 'ICP/USDT']
# symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT',
#            'DOT/USDT', 'LINK/USDT', 'IMX/USDT', 'ICP/USDT']
timeframes = ['15m', '30m', '1h', '2h', '4h']

symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT',
           'DOT/USDT', 'LINK/USDT', 'IMX/USDT', 'ICP/USDT']
# timeframes = ['4h']

# symbols = ['BTC/USDT', 'ETH/USDT']
# timeframes = ['1h', '4h']


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

    for index, row in df.iterrows():
        if row['potential_signal'] == 1 and row['HA_close'] > row['SuperTrend'] and row['HA_close'] > row['SMA200']:
            # Confirm buy signal if above SuperTrend and SMA
            row['signal'] = 1
        elif row['potential_signal'] == -1 and row['HA_close'] < row['SuperTrend'] and row['HA_close'] < row['SMA200']:
            # Confirm sell signal if below SuperTrend and SMA
            row['signal'] = -1

        if 'signal' in row and row['signal'] != 0:
            entry_price = row['open']
            atr = row['ATR']
            super_trend = row['SuperTrend']
            stop_loss = super_trend - atr if row['signal'] == 1 else super_trend + atr
            take_profit = entry_price + (entry_price - stop_loss) if row['signal'] == 1 else entry_price - (stop_loss - entry_price)
            position_size = (risk_per_trade * capital) / abs(entry_price - stop_loss)
            exit_price = None

            for j, future_row in df.loc[index:].iterrows():
                if (future_row['low'] <= stop_loss and row['signal'] == 1) or (future_row['high'] >= stop_loss and row['signal'] == -1):
                    exit_price = stop_loss
                    break
                if (future_row['high'] >= take_profit and row['signal'] == 1) or (future_row['low'] <= take_profit and row['signal'] == -1):
                    exit_price = take_profit
                    break

            if exit_price is None:
                exit_price = future_row['close']  # Default exit price

            profit = (exit_price - entry_price) * position_size if row['signal'] == 1 else (entry_price - exit_price) * position_size
            capital += profit
            peak_capital = max(peak_capital, capital)
            current_drawdown = peak_capital - capital
            max_drawdown = max(max_drawdown, current_drawdown)

            trade = {'Entry': entry_price, 'Exit': exit_price, 'Profit': profit, 'Type': 'Long' if row['signal'] == 1 else 'Short', 'Entry Date': index, 'Exit Date': j if exit_price else None}
            trades.append(trade)

    win_rate = sum(1 for trade in trades if trade['Profit'] > 0) / len(trades) if trades else 0
    total_profit = sum(trade['Profit'] for trade in trades)
    return {'Trades': trades, 'Win Rate': win_rate, 'Total Profit': total_profit, 'Max Drawdown': max_drawdown}



def calculate_indicators(df):
    # Ensure you use the correct column name, which seems to be 'HA_close' based on your Heikin Ashi calculation
    df['SMA200'] = ta.sma(df['HA_close'], length=200)

    # Adjust MACD calculation to use 'HA_close'
    macd_values = ta.macd(df['HA_close'], fast=12, slow=26, signal=9)
    df['macd'] = macd_values['MACD_12_26_9']
    df['macd_signal'] = macd_values['MACDs_12_26_9']
    df['macd_hist'] = macd_values['MACDh_12_26_9']

    # Adjust SuperTrend calculation to use 'HA_high', 'HA_low', 'HA_close'
    supertrend_values = ta.supertrend(df['HA_high'], df['HA_low'], df['HA_close'], length=12, multiplier=3)
    df['SuperTrend'] = supertrend_values['SUPERT_12_3.0']

    # ATR calculation should also use 'HA_high', 'HA_low', 'HA_close'
    df['ATR'] = ta.atr(df['HA_high'], df['HA_low'], df['HA_close'], length=14)

    return df



def save_results_to_excel(results, filename='backtesting_results.xlsx'):
    """Save the backtesting results to an Excel file."""
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for (symbol, timeframe), data in results.items():
            # Replace invalid characters for Excel sheet names
            safe_sheet_name = f'{symbol.replace("/", "-")}_{timeframe}'
            df = pd.DataFrame(data)
            # Write DataFrame to an Excel sheet with a valid name
            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        # No need to call writer.save() - it's handled automatically


def save_summary_to_excel(summary, filename='backtesting_summary.xlsx'):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # Create DataFrame from summary list including the 'Trades Count'
        df_summary = pd.DataFrame(summary)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)




def main():
    results = {}
    summary = []
    leverage = 30  # Leverage factor

    for symbol in symbols:
        for timeframe in timeframes:
            df = update_data(symbol, timeframe)
            df = calculate_heikin_ashi(df)
            df = calculate_indicators(df)
            df = macd_signals(df)
            metrics = backtest_strategy(df)
            results[(symbol, timeframe)] = metrics['Trades']
            summary.append({
                'Symbol': symbol,
                'Timeframe': timeframe,
                'Win Rate': metrics['Win Rate'],
                'Total Profit': metrics['Total Profit'],
                'Max Drawdown': metrics['Max Drawdown'],
                'Trades Count': len(metrics['Trades'])  # Include count of trades here
            })

            print(f'Done backtesting for {symbol} with the {timeframe} timeframe')

    save_results_to_excel(results)
    save_summary_to_excel(summary)
    print("Backtesting completed and results are saved.")

if __name__ == "__main__":
    main()



