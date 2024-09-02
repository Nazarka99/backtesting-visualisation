import pandas as pd
from _managing_data import update_data
from common_functions import calculate_heikin_ashi
import pandas_ta as ta
import datetime

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

def calculate_indicators(df):
    """Calculate SuperTrend, MACD, ATR, RSI, and Kaufman Efficiency Ratio indicators."""
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

def calculate_kaufman_efficiency_ratio(data_series, period):
    """Calculate the Kaufman Efficiency Ratio (KER) for a given period."""
    signal = data_series.diff(period).abs()  # Signal = |Data_t - Data_(t-N)|
    noise = data_series.diff().abs().rolling(window=period).sum()  # Noise = Sum of absolute differences over the period
    ker = signal / noise  # Kaufman Efficiency Ratio
    return ker

def align_higher_timeframe_data(df, higher_df):
    """Align higher timeframe data with the lower timeframe data."""
    higher_df = higher_df.resample('1T').ffill().reindex(df.index).ffill()  # Resample to match the lower timeframe
    return higher_df

def generate_signals(df):
    """Generate long and short signals based on MACD crossing and SuperTrend position."""
    df['Long_Signal'] = ((df['MACD_12_26_9'].shift(1) < df['MACDs_12_26_9'].shift(1)) &
                         (df['MACD_12_26_9'] > df['MACDs_12_26_9']) &
                         (df['SUPERT'] > df['HA_close'])).astype(int)

    df['Short_Signal'] = ((df['MACD_12_26_9'].shift(1) > df['MACDs_12_26_9'].shift(1)) &
                          (df['MACD_12_26_9'] < df['MACDs_12_26_9']) &
                          (df['SUPERT'] < df['HA_close'])).astype(int)

    return df

def calculate_tp_sl(df):
    """Calculate Take Profit (TP) and Stop Loss (SL) levels."""
    df['Long_TP'] = df['SUPERT'] - 0.5 * df['ATR']
    df['Long_SL'] = df['HA_low'].rolling(window=10).min() - 0.5 * df['ATR']
    df['Short_TP'] = df['SUPERT'] + 0.5 * df['ATR']
    df['Short_SL'] = df['HA_high'].rolling(window=10).max() + 0.5 * df['ATR']
    return df

def calculate_position_size(equity, risk_per_trade, entry_price, stop_loss_price):
    """Calculate the position size based on the risk per trade."""
    risk_amount = equity * risk_per_trade
    position_size = risk_amount / abs(entry_price - stop_loss_price)
    return position_size

def backtest(df, higher_df, symbol, timeframe, initial_deposit=10000, risk_per_trade=0.05):
    """Perform backtesting for the given dataframe, symbol, and timeframe."""
    equity = initial_deposit
    trades = []
    max_drawdown = 0
    current_drawdown = 0
    wins = 0
    losses = 0
    long_position_open = False
    short_position_open = False

    for i in range(1, len(df)):

        current_time = df.index[i]
        weekday = current_time.weekday()  # Monday is 0, Sunday is 6

        # Skip if it's Saturday or Sunday
        if weekday in [5, 6]:
            continue

        # Check for a long trade entry
        if df['Long_Signal'].iloc[i] == 1 and not long_position_open:
            # Ensure that the raw price is between the raw open and close
            if df['open'].iloc[i] <= df['close'].iloc[i]:
                entry_price = df['close'].iloc[i]
            else:
                entry_price = df['open'].iloc[i]

            if df['open'].iloc[i] <= entry_price <= df['close'].iloc[i] or df['close'].iloc[i] <= entry_price <= df['open'].iloc[i]:
                tp_price = df['Long_TP'].iloc[i]
                sl_price = df['Long_SL'].iloc[i]
                position_size = calculate_position_size(equity, risk_per_trade, entry_price, sl_price)
                trade_time = df.index[i]  # Capture the trade's timestamp
                rsi_7 = df['RSI_7'].iloc[i]
                rsi_14 = df['RSI_14'].iloc[i]
                rsi_21 = df['RSI_21'].iloc[i]
                ker_rsi_7 = df['KER_RSI_7'].iloc[i]
                ker_rsi_14 = df['KER_RSI_14'].iloc[i]
                ker_rsi_21 = df['KER_RSI_21'].iloc[i]
                macd_line = df['MACD_12_26_9'].iloc[i]
                signal_line = df['MACDs_12_26_9'].iloc[i]
                macd_histogram = df['MACDh_12_26_9'].iloc[i]

                # Higher timeframe data
                higher_rsi_7 = higher_df['RSI_7'].iloc[i]
                higher_rsi_14 = higher_df['RSI_14'].iloc[i]
                higher_rsi_21 = higher_df['RSI_21'].iloc[i]
                higher_ker_rsi_7 = higher_df['KER_RSI_7'].iloc[i]
                higher_ker_rsi_14 = higher_df['KER_RSI_14'].iloc[i]
                higher_ker_rsi_21 = higher_df['KER_RSI_21'].iloc[i]
                higher_macd_line = higher_df['MACD_12_26_9'].iloc[i]
                higher_signal_line = higher_df['MACDs_12_26_9'].iloc[i]
                higher_macd_histogram = higher_df['MACDh_12_26_9'].iloc[i]

                long_position_open = True

                for j in range(i + 1, len(df)):
                    if df['low'].iloc[j] <= sl_price:
                        trade_profit = (sl_price - entry_price) * position_size
                        result = 'Loss' if trade_profit < 0 else 'Win'
                        trades.append({'Symbol': symbol, 'Timeframe': timeframe, 'Type': 'Long',
                                       'Open_Time': trade_time,  # Add opening time
                                       'Result': result, 'RSI_7': rsi_7, 'RSI_14': rsi_14, 'RSI_21': rsi_21,
                                       'KER_RSI_7': ker_rsi_7, 'KER_RSI_14': ker_rsi_14, 'KER_RSI_21': ker_rsi_21,
                                       'MACD_Line': macd_line, 'Signal_Line': signal_line, 'MACD_Histogram': macd_histogram,
                                       'Higher_RSI_7': higher_rsi_7, 'Higher_RSI_14': higher_rsi_14, 'Higher_RSI_21': higher_rsi_21,
                                       'Higher_KER_RSI_7': higher_ker_rsi_7, 'Higher_KER_RSI_14': higher_ker_rsi_14, 'Higher_KER_RSI_21': higher_ker_rsi_21,
                                       'Higher_MACD_Line': higher_macd_line, 'Higher_Signal_Line': higher_signal_line, 'Higher_MACD_Histogram': higher_macd_histogram})
                        equity += trade_profit
                        losses += 1
                        current_drawdown += trade_profit
                        long_position_open = False  # Close the long position
                        break
                    elif df['high'].iloc[j] >= tp_price:
                        trade_profit = (tp_price - entry_price) * position_size
                        result = 'Loss' if trade_profit < 0 else 'Win'
                        trades.append({'Symbol': symbol, 'Timeframe': timeframe, 'Type': 'Long',
                                       'Open_Time': trade_time,  # Add opening time
                                       'Result': result, 'RSI_7': rsi_7, 'RSI_14': rsi_14, 'RSI_21': rsi_21,
                                       'KER_RSI_7': ker_rsi_7, 'KER_RSI_14': ker_rsi_14, 'KER_RSI_21': ker_rsi_21,
                                       'MACD_Line': macd_line, 'Signal_Line': signal_line, 'MACD_Histogram': macd_histogram,
                                       'Higher_RSI_7': higher_rsi_7, 'Higher_RSI_14': higher_rsi_14, 'Higher_RSI_21': higher_rsi_21,
                                       'Higher_KER_RSI_7': higher_ker_rsi_7, 'Higher_KER_RSI_14': higher_ker_rsi_14, 'Higher_KER_RSI_21': higher_ker_rsi_21,
                                       'Higher_MACD_Line': higher_macd_line, 'Higher_Signal_Line': higher_signal_line, 'Higher_MACD_Histogram': higher_macd_histogram})
                        equity += trade_profit
                        wins += 1
                        current_drawdown = max(0, current_drawdown + trade_profit)
                        long_position_open = False  # Close the long position
                        break

        # Check for a short trade entry
        elif df['Short_Signal'].iloc[i] == 1 and not short_position_open:
            # Ensure that the raw price is between the raw open and close
            if df['open'].iloc[i] >= df['close'].iloc[i]:
                entry_price = df['close'].iloc[i]
            else:
                entry_price = df['open'].iloc[i]

            if df['open'].iloc[i] >= entry_price >= df['close'].iloc[i] or df['close'].iloc[i] >= entry_price >= df['open'].iloc[i]:
                tp_price = df['Short_TP'].iloc[i]
                sl_price = df['Short_SL'].iloc[i]
                position_size = calculate_position_size(equity, risk_per_trade, entry_price, sl_price)
                trade_time = df.index[i]  # Capture the trade's timestamp
                rsi_7 = df['RSI_7'].iloc[i]
                rsi_14 = df['RSI_14'].iloc[i]
                rsi_21 = df['RSI_21'].iloc[i]
                ker_rsi_7 = df['KER_RSI_7'].iloc[i]
                ker_rsi_14 = df['KER_RSI_14'].iloc[i]
                ker_rsi_21 = df['KER_RSI_21'].iloc[i]
                macd_line = df['MACD_12_26_9'].iloc[i]
                signal_line = df['MACDs_12_26_9'].iloc[i]
                macd_histogram = df['MACDh_12_26_9'].iloc[i]

                # Higher timeframe data
                higher_rsi_7 = higher_df['RSI_7'].iloc[i]
                higher_rsi_14 = higher_df['RSI_14'].iloc[i]
                higher_rsi_21 = higher_df['RSI_21'].iloc[i]
                higher_ker_rsi_7 = higher_df['KER_RSI_7'].iloc[i]
                higher_ker_rsi_14 = higher_df['KER_RSI_14'].iloc[i]
                higher_ker_rsi_21 = higher_df['KER_RSI_21'].iloc[i]
                higher_macd_line = higher_df['MACD_12_26_9'].iloc[i]
                higher_signal_line = higher_df['MACDs_12_26_9'].iloc[i]
                higher_macd_histogram = higher_df['MACDh_12_26_9'].iloc[i]

                short_position_open = True

                for j in range(i + 1, len(df)):
                    if df['high'].iloc[j] >= sl_price:
                        trade_profit = (entry_price - sl_price) * position_size
                        result = 'Loss' if trade_profit < 0 else 'Win'
                        trades.append({'Symbol': symbol, 'Timeframe': timeframe, 'Type': 'Short',
                                       'Open_Time': trade_time,  # Add opening time
                                       'Result': result, 'RSI_7': rsi_7, 'RSI_14': rsi_14, 'RSI_21': rsi_21,
                                       'KER_RSI_7': ker_rsi_7, 'KER_RSI_14': ker_rsi_14, 'KER_RSI_21': ker_rsi_21,
                                       'MACD_Line': macd_line, 'Signal_Line': signal_line, 'MACD_Histogram': macd_histogram,
                                       'Higher_RSI_7': higher_rsi_7, 'Higher_RSI_14': higher_rsi_14, 'Higher_RSI_21': higher_rsi_21,
                                       'Higher_KER_RSI_7': higher_ker_rsi_7, 'Higher_KER_RSI_14': higher_ker_rsi_14, 'Higher_KER_RSI_21': higher_ker_rsi_21,
                                       'Higher_MACD_Line': higher_macd_line, 'Higher_Signal_Line': higher_signal_line, 'Higher_MACD_Histogram': higher_macd_histogram})
                        equity += trade_profit
                        losses += 1
                        current_drawdown += trade_profit
                        short_position_open = False  # Close the short position
                        break
                    elif df['low'].iloc[j] <= tp_price:
                        trade_profit = (entry_price - tp_price) * position_size
                        result = 'Loss' if trade_profit < 0 else 'Win'
                        trades.append({'Symbol': symbol, 'Timeframe': timeframe, 'Type': 'Short',
                                       'Open_Time': trade_time,  # Add opening time
                                       'Result': result, 'RSI_7': rsi_7, 'RSI_14': rsi_14, 'RSI_21': rsi_21,
                                       'KER_RSI_7': ker_rsi_7, 'KER_RSI_14': ker_rsi_14, 'KER_RSI_21': ker_rsi_21,
                                       'MACD_Line': macd_line, 'Signal_Line': signal_line, 'MACD_Histogram': macd_histogram,
                                       'Higher_RSI_7': higher_rsi_7, 'Higher_RSI_14': higher_rsi_14, 'Higher_RSI_21': higher_rsi_21,
                                       'Higher_KER_RSI_7': higher_ker_rsi_7, 'Higher_KER_RSI_14': higher_ker_rsi_14, 'Higher_KER_RSI_21': higher_ker_rsi_21,
                                       'Higher_MACD_Line': higher_macd_line, 'Higher_Signal_Line': higher_signal_line, 'Higher_MACD_Histogram': higher_macd_histogram})
                        equity += trade_profit
                        wins += 1
                        current_drawdown = max(0, current_drawdown + trade_profit)
                        short_position_open = False  # Close the short position
                        break

        max_drawdown = min(max_drawdown, current_drawdown)

    total_trades = len(trades)
    net_profit = equity - initial_deposit
    win_rate = wins / total_trades if total_trades > 0 else 0

    # Return the summary and trade details
    summary = {
        'Symbol': symbol,
        'Timeframe': timeframe,
        'Risk Management': '5% Equity Risk Per Trade',
        'Total Trades': total_trades,
        'Net Profit': net_profit,
        'Win Rate': win_rate,
        'Max Drawdown': max_drawdown
    }

    return summary, trades

def save_results_to_excel(summary_results, trades_results, filename='backtest_results_multi_symbol.xlsx'):
    """Save the summary and trades results to an Excel file."""
    summary_df = pd.DataFrame(summary_results)
    trades_df = pd.DataFrame(trades_results)

    # Drop the 'Entry', 'Exit', and 'Profit' columns from the trades DataFrame before saving
    trades_df = trades_df.drop(columns=['Entry', 'Exit', 'Profit'], errors='ignore')

    # Remove rows with any empty cells in the trades DataFrame
    trades_df = trades_df.dropna(how='any').reset_index(drop=True)

    with pd.ExcelWriter(filename) as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        trades_df.to_excel(writer, sheet_name='Trades', index=False)

if __name__ == "__main__":

    # Symbols and timeframes
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
               'DOGE/USDT', 'MATIC/USDT', 'DOT/USDT', 'LINK/USDT', 'IMX/USDT', 'ICP/USDT']
    timeframes = ['15m', '30m', '1h', '2h', '4h']

    # Symbols and timeframes for fast test
    # symbols = ['BTC/USDT', 'ETH/USDT']
    # timeframes = ['15m', '30m']

    # Initialize lists to store results
    summary_results = []
    trades_results = []

    for symbol in symbols:
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
                df = calculate_tp_sl(df)

                # Backtest for this symbol and timeframe
                summary, trades = backtest(df, higher_df, symbol, timeframe, initial_deposit=10000, risk_per_trade=0.05)

                # Store results
                summary_results.append(summary)
                trades_results.extend(trades)
            print(f"Done for {symbol} with {timeframe}")

    # Save the results to an Excel file
    save_results_to_excel(summary_results, trades_results)
