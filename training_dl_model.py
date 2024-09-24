import pandas as pd
import pandas_ta as ta

# Assuming these functions are defined in your local modules
from _managing_data import update_data

def calculate_macd(df):
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    return df

def find_macd_signals(df):
    signals = []
    for i in range(1, len(df)):
        if df.iloc[i]['MACD'] > df.iloc[i]['MACD_signal'] and df.iloc[i - 1]['MACD'] <= df.iloc[i - 1]['MACD_signal']:
            signals.append(df.index[i])
    return signals

def get_previous_rsi(df_higher, signal_time):
    # Find the index for the last entry before the signal time
    valid_times = df_higher[df_higher.index <= signal_time]
    if not valid_times.empty:
        last_time = valid_times.index[-1]
        return df_higher.loc[last_time]['RSI'], last_time
    return None, None

def main():
    symbol = 'BTC/USDT'
    df_15m = update_data(symbol, '15m')
    df_15m = calculate_macd(df_15m)
    df_1h = update_data(symbol, '1h')
    df_1h['RSI'] = ta.rsi(df_1h['close'], length=7)

    signals = find_macd_signals(df_15m)
    if signals:
        for signal_time in signals:
            rsi_value, last_time = get_previous_rsi(df_1h, signal_time)
            print(f"MACD signal time (15m): {signal_time}")
            print(f"Previous RSI value from 1h timeframe at {last_time}: {rsi_value}")
    else:
        print("No MACD signals found.")

if __name__ == "__main__":
    main()
