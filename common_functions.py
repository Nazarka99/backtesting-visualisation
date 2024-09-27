import pytz
import pandas_ta as ta

def convert_timestamps(data):
    data.index = data.index.tz_localize(pytz.utc).tz_convert(pytz.timezone('Etc/GMT-2'))
    return data


# Calculate Heikin Ashi candles
def calculate_heikin_ashi(data):
    data['HA_close'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    data['HA_open'] = (data['open'].shift(1) + data['close'].shift(1)) / 2
    data['HA_open'].iloc[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2
    data['HA_high'] = data[['high', 'HA_open', 'HA_close']].max(axis=1)
    data['HA_low'] = data[['low', 'HA_open', 'HA_close']].min(axis=1)
    return data

# Function to calculate SuperTrend using pandas_ta
def calculate_supertrend(df, period=12, multiplier=3):
    df['HA_hl2'] = (df['HA_high'] + df['HA_low']) / 2  # Median price needed for SuperTrend
    supertrend = ta.supertrend(high=df['HA_high'], low=df['HA_low'], close=df['HA_close'], length=period, multiplier=multiplier)
    # Ensure we access the correct columns, check available columns first
    # print(supertrend.columns)  # Uncomment this to see all column names available

    df['supertrend'] = supertrend['SUPERT_12_3.0']
    return df


# Function to calculate MACD
def calculate_macd(df, slow=26, fast=12, signal=9):
    # Calculate MACD using pandas_ta by applying it to the DataFrame directly
    macd = df.ta.macd(close='HA_close', fast=fast, slow=slow, signal=signal)

    # Use the correct column names as shown in the output
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']

    return df

def macd_signals(df):
    """Generate potential trading opportunities based on MACD crossover."""
    df['potential_signal'] = 0
    for i in range(1, len(df)):
        if df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = 1  # Potential buy
        elif df['macd'].iloc[i] < df['macd_signal'].iloc[i]:
            df.loc[df.index[i], 'potential_signal'] = -1  # Potential sell
    return df

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