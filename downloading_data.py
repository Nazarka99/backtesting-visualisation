import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta

"""This script is for downloading data for the past year (365 days)"""

# Initialize the exchange
exchange = ccxt.binance()

# Define symbols and timeframes
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT',
               'DOT/USDT', 'LINK/USDT', 'IMX/USDT', 'ICP/USDT']

timeframes = ['15m', '30m', '1h', '2h', '4h', '1d']

# symbols = ['BTC/USDT']
# timeframes = ['1d']

# Ensure the data directory exists
data_directory = "_data"
os.makedirs(data_directory, exist_ok=True)

def fetch_and_save_data(symbol, timeframe):
    # Calculate the end time for data (current time)
    end_time = datetime.utcnow()
    # Calculate the start time for data (365 days ago)
    start_time = end_time - timedelta(days=45)

    all_data = []

    while start_time < end_time:
        # Fetch the data
        limit = 1000
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=int(start_time.timestamp() * 1000), limit=limit)
        if not ohlcv:
            break
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # Remove the last row to avoid using an unclosed candle
        df = df[:-1]
        all_data.append(df)

        # Update the start time to the last timestamp plus the interval
        if not df.empty:
            last_time = df.index.max()
            start_time = last_time + timedelta(minutes=exchange.parse_timeframe(timeframe))
        else:
            break

    # Concatenate all data into a single DataFrame
    if all_data:
        full_data = pd.concat(all_data)
        # Save to Excel
        file_path = os.path.join(data_directory, f"{symbol.replace('/', '')}_{timeframe}.xlsx")
        full_data.to_excel(file_path)
        print(f"Data for {symbol} at {timeframe} saved to {file_path}")

# Loop through each symbol and timeframe
for symbol in symbols:
    for timeframe in timeframes:
        fetch_and_save_data(symbol, timeframe)

print("All data has been downloaded and saved.")
