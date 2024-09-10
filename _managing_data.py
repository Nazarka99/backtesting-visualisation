import ccxt
import pandas as pd
import os
from time import sleep
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(symbol, timeframe, limit=1000):
    """Fetches OHLCV data for a given cryptocurrency symbol and timeframe."""
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        else:
            logging.warning(f"No data returned for {symbol} on {timeframe}.")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching data for {symbol} on {timeframe}: {e}")
        return pd.DataFrame()

def update_data(symbol, timeframe):
    """Updates or creates an Excel file with OHLCV data for a given symbol and timeframe."""
    filename = f"__data/{symbol.replace('/', '')}_{timeframe}.xlsx"
    os.makedirs('__data', exist_ok=True)  # Ensure the directory exists

    try:
        if os.path.exists(filename):
            existing_df = pd.read_excel(filename, index_col='timestamp')
            existing_df.index = pd.to_datetime(existing_df.index)
            latest_timestamp = existing_df.index.max().to_pydatetime()
            new_data = fetch_data(symbol, timeframe, limit=1000)
            if not new_data.empty:
                new_data = new_data[new_data.index > latest_timestamp]
                updated_df = pd.concat([existing_df, new_data])
            else:
                updated_df = existing_df
        else:
            updated_df = fetch_data(symbol, timeframe, limit=1000)

        if not updated_df.empty:
            updated_df.to_excel(filename)
            logging.info(f"Data for {symbol} on {timeframe} saved/updated successfully.")
        return updated_df

    except Exception as e:
        logging.error(f"Failed to update data for {symbol} on {timeframe}: {e}")
        return None

if __name__ == "__main__":
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT',
               'DOT/USDT', 'LINK/USDT', 'IMX/USDT', 'ICP/USDT']
    timeframes = ['15m', '30m', '1h', '2h', '4h']
    # timeframes = ['8h']

    for symbol in symbols:
        for timeframe in timeframes:
            update_data(symbol, timeframe)