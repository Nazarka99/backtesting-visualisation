import pandas as pd
from _managing_data import update_data


def run_correlation_analysis():
    """
    Runs a correlation analysis across multiple cryptocurrencies and timeframes.
    Saves the resulting correlation matrices to an Excel file.
    """
    # List of symbols and timeframes to analyze
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
               'DOGE/USDT', 'MATIC/USDT', 'DOT/USDT', 'LINK/USDT', 'IMX/USDT', 'ICP/USDT']
    timeframes = ['15m', '30m', '1h', '2h', '4h']

    data_dict = {}  # Dictionary to store DataFrame for each timeframe
    for timeframe in timeframes:
        temp_data = {}  # Temporary dictionary to store close prices of each symbol
        for symbol in symbols:
            try:
                df = update_data(symbol, timeframe)
                temp_data[symbol] = df['close']  # Store only the close prices
            except Exception as e:
                print(f"Error fetching data for {symbol} on {timeframe}: {e}")
        if temp_data:  # Check if there is any data collected
            combined_df = pd.concat(temp_data.values(), axis=1, keys=temp_data.keys())
            data_dict[timeframe] = combined_df

    # Write the correlation matrices to an Excel file
    with pd.ExcelWriter('correlation_matrices.xlsx') as writer:
        for timeframe, data in data_dict.items():
            if not data.empty:  # Check if the DataFrame is not empty
                correlation_matrix = data.corr()  # Compute the correlation matrix
                print(f"Correlation matrix for timeframe {timeframe}:")
                print(correlation_matrix)
                correlation_matrix.to_excel(writer,
                                            sheet_name=timeframe)  # Save each correlation matrix in a separate sheet

    print("All correlation matrices have been saved to 'correlation_matrices.xlsx'.")


if __name__ == '__main__':
    run_correlation_analysis()  # Execute the function if the script is run as the main program
