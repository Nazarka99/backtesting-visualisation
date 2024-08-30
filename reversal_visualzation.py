"""For visualization"""

import pandas as pd
import plotly.graph_objects as go
from _managing_data import update_data
from common_functions import calculate_heikin_ashi
import pandas_ta as ta
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Fetch and process data
    df = update_data('BTC/USDT', '30m')
    df = calculate_heikin_ashi(df)

    # Calculate SuperTrend
    df[['SUPERT', 'direction', 'long', 'short']] = ta.supertrend(df.HA_high,
                                                                 df.HA_low,
                                                                 df.HA_close,
                                                                 length=12,
                                                                 multiplier=3)

    # Calculate MACD using Heikin Ashi close price
    macd = ta.macd(df['HA_close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # Generate long signal: MACD crossing above Signal line and SuperTrend above close price
    df['Long_Signal'] = ((df['MACD_12_26_9'].shift(1) < df['MACDs_12_26_9'].shift(1)) &
                         (df['MACD_12_26_9'] > df['MACDs_12_26_9']) &
                         (df['SUPERT'] > df['HA_close'])).astype(int)

    # Generate short signal: MACD crossing below Signal line and SuperTrend below close price
    df['Short_Signal'] = ((df['MACD_12_26_9'].shift(1) > df['MACDs_12_26_9'].shift(1)) &
                          (df['MACD_12_26_9'] < df['MACDs_12_26_9']) &
                          (df['SUPERT'] < df['HA_close'])).astype(int)

    # Plotting
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['HA_open'],
                                         high=df['HA_high'],
                                         low=df['HA_low'],
                                         close=df['HA_close'],
                                         name='Candles')])

    # Add SuperTrend line to the chart
    fig.add_trace(go.Scatter(x=df.index, y=df['SUPERT'],
                             mode='lines',
                             line=dict(width=1.5),
                             name='SuperTrend'))

    # Add MACD line and Signal line
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'],
                             mode='lines',
                             line=dict(width=1.5, color='blue'),
                             name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'],
                             mode='lines',
                             line=dict(width=1.5, color='orange'),
                             name='Signal Line'))

    # Visualize long and short signals
    fig.add_trace(go.Scatter(x=df.index[df['Long_Signal'] == 1],
                             y=df['HA_close'][df['Long_Signal'] == 1],
                             mode='markers',
                             marker=dict(symbol='triangle-up', color='green', size=10),
                             name='Long Signal'))
    fig.add_trace(go.Scatter(x=df.index[df['Short_Signal'] == 1],
                             y=df['HA_close'][df['Short_Signal'] == 1],
                             mode='markers',
                             marker=dict(symbol='triangle-down', color='red', size=10),
                             name='Short Signal'))

    # Update the layout to better display the chart
    fig.update_layout(
        title=f"Heikin Ashi Candles, SuperTrend, and MACD for BTC/USDT 30m",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    # Display the plot
    fig.show()

