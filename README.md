# CoinSpikeForecaster

**CoinSpikeForecaster** is a Python tool designed to predict "price spikes" — rapid, significant price movements in cryptocurrencies. It leverages market data from Coinbase, technical indicators (MACD, OBV), and social media activity (e.g., X posts) to forecast spikes using a CNN-LSTM neural network. Results are visualized with interactive Bokeh plots, enabling traders and analysts to identify potential opportunities.

## Features
- Fetches real-time OHLCV data from Coinbase.
- Incorporates social media activity for enhanced predictions.
- Uses a CNN-LSTM model to detect price spikes.
- Generates interactive visualizations with Bokeh.
- Configurable for various symbols and timeframes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CoinSpikeForecaster.git
   cd CoinSpikeForecaster
