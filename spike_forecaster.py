import ccxt
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource
from datetime import datetime, timedelta
import os

class CoinSpikeForecaster:
    def __init__(self, symbol='LTC/USD', timeframe='1h', lookback_days=14, api_key=None, api_secret=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.exchange = ccxt.coinbase({'apiKey': api_key, 'secret': api_secret})
        self.scaler = MinMaxScaler()

    def fetch_ohlcv(self):
        """Получение исторических данных с Coinbase."""
        since = int((datetime.now() - timedelta(days=self.lookback_days)).timestamp() * 1000)
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_social_activity(self):
        """Получение активности в X (заглушка для API X)."""
        # Реальная версия требует API X для анализа активности
        return np.random.rand(len(self.fetch_ohlcv())) * 75

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Расчет MACD."""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def prepare_data(self, df):
        """Подготовка данных для модели."""
        df['returns'] = df['close'].pct_change()
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['social_activity'] = self.fetch_social_activity()
        features = df[['close', 'macd', 'obv', 'social_activity']].dropna()

        scaled_data = self.scaler.fit_transform(features)
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(1 if scaled_data[i, 0] > np.percentile(scaled_data[:, 0], 90) else 0)  # Ценовой скачок
        return np.array(X), np.array(y)

    def build_model(self):
        """Создание CNN-LSTM модели."""
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(60, 4)),
            LSTM(48, return_sequences=True),
            Dropout(0.25),
            LSTM(24),
            Dropout(0.25),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X, y):
        """Обучение модели."""
        model = self.build_model()
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        return model

    def predict_spike(self, model, X):
        """Прогноз ценовых скачков."""
        predictions = model.predict(X)
        return (predictions > 0.5).astype(int)

    def visualize_results(self, df, predictions):
        """Визуализация с Bokeh."""
        df = df.iloc[60:].copy()
        df['spike_prediction'] = predictions
        source = ColumnDataSource(df)

        p = figure(title=f'Price Spikes for {self.symbol}', x_axis_type='datetime', width=800, height=400)
        p.line('timestamp', 'close', source=source, legend_label='Price', color='blue')
        p.circle('timestamp', 'close', source=source[source['spike_prediction'] == 1],
                 size=10, color='red', legend_label='Predicted Spike')

        output_file('data/sample_output/spike_forecast.html')
        save(p)

    def run(self):
        """Основной метод анализа."""
        df = self.fetch_ohlcv()
        X, y = self.prepare_data(df)
        model = self.train_model(X, y)
        predictions = self.predict_spike(model, X)
        self.visualize_results(df, predictions)
        print(f"Price spikes predicted: {np.sum(predictions)} out of {len(predictions)} periods.")

if __name__ == "__main__":
    forecaster = CoinSpikeForecaster(symbol='LTC/USD', timeframe='1h', lookback_days=14)
    forecaster.run()
