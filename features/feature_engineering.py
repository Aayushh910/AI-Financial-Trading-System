import pandas as pd
import ta

def create_features(data):

    data = data.copy()

    # Ensure columns are proper Series
    for col in ['Close', 'High', 'Low', 'Volume']:
        data[col] = data[col].squeeze()

    # Basic Returns
    data['Return'] = data['Close'].pct_change()

    # Volatility
    data['Volatility'] = data['Return'].rolling(20).std()

    # Lag Features
    data['Lag_1'] = data['Close'].shift(1)
    data['Lag_2'] = data['Close'].shift(2)

    # Momentum
    data['Momentum'] = data['Close'] - data['Close'].shift(5)

    # Rolling STD
    data['Rolling_STD'] = data['Close'].rolling(10).std()

    # RSI
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'],window=14).rsi()

    # MACD
    macd = ta.trend.MACD(close=data['Close'])

    data['MACD'] = macd.macd()
    data['MACD_SIGNAL'] = macd.macd_signal()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=data['Close'],window=20)

    data['BB_HIGH'] = bollinger.bollinger_hband()
    data['BB_LOW'] = bollinger.bollinger_lband()

    # ATR (Average True Range)
    atr = ta.volatility.AverageTrueRange(
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )

    data['ATR'] = atr.average_true_range()

    # Volume Features
    data['Volume_Change'] = data['Volume'].pct_change()

    # Target
    data['Target'] = (data['Close'].shift(-1) / data['Close']) - 1
    data = data.dropna()

    return data