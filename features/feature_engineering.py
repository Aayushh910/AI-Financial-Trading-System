import pandas as pd
import ta


def create_features(data):

    data = data.copy()

    for col in ["Close", "High", "Low", "Volume"]:
        data[col] = data[col].squeeze()

    # Returns
    data["Return"] = data["Close"].pct_change()

    # Volatility
    data["Volatility"] = data["Return"].rolling(20).std()

    # Lag Features
    data["Lag_1"] = data["Close"].shift(1)
    data["Lag_2"] = data["Close"].shift(2)
    data["Lag_3"] = data["Close"].shift(3)
    data["Lag_5"] = data["Close"].shift(5)

    # Momentum
    data["Momentum"] = data["Close"] - data["Close"].shift(5)

    # Rolling Std
    data["Rolling_STD"] = data["Close"].rolling(10).std()

    # RSI
    data["RSI"] = ta.momentum.RSIIndicator(
        close=data["Close"],
        window=14
    ).rsi()

    # MACD
    macd = ta.trend.MACD(close=data["Close"])

    data["MACD"] = macd.macd()
    data["MACD_SIGNAL"] = macd.macd_signal()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(
        close=data["Close"],
        window=20
    )

    data["BB_HIGH"] = bollinger.bollinger_hband()
    data["BB_LOW"] = bollinger.bollinger_lband()

    # ATR
    atr = ta.volatility.AverageTrueRange(
        high=data["High"],
        low=data["Low"],
        close=data["Close"]
    )

    data["ATR"] = atr.average_true_range()

    # Volume Features
    data["Volume_Change"] = data["Volume"].pct_change()
    data["Volume_MA"] = data["Volume"].rolling(20).mean()

    # EMA Features
    data["EMA_10"] = data["Close"].ewm(span=10).mean()
    data["EMA_20"] = data["Close"].ewm(span=20).mean()
    data["EMA_50"] = data["Close"].ewm(span=50).mean()

    # SMA Features
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()

    # Price Range
    data["Price_Range"] = data["High"] - data["Low"]

    # Target
    data["Target"] = (
        data["Close"].shift(-1) / data["Close"]
    ) - 1

    data.dropna(inplace=True)

    return data