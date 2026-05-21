import pandas as pd
import numpy as np

def add_indicators(data):

    # SMA
    data['SMA_20'] = (
        data['Close']
        .rolling(20)
        .mean()
    )

    # EMA
    data['EMA_20'] = (
        data['Close']
        .ewm(span=20)
        .mean()
    )

    # RSI
    delta = data['Close'].diff()

    gain = delta.where(
        delta > 0,
        0
    )

    loss = -delta.where(
        delta < 0,
        0
    )

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data['RSI'] = (
        100 - (100 / (1 + rs))
    )

    # Bollinger Bands
    rolling_mean = (
        data['Close']
        .rolling(20)
        .mean()
    )

    rolling_std = (
        data['Close']
        .rolling(20)
        .std()
    )

    data['BB_upper'] = (
        rolling_mean + 2 * rolling_std
    )

    data['BB_lower'] = (
        rolling_mean - 2 * rolling_std
    )

    return data