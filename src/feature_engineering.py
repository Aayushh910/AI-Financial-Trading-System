import numpy as np

def create_features(data):

    # Returns
    data['Returns'] = (
        data['Close']
        .pct_change()
    )

    # Log Returns
    data['Log_Returns'] = np.log(
        data['Close'] /
        data['Close'].shift(1)
    )

    # Volatility
    data['Volatility'] = (
        data['Returns']
        .rolling(20)
        .std()
    )

    # Momentum
    data['Momentum'] = (
        data['Close'] -
        data['Close'].shift(10)
    )

    # Volume Change
    data['Volume_Change'] = (
        data['Volume']
        .pct_change()
    )

    # Target
    data['Target'] = (
        data['Returns']
        .shift(-1)
    )

    data = data.dropna()

    return data