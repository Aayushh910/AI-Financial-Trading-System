def create_features(data):
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(20).std()

    data['Lag_1'] = data['Close'].shift(1)
    data['Lag_2'] = data['Close'].shift(2)

    data['Momentum'] = data['Close'] - data['Close'].shift(5)
    data['Rolling_STD'] = data['Close'].rolling(10).std()

    data['Target'] = data['Close'].shift(-1) / data['Close'] - 1

    data = data.dropna()
    return data