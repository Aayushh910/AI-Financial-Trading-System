import yfinance as yf
from features.feature_engineering import create_features
from risk.volatility_model import VolatilityModel
from backtesting.strategy import apply_risk_filter

import pandas as pd

assets = ["AAPL", "MSFT", "GOOG"]

data = pd.DataFrame()

for asset in assets:
    df = yf.download(asset, start="2022-01-01")['Close']
    data[asset] = df

returns = data.pct_change().dropna()

weights, ret, risk = optimize_portfolio(returns)

print("Weights:", weights)
print("Return:", ret)
print("Risk:", risk)


# Load data
data = yf.download("AAPL", start="2022-01-01")

# Features
data = create_features(data)

features = ['Lag_1','Lag_2','Momentum','Rolling_STD']
X = data[features]
y_vol = data['Volatility']

# Split
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_vol[:split], y_vol[split:]

# Train volatility model
vol_model = VolatilityModel()
vol_model.train(X_train, y_train)

# Predict volatility
vol_pred = vol_model.predict(X_test)

# Dummy signals (from your ML model)
signals = [1 if i % 2 == 0 else -1 for i in range(len(vol_pred))]

# Apply risk filter
threshold = y_vol.mean()
final_signals = apply_risk_filter(signals, vol_pred, threshold)

print(final_signals[:10])