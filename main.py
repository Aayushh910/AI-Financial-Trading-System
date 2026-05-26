import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from features.feature_engineering import create_features
from risk.volatility_model import VolatilityModel
from backtesting.strategy import apply_risk_filter
from models.portfolio_optimizer import optimize_portfolio
from models.hyperparameter_tuning import tune_model

# 1. PORTFOLIO OPTIMIZATION

assets = ["AAPL", "MSFT", "GOOG"]
data_portfolio = pd.DataFrame()

for asset in assets:
    df = yf.download(asset, start="2022-01-01")['Close']
    data_portfolio[asset] = df

returns = data_portfolio.pct_change().dropna()

weights, ret, risk = optimize_portfolio(returns)

print("Weights:", weights)
print("Return:", ret)
print("Risk:", risk)

# 2. LOAD + FEATURE ENGINEERING

data = yf.download("AAPL", start="2022-01-01")
data = create_features(data)

features = ['Lag_1','Lag_2','Momentum','Rolling_STD']
X = data[features]

# ✅ Correct target (NOT volatility)
y = data['Target']  # you created this in Day 3

# 3. TRAIN-TEST SPLIT

split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. HYPERPARAMETER TUNING

best_params = tune_model(X_train, y_train, X_test, y_test)

model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train)

preds = model.predict(X_test)

threshold = 0.0002

signals = []
for p in preds:
    if p > threshold:
        signals.append(1)
    elif p < -threshold:
        signals.append(-1)
    else:
        signals.append(0)

# 5. VOLATILITY MODEL (RISK)

y_vol = data['Volatility']

vol_model = VolatilityModel()
vol_model.train(X_train, y_vol[:split])

vol_pred = vol_model.predict(X_test)

# 6. APPLY RISK FILTER

threshold = y_vol.mean()
final_signals = apply_risk_filter(signals, vol_pred, threshold)

print(final_signals[:10])