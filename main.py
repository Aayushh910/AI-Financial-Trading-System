import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from rl_agents.train_rl import train_agent
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

returns = data['Target'][split:].reset_index(drop=True)

import pandas as pd
signals = pd.Series(final_signals)

strategy_returns = returns * signals


cost = 0.001
strategy_returns = strategy_returns - (cost * (signals != 0))


import numpy as np

equity_curve = (1 + strategy_returns).cumprod()


import matplotlib.pyplot as plt

plt.plot(equity_curve, label="Strategy")
plt.title("Equity Curve")
plt.legend()
plt.show()


total_return = equity_curve.iloc[-1] - 1
sharpe = np.mean(strategy_returns) / np.std(strategy_returns)

rolling_max = equity_curve.cummax()
drawdown = equity_curve / rolling_max - 1
max_drawdown = drawdown.min()

print("Return:", total_return)
print("Sharpe:", sharpe)
print("Max Drawdown:", max_drawdown)



buy_hold = (1 + returns).cumprod()

plt.plot(equity_curve, label="Strategy")
plt.plot(buy_hold, label="Buy & Hold")
plt.legend()
plt.show()


# RL AGENT TRAINING

rl_model = train_agent(data)

print("RL Agent Training Completed")