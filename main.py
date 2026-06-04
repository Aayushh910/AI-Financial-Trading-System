import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from features.feature_engineering import create_features
from risk.volatility_model import VolatilityModel
from backtesting.strategy import apply_risk_filter
from models.portfolio_optimizer import optimize_portfolio

from rl_agents.train_rl import train_agent
from rl_agents.evaluate_rl import evaluate_agent


# Portfolio Optimization

assets = ["AAPL", "MSFT", "GOOG"]

data_portfolio = pd.DataFrame()

for asset in assets:
    df = yf.download(asset, start="2022-01-01")["Close"]
    data_portfolio[asset] = df

returns = data_portfolio.pct_change().dropna()

weights, ret, risk = optimize_portfolio(returns)

print("Weights:", weights)
print("Return:", ret)
print("Risk:", risk)


# Data Loading

data = yf.download(
    "AAPL",
    start="2022-01-01",
    auto_adjust=False
)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = create_features(data)


# Features

features = [
    "Lag_1",
    "Lag_2",
    "Momentum",
    "Rolling_STD",
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "BB_HIGH",
    "BB_LOW",
    "ATR",
    "Volume_Change"
]

X = data[features]
y = data["Target"]


# Train/Test Split

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

joblib.dump(scaler, "scaler.pkl")


# XGBoost Model

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)


# Feature Importance

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance\n")
print(importance)

plt.figure(figsize=(10, 5))
plt.bar(
    importance["Feature"],
    importance["Importance"]
)
plt.xticks(rotation=45)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()


# Signal Generation

threshold = 0.0002

signals = []

for p in preds:
    if p > threshold:
        signals.append(1)
    elif p < -threshold:
        signals.append(-1)
    else:
        signals.append(0)


# Volatility Model

y_vol = data["Volatility"]

vol_model = VolatilityModel()

vol_model.train(
    X_train,
    y_vol[:split]
)

vol_pred = vol_model.predict(X_test)


# Risk Filter

vol_threshold = y_vol.mean()

final_signals = apply_risk_filter(
    signals,
    vol_pred,
    vol_threshold
)

print("\nSignals:")
print(final_signals[:10])


# Backtesting

returns = data["Target"][split:].reset_index(drop=True)

signals = pd.Series(final_signals)

strategy_returns = returns * signals

transaction_cost = 0.001

strategy_returns = (
    strategy_returns
    - transaction_cost * (signals != 0)
)

equity_curve = (
    1 + strategy_returns
).cumprod()

plt.figure(figsize=(8, 5))
plt.plot(
    equity_curve,
    label="Strategy"
)
plt.title("Equity Curve")
plt.legend()
plt.show()


total_return = equity_curve.iloc[-1] - 1

sharpe = (
    np.mean(strategy_returns)
    / np.std(strategy_returns)
)

rolling_max = equity_curve.cummax()

drawdown = (
    equity_curve / rolling_max
) - 1

max_drawdown = drawdown.min()

print("\n===== Backtest Results =====")
print("Return:", total_return)
print("Sharpe:", sharpe)
print("Max Drawdown:", max_drawdown)


# Buy & Hold Comparison

buy_hold = (
    1 + returns
).cumprod()

plt.figure(figsize=(8, 5))
plt.plot(
    equity_curve,
    label="Strategy"
)

plt.plot(
    buy_hold,
    label="Buy & Hold"
)

plt.legend()
plt.title("Strategy vs Buy & Hold")
plt.show()


# RL Training

rl_model = train_agent(data)

print("\nRL Agent Training Completed")


# RL Evaluation

evaluate_agent(data)