import os
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from features.feature_engineering import create_features
from risk.volatility_model import VolatilityModel
from backtesting.strategy import apply_risk_filter
from models.portfolio_optimizer import optimize_portfolio

from rl_agents.train_rl import train_agent
from rl_agents.evaluate_rl import evaluate_agent

from validation.walk_forward import walk_forward_validation

os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# ==========================
# Portfolio Optimization
# ==========================

assets = ["AAPL", "MSFT", "GOOG"]

data_portfolio = pd.DataFrame()

for asset in assets:
    df = yf.download(
        asset,
        start="2015-01-01"
    )["Close"]

    data_portfolio[asset] = df

returns = data_portfolio.pct_change().dropna()

weights, ret, risk = optimize_portfolio(returns)

weights_df = pd.DataFrame({
    "Asset": assets,
    "Weight": weights
})

weights_df.to_csv(
    "outputs/reports/portfolio_weights.csv",
    index=False
)

print("Weights:", weights)
print("Return:", ret)
print("Risk:", risk)

# ==========================
# Load Data
# ==========================

data = yf.download(
    "AAPL",
    start="2015-01-01",
    auto_adjust=False
)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = create_features(data)


# ==========================
# Features
# ==========================

features = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_5",
    "Momentum",
    "Rolling_STD",
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "BB_HIGH",
    "BB_LOW",
    "ATR",
    "Volume_Change",
    "EMA_10",
    "EMA_20",
    "EMA_50",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "Price_Range",
    "Volume_MA"
]

wf_results = walk_forward_validation(
    data,
    features
)

X = data[features]
y = data["Target"]


# ==========================
# Train Test Split
# ==========================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


# ==========================
# Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(
    X_train,
    columns=features
)

X_test = pd.DataFrame(
    X_test,
    columns=features
)

os.makedirs("saved_models", exist_ok=True)

joblib.dump(
    scaler,
    "saved_models/scaler.pkl"
)


# ==========================
# XGBoost Model
# ==========================

model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

pred_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": preds
})

pred_df.to_csv(
    "outputs/predictions/predictions.csv",
    index=False
)

joblib.dump(
    model,
    "saved_models/xgb_model.pkl"
)


# ==========================
# Metrics
# ==========================

mae = mean_absolute_error(
    y_test,
    preds
)

mse = mean_squared_error(
    y_test,
    preds
)

print("\n===== Model Metrics =====")
print("MAE:", mae)
print("MSE:", mse)


# ==========================
# Feature Importance
# ==========================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== Feature Importance =====")
print(importance)

plt.figure(figsize=(12, 6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=45)

plt.title(
    "XGBoost Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "outputs/charts/feature_importance.png",
    bbox_inches="tight"
)
plt.show()
plt.close()


# ==========================
# Signal Generation
# ==========================

buy_threshold = np.percentile(
    preds,
    75
)

sell_threshold = np.percentile(
    preds,
    25
)

signals = []

for p in preds:

    if p > buy_threshold:
        signals.append(1)

    elif p < sell_threshold:
        signals.append(-1)

    else:
        signals.append(0)


# ==========================
# Volatility Model
# ==========================

y_vol = data["Volatility"]

vol_model = VolatilityModel()

vol_model.train(
    X_train,
    y_vol[:split]
)

vol_pred = vol_model.predict(
    X_test
)


# ==========================
# Risk Filter
# ==========================

vol_threshold = y_vol.quantile(
    0.60
)

final_signals = apply_risk_filter(
    signals,
    vol_pred,
    vol_threshold
)

print("\nSignals:")
print(final_signals[:20])


# ==========================
# Backtesting
# ==========================

returns = data["Target"][split:].reset_index(
    drop=True
)

signals = pd.Series(final_signals)

strategy_returns = (
    returns * signals
)

transaction_cost = 0.001

strategy_returns = (
    strategy_returns
    - transaction_cost * (signals != 0)
)

equity_curve = (
    1 + strategy_returns
).cumprod()

plt.figure(figsize=(10, 5))

plt.plot(
    equity_curve,
    label="Strategy"
)

plt.title("Equity Curve")

plt.legend()
plt.savefig(
    "outputs/charts/equity_curve.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

gross_profit = strategy_returns[
    strategy_returns > 0
].sum()

gross_loss = abs(
    strategy_returns[
        strategy_returns < 0
    ].sum()
)

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0
    else 0
)

total_return = (
    equity_curve.iloc[-1] - 1
)

sharpe = (
    np.mean(strategy_returns)
    / (np.std(strategy_returns) + 1e-8)
)

rolling_max = (
    equity_curve.cummax()
)

drawdown = (
    equity_curve / rolling_max
) - 1

max_drawdown = (
    drawdown.min()
)

years = len(strategy_returns) / 252

cagr = (
    equity_curve.iloc[-1]
) ** (1 / years) - 1

downside_returns = strategy_returns[
    strategy_returns < 0
]

downside_std = (
    downside_returns.std()
    + 1e-8
)

sortino = (
    strategy_returns.mean()
    / downside_std
)


trade_returns = strategy_returns[
    signals != 0
]

total_trades = len(trade_returns)

winning_trades = len(
    trade_returns[
        trade_returns > 0
    ]
)

losing_trades = len(
    trade_returns[
        trade_returns < 0
    ]
)

win_rate = (
    winning_trades
    / total_trades
    * 100
    if total_trades > 0
    else 0
)

avg_trade_return = (
    trade_returns.mean()
    if total_trades > 0
    else 0
)


print("\n===== Backtest Results =====")

print("Return:", round(total_return, 4))
print("CAGR:", round(cagr, 4))
print("Sharpe:", round(sharpe, 4))
print("Sortino:", round(sortino, 4))
print("Profit Factor:", round(profit_factor, 4))
print("Max Drawdown:", round(max_drawdown, 4))

print("Total Trades:", total_trades)
print("Winning Trades:", winning_trades)
print("Losing Trades:", losing_trades)
print("Win Rate (%):", round(win_rate, 2))
print("Average Trade Return:", round(avg_trade_return, 5))

print(
    "\nWalk Forward MAE:",
    round(wf_results["MAE"], 6)
)

print(
    "Walk Forward MSE:",
    round(wf_results["MSE"], 6)
)

print(
    "Walk Forward Return:",
    round(wf_results["Return"], 4)
)

print(
    "Walk Forward Sharpe:",
    round(wf_results["Sharpe"], 4)
)

# ==========================
# Buy & Hold Comparison
# ==========================

buy_hold = (
    1 + returns
).cumprod()

plt.figure(figsize=(10, 5))

plt.plot(
    equity_curve,
    label="Strategy"
)

plt.plot(
    buy_hold,
    label="Buy & Hold"
)

plt.title(
    "Strategy vs Buy & Hold"
)

plt.legend()
plt.savefig(
    "outputs/charts/strategy_vs_buy_hold.png",
    bbox_inches="tight"
)
plt.show()
plt.close()


# ==========================
# RL Training
# ==========================

rl_model = train_agent(data)

print(
    "\nRL Agent Training Completed"
)


# ==========================
# RL Evaluation
# ==========================

evaluate_agent(data)

with open(
    "outputs/reports/metrics.txt",
    "w"
) as f:

    f.write("===== MODEL =====\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"MSE: {mse}\n\n")

    f.write("===== BACKTEST =====\n")
    f.write(f"Return: {total_return}\n")
    f.write(f"Sharpe: {sharpe}\n")
    f.write(f"Max Drawdown: {max_drawdown}\n\n")

    f.write("===== PORTFOLIO =====\n")
    f.write(f"Weights: {weights}\n")
    f.write(f"Return: {ret}\n")
    f.write(f"Risk: {risk}\n")