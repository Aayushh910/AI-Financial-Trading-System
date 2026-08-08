# dashboard/utils/loaders.py

import os
import sys
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import streamlit as st

# Add project root to path for backend module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from features.feature_engineering import create_features
from risk.volatility_model import VolatilityModel
from backtesting.strategy import apply_risk_filter
from models.portfolio_optimizer import optimize_portfolio
from rl_agents.train_rl import train_agent
from rl_agents.evaluate_rl import evaluate_agent

FEATURE_COLUMNS = [
    "Lag_1", "Lag_2", "Lag_3", "Lag_5", "Momentum", "Rolling_STD",
    "RSI", "MACD", "MACD_SIGNAL", "BB_HIGH", "BB_LOW", "ATR",
    "Volume_Change", "EMA_10", "EMA_20", "EMA_50",
    "SMA_10", "SMA_20", "SMA_50", "Price_Range", "Volume_MA"
]


@st.cache_data(show_spinner=False)
def load_market_data(ticker="AAPL", start="2015-01-01", end=None):
    """Fetch market data for a given ticker and date range with indicator features."""
    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")
        
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        return None
        
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    processed_df = create_features(data)
    return processed_df


@st.cache_data(show_spinner=False)
def load_multi_asset_data(assets=["AAPL", "MSFT", "GOOG"], start="2015-01-01", end=None):
    """Download daily close prices for multiple assets for portfolio optimization."""
    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")
        
    data_portfolio = pd.DataFrame()
    for asset in assets:
        try:
            df_asset = yf.download(asset, start=start, end=end, auto_adjust=False, progress=False)
            if not df_asset.empty:
                if isinstance(df_asset.columns, pd.MultiIndex):
                    df_asset.columns = df_asset.columns.get_level_values(0)
                data_portfolio[asset] = df_asset["Close"]
        except Exception:
            continue
            
    if data_portfolio.empty:
        return None
        
    returns_portfolio = data_portfolio.pct_change().dropna()
    return data_portfolio, returns_portfolio


def run_ml_pipeline(ticker="AAPL", start="2015-01-01", end=None):
    """Execute complete XGBoost training pipeline and return results dict."""
    data = load_market_data(ticker, start, end)
    if data is None or data.empty:
        raise ValueError(f"No market data available for {ticker}")
        
    X = data[FEATURE_COLUMNS]
    y = data["Target"]
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLUMNS)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLUMNS)
    
    os.makedirs(os.path.join(PROJECT_ROOT, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "predictions"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "reports"), exist_ok=True)
    
    joblib.dump(scaler, os.path.join(PROJECT_ROOT, "saved_models", "scaler.pkl"))
    
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )
    
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, os.path.join(PROJECT_ROOT, "saved_models", "xgb_model.pkl"))
    
    preds = model.predict(X_test_scaled)
    
    pred_df = pd.DataFrame({
        "Date": y_test.index,
        "Actual": y_test.values,
        "Predicted": preds,
        "Close": data["Close"].iloc[split:].values
    })
    pred_df.to_csv(os.path.join(PROJECT_ROOT, "outputs", "predictions", "predictions.csv"), index=False)
    
    mae = float(mean_absolute_error(y_test, preds))
    mse = float(mean_squared_error(y_test, preds))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, preds))
    directional_acc = float(np.mean(np.sign(preds) == np.sign(y_test.values)) * 100)
    
    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    # Save metrics report
    with open(os.path.join(PROJECT_ROOT, "outputs", "reports", "metrics.txt"), "w") as f:
        f.write("===== MODEL METRICS =====\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"R2: {r2:.6f}\n")
        f.write(f"Directional Accuracy (%): {directional_acc:.2f}%\n")
        
    return {
        "model": model,
        "scaler": scaler,
        "pred_df": pred_df,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "directional_acc": directional_acc,
        "importance": importance,
        "data": data,
        "split_idx": split
    }


def run_backtest_pipeline(data, preds, split_idx):
    """Execute backtest strategy with risk filtering and compute complete performance metrics."""
    X = data[FEATURE_COLUMNS]
    split = split_idx
    
    scaler_path = os.path.join(PROJECT_ROOT, "saved_models", "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_train_scaled = scaler.transform(X[:split])
        X_test_scaled = scaler.transform(X[split:])
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[:split])
        X_test_scaled = scaler.transform(X[split:])
        
    # Generate initial signals
    std_preds = np.std(preds) + 1e-8
    buy_threshold = 0.1 * std_preds
    sell_threshold = -0.1 * std_preds
    
    signals = []
    for p in preds:
        if p > buy_threshold:
            signals.append(1)
        elif p < sell_threshold:
            signals.append(-1)
        else:
            signals.append(0)
            
    # Train Volatility Risk Filter
    y_vol = data["Volatility"]
    vol_model = VolatilityModel()
    vol_model.train(X_train_scaled, y_vol[:split])
    vol_pred = vol_model.predict(X_test_scaled)
    
    vol_threshold = y_vol[:split].quantile(0.60)
    final_signals = apply_risk_filter(signals, vol_pred, vol_threshold)
    
    returns = data["Target"][split:].reset_index(drop=True)
    dates = data.index[split:]
    signals_series = pd.Series(final_signals)
    
    trades = (signals_series.diff().fillna(signals_series) != 0) & (signals_series != 0)
    transaction_cost = 0.001
    
    strategy_returns = (returns * signals_series) - (trades.astype(float) * transaction_cost)
    equity_curve = (1 + strategy_returns).cumprod()
    buy_hold_equity = (1 + returns).cumprod()
    
    total_return = float(equity_curve.iloc[-1] - 1)
    buy_hold_return = float(buy_hold_equity.iloc[-1] - 1)
    outperformance = total_return - buy_hold_return
    
    sharpe = float((np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)) * np.sqrt(252))
    
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1
    max_drawdown = float(drawdown.min())
    
    years = max(len(strategy_returns) / 252.0, 0.1)
    cagr = float((equity_curve.iloc[-1]) ** (1 / years) - 1)
    
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = downside_returns.std() + 1e-8
    sortino = float((strategy_returns.mean() / downside_std) * np.sqrt(252)) if len(downside_returns) > 0 else 0.0
    
    calmar = float(cagr / (abs(max_drawdown) + 1e-8))
    
    gross_profit = float(strategy_returns[strategy_returns > 0].sum())
    gross_loss = abs(float(strategy_returns[strategy_returns < 0].sum()))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    
    trade_returns = strategy_returns[trades]
    total_trades = len(trade_returns)
    winning_trades = len(trade_returns[trade_returns > 0])
    losing_trades = len(trade_returns[trade_returns < 0])
    win_rate = float(winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    avg_trade_return = float(trade_returns.mean()) if total_trades > 0 else 0.0
    
    # Detailed trades dataframe
    trade_list = []
    prices = data["Close"].iloc[split:].values
    curr_pos = 0
    entry_p = 0.0
    entry_d = None
    
    for i in range(len(signals_series)):
        sig = signals_series.iloc[i]
        p = prices[i]
        d = dates[i]
        
        if curr_pos == 0 and sig != 0:
            curr_pos = sig
            entry_p = p
            entry_d = d
        elif curr_pos != 0 and (sig != curr_pos):
            exit_p = p
            ret = (exit_p - entry_p) / entry_p if curr_pos == 1 else (entry_p - exit_p) / entry_p
            pnl = ret * 10000.0  # Assumes $10,000 capital per trade for PnL visualization
            trade_list.append({
                "Entry Date": entry_d,
                "Exit Date": d,
                "Action": "BUY" if curr_pos == 1 else "SELL",
                "Entry Price": entry_p,
                "Exit Price": exit_p,
                "Return (%)": ret * 100,
                "PnL ($)": pnl
            })
            if sig != 0:
                curr_pos = sig
                entry_p = p
                entry_d = d
            else:
                curr_pos = 0
                
    trade_df = pd.DataFrame(trade_list)
    
    return {
        "equity_curve": equity_curve,
        "buy_hold_equity": buy_hold_equity,
        "strategy_returns": strategy_returns,
        "dates": dates,
        "signals": final_signals,
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "outperformance": outperformance,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "trade_df": trade_df,
        "volatility_pred": vol_pred,
        "volatility_threshold": vol_threshold
    }


def run_portfolio_pipeline(assets=["AAPL", "MSFT", "GOOG"], start="2015-01-01", end=None):
    """Run portfolio optimization across multi-asset returns."""
    data_portfolio, returns_portfolio = load_multi_asset_data(assets, start, end)
    if returns_portfolio is None or returns_portfolio.empty:
        raise ValueError("Could not load return data for specified portfolio assets.")
        
    weights, ret, risk = optimize_portfolio(returns_portfolio)
    sharpe = float(ret / (risk + 1e-8))
    
    weights_df = pd.DataFrame({
        "Asset": returns_portfolio.columns,
        "Weight (%)": np.round(weights * 100, 2)
    })
    
    weights_df.to_csv(os.path.join(PROJECT_ROOT, "outputs", "reports", "portfolio_weights.csv"), index=False)
    
    return {
        "assets": list(returns_portfolio.columns),
        "weights": weights,
        "weights_df": weights_df,
        "expected_return": float(ret),
        "portfolio_risk": float(risk),
        "sharpe_ratio": sharpe,
        "returns_matrix": returns_portfolio,
        "price_data": data_portfolio
    }


def run_rl_pipeline(ticker="AAPL", start="2015-01-01", end=None, timesteps=40000):
    """Train PPO agent and evaluate on historical data."""
    data = load_market_data(ticker, start, end)
    if data is None or data.empty:
        raise ValueError(f"No market data available for RL training on {ticker}")
        
    model = train_agent(data)
    eval_metrics = evaluate_agent(data)
    return eval_metrics
