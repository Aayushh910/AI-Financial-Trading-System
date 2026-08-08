# dashboard/components/charts.py

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def apply_trading_chart_theme(fig, is_dark=True, title=""):
    """Apply standard professional trading terminal styling to Plotly figures."""
    bg_color = "#0d1117" if is_dark else "#ffffff"
    paper_color = "#161b22" if is_dark else "#f6f8fa"
    text_color = "#c9d1d9" if is_dark else "#24292f"
    grid_color = "#21262d" if is_dark else "#e1e4e8"

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=text_color, family="Inter, sans-serif")
        ),
        paper_bgcolor=paper_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            color=text_color,
            showgrid=True
        ),
        yaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            color=text_color,
            showgrid=True
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=text_color)
        ),
        hovermode="x unified"
    )
    return fig


def create_price_line_chart(df, ticker="AAPL", overlays=None, is_dark=True):
    """Create interactive line price chart with indicator overlays."""
    if overlays is None:
        overlays = []

    fig = go.Figure()

    # Base Close price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#58a6ff", width=2)
    ))

    # Overlays
    if "EMA_10" in overlays and "EMA_10" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_10"], mode="lines", name="EMA 10", line=dict(color="#f1e05a", width=1.5)))
    if "EMA_20" in overlays and "EMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], mode="lines", name="EMA 20", line=dict(color="#ff9b00", width=1.5)))
    if "EMA_50" in overlays and "EMA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], mode="lines", name="EMA 50", line=dict(color="#a371f7", width=1.5)))
    if "SMA_10" in overlays and "SMA_10" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_10"], mode="lines", name="SMA 10", line=dict(color="#79c0ff", width=1.5, dash="dash")))
    if "SMA_20" in overlays and "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20", line=dict(color="#d2a8ff", width=1.5, dash="dash")))
    if "SMA_50" in overlays and "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode="lines", name="SMA 50", line=dict(color="#ffa657", width=1.5, dash="dash")))
    if "Bollinger Bands" in overlays and "BB_HIGH" in df.columns and "BB_LOW" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_HIGH"], mode="lines", name="BB Upper", line=dict(color="#388bfd", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], mode="lines", name="BB Lower", line=dict(color="#388bfd", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(56, 139, 253, 0.05)"))

    return apply_trading_chart_theme(fig, is_dark=is_dark, title=f"{ticker} — Interactive Price Chart")


def create_candlestick_chart(df, ticker="AAPL", overlays=None, is_dark=True):
    """Create interactive OHLC Candlestick chart."""
    if overlays is None:
        overlays = []

    fig = go.Figure()

    # Candlestick trace
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"] if "Open" in df.columns else df["Close"],
        high=df["High"] if "High" in df.columns else df["Close"],
        low=df["Low"] if "Low" in df.columns else df["Close"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="#2ea043",
        decreasing_line_color="#f85149"
    ))

    # Overlays
    if "EMA 10" in overlays and "EMA_10" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_10"], mode="lines", name="EMA 10", line=dict(color="#f1e05a", width=1.5)))
    if "EMA 20" in overlays and "EMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], mode="lines", name="EMA 20", line=dict(color="#ff9b00", width=1.5)))
    if "EMA 50" in overlays and "EMA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], mode="lines", name="EMA 50", line=dict(color="#a371f7", width=1.5)))
    if "SMA 20" in overlays and "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20", line=dict(color="#79c0ff", width=1.5, dash="dash")))
    if "Bollinger Bands" in overlays and "BB_HIGH" in df.columns and "BB_LOW" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_HIGH"], mode="lines", name="BB High", line=dict(color="#388bfd", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], mode="lines", name="BB Low", line=dict(color="#388bfd", width=1, dash="dot")))

    fig.update_layout(xaxis_rangeslider_visible=False)
    return apply_trading_chart_theme(fig, is_dark=is_dark, title=f"{ticker} — Technical Candlestick Terminal")


def create_volume_chart(df, is_dark=True):
    """Create trading volume bar chart with Volume Moving Average."""
    fig = go.Figure()
    
    colors = ["#2ea043" if r >= 0 else "#f85149" for r in df["Close"].pct_change().fillna(0)]
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color=colors,
        opacity=0.7
    ))
    
    if "Volume_MA" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Volume_MA"],
            mode="lines",
            name="Volume MA (20)",
            line=dict(color="#58a6ff", width=1.5)
        ))

    return apply_trading_chart_theme(fig, is_dark=is_dark, title="Trading Volume Analysis")


def create_rsi_chart(df, is_dark=True):
    """Create RSI indicator plot with 70/30 threshold bounds."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["RSI"],
        mode="lines",
        name="RSI",
        line=dict(color="#a371f7", width=2)
    ))
    
    fig.add_hline(y=70, line_dash="dash", line_color="#f85149", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="#2ea043", annotation_text="Oversold (30)")
    
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="Relative Strength Index (RSI)")


def create_macd_chart(df, is_dark=True):
    """Create MACD line, Signal line, and Histogram chart."""
    fig = go.Figure()
    
    hist = df["MACD"] - df["MACD_SIGNAL"]
    colors = ["#2ea043" if h >= 0 else "#f85149" for h in hist]
    
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram", marker_color=colors, opacity=0.5))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", line=dict(color="#58a6ff", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="Signal", line=dict(color="#ff9b00", width=1.5)))
    
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="MACD (Moving Average Convergence Divergence)")


def create_volatility_chart(df, is_dark=True):
    """Create Rolling Volatility & ATR line chart."""
    fig = go.Figure()
    
    if "Volatility" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Volatility"], mode="lines", name="Rolling Volatility (20d)", line=dict(color="#f85149", width=2)))
    if "ATR" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], mode="lines", name="ATR (14d)", line=dict(color="#d29922", width=1.5, dash="dash")))
        
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="Volatility & ATR Analysis")


def create_actual_vs_predicted_chart(pred_df, is_dark=True):
    """Create Actual Return vs Predicted Return line plot for ML evaluation."""
    fig = go.Figure()
    
    x_axis = pred_df["Date"] if "Date" in pred_df.columns else pred_df.index
    
    fig.add_trace(go.Scatter(x=x_axis, y=pred_df["Actual"], mode="lines", name="Actual Target Return", line=dict(color="#58a6ff", width=1.5)))
    fig.add_trace(go.Scatter(x=x_axis, y=pred_df["Predicted"], mode="lines", name="XGBoost Predicted Return", line=dict(color="#ff9b00", width=1.5, dash="dot")))
    
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="ML Predictions vs Actual Market Returns")


def create_feature_importance_chart(importance_df, is_dark=True):
    """Create horizontal bar chart of feature importance."""
    sorted_df = importance_df.sort_values("Importance", ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_df["Importance"],
        y=sorted_df["Feature"],
        orientation="h",
        marker_color="#58a6ff"
    ))
    
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="XGBoost Feature Importance Ranking")


def create_residual_chart(pred_df, is_dark=True):
    """Create scatter plot of prediction residuals."""
    residuals = pred_df["Actual"] - pred_df["Predicted"]
    x_axis = pred_df["Date"] if "Date" in pred_df.columns else pred_df.index
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=residuals,
        mode="markers",
        name="Residuals",
        marker=dict(color="#d29922", size=5, opacity=0.7)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e")
    
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="ML Model Prediction Residuals")


def create_equity_curve_chart(equity_series, buy_hold_series, dates=None, is_dark=True):
    """Create interactive Backtest Strategy Equity Curve vs Buy & Hold benchmark."""
    fig = go.Figure()
    
    x_axis = dates if dates is not None else equity_series.index
    
    fig.add_trace(go.Scatter(x=x_axis, y=equity_series, mode="lines", name="AI Strategy", line=dict(color="#2ea043", width=2.5)))
    fig.add_trace(go.Scatter(x=x_axis, y=buy_hold_series, mode="lines", name="Buy & Hold Benchmark", line=dict(color="#8b949e", width=1.5, dash="dash")))
    
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="Cumulative Equity Growth (Strategy vs Benchmark)")


def create_drawdown_chart(equity_series, dates=None, is_dark=True):
    """Create Drawdown Area Chart."""
    rolling_max = equity_series.cummax()
    drawdown = (equity_series / rolling_max - 1) * 100
    x_axis = dates if dates is not None else equity_series.index
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=drawdown,
        mode="lines",
        name="Drawdown (%)",
        line=dict(color="#f85149", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(248, 81, 73, 0.15)"
    ))
    
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="Portfolio Drawdown Curve (%)")


def create_portfolio_pie_chart(weights_df, is_dark=True):
    """Create interactive Donut Chart for Portfolio Asset Weights."""
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=weights_df["Asset"],
        values=weights_df["Weight (%)"],
        hole=0.4,
        marker=dict(colors=["#58a6ff", "#2ea043", "#ff9b00", "#a371f7", "#d29922", "#f85149"]),
        textinfo="label+percent",
        hoverinfo="label+value+percent"
    ))
    
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="Optimal Portfolio Asset Allocation")


def create_correlation_heatmap(returns_df, is_dark=True):
    """Create correlation heatmap for multi-asset returns."""
    corr = returns_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="Blues" if not is_dark else "Viridis",
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        showscale=True
    ))
    
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="Asset Cross-Correlation Matrix")


def create_rl_portfolio_chart(portfolio_values, is_dark=True):
    """Create RL Agent Portfolio Value curve over step timesteps."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=portfolio_values,
        mode="lines",
        name="RL Portfolio Balance",
        line=dict(color="#3fb950", width=2)
    ))
    
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="PPO Reinforcement Learning Portfolio Growth ($)")


def create_rl_action_chart(actions_series, is_dark=True):
    """Create RL Action Distribution Pie Chart."""
    action_counts = actions_series.value_counts()
    labels = [ "HOLD" if a == 0 else ("BUY" if a == 1 else "SELL") for a in action_counts.index ]
    
    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=action_counts.values,
        hole=0.4,
        marker=dict(colors=["#d29922", "#3fb950", "#f85149"])
    ))
    
    return apply_trading_chart_theme(fig, is_dark=is_dark, title="RL Agent Action Distribution")
