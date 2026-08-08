# dashboard/components/metrics.py

import streamlit as st
from dashboard.components.cards import render_kpi_card
from dashboard.utils.formatting import format_currency, format_pct, format_number, get_signal_label


def render_home_kpis(results=None, market_data=None, ticker="AAPL"):
    """Render 11 Home KPI Cards in clean, responsive grid layout."""
    
    # Extract actual numbers from results dict if present
    price = market_data["Close"].iloc[-1] if market_data is not None and not market_data.empty else "N/A"
    
    pred_ret = results.get("pred_df")["Predicted"].iloc[-1] if results and "pred_df" in results and not results["pred_df"].empty else "N/A"
    sig_raw = results.get("signals")[-1] if results and "signals" in results and len(results["signals"]) > 0 else "N/A"
    signal = get_signal_label(sig_raw) if sig_raw != "N/A" else "N/A"
    
    port_ret = results.get("total_return") if results and "total_return" in results else "N/A"
    sharpe = results.get("sharpe") if results and "sharpe" in results else "N/A"
    win_rate = results.get("win_rate") if results and "win_rate" in results else "N/A"
    profit_factor = results.get("profit_factor") if results and "profit_factor" in results else "N/A"
    total_trades = results.get("total_trades") if results and "total_trades" in results else "N/A"
    max_dd = results.get("max_drawdown") if results and "max_drawdown" in results else "N/A"
    
    rl_res = st.session_state.get("rl_results")
    rl_ret = rl_res.get("return_pct") if rl_res else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Current Asset", ticker)
    with col2:
        render_kpi_card("Latest Price", format_currency(price))
    with col3:
        render_kpi_card("Predicted Return", format_number(pred_ret, 5))
    with col4:
        render_kpi_card("Trading Signal", signal)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        render_kpi_card("Strategy Return", format_pct(port_ret * 100 if isinstance(port_ret, (int, float)) else port_ret))
    with col6:
        render_kpi_card("Sharpe Ratio", format_number(sharpe, 3))
    with col7:
        render_kpi_card("Win Rate", format_pct(win_rate))
    with col8:
        render_kpi_card("Profit Factor", format_number(profit_factor, 3))

    col9, col10, col11 = st.columns(3)
    with col9:
        render_kpi_card("Total Trades", str(total_trades))
    with col10:
        render_kpi_card("RL Agent Return", format_pct(rl_ret))
    with col11:
        render_kpi_card("Max Drawdown", format_pct(max_dd * 100 if isinstance(max_dd, (int, float)) and abs(max_dd) <= 1.0 else max_dd))


def render_ml_metrics_grid(metrics_dict=None):
    """Render Machine Learning Evaluation metrics cards."""
    mae = metrics_dict.get("mae", "N/A") if metrics_dict else "N/A"
    mse = metrics_dict.get("mse", "N/A") if metrics_dict else "N/A"
    rmse = metrics_dict.get("rmse", "N/A") if metrics_dict else "N/A"
    r2 = metrics_dict.get("r2", "N/A") if metrics_dict else "N/A"
    dir_acc = metrics_dict.get("directional_acc", "N/A") if metrics_dict else "N/A"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("MAE", format_number(mae, 6))
    with col2:
        render_kpi_card("MSE", format_number(mse, 6))
    with col3:
        render_kpi_card("RMSE", format_number(rmse, 6))
    with col4:
        render_kpi_card("R² Score", format_number(r2, 4))
    with col5:
        render_kpi_card("Directional Accuracy", format_pct(dir_acc))


def render_backtest_metrics_grid(bt_results=None):
    """Render Backtesting performance metrics cards."""
    tot_ret = bt_results.get("total_return") if bt_results else "N/A"
    cagr = bt_results.get("cagr") if bt_results else "N/A"
    sharpe = bt_results.get("sharpe") if bt_results else "N/A"
    sortino = bt_results.get("sortino") if bt_results else "N/A"
    profit_factor = bt_results.get("profit_factor") if bt_results else "N/A"
    max_dd = bt_results.get("max_drawdown") if bt_results else "N/A"
    win_rate = bt_results.get("win_rate") if bt_results else "N/A"
    tot_trades = bt_results.get("total_trades") if bt_results else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total Strategy Return", format_pct(tot_ret * 100 if isinstance(tot_ret, (int, float)) else tot_ret))
    with col2:
        render_kpi_card("CAGR", format_pct(cagr * 100 if isinstance(cagr, (int, float)) else cagr))
    with col3:
        render_kpi_card("Sharpe Ratio", format_number(sharpe, 3))
    with col4:
        render_kpi_card("Sortino Ratio", format_number(sortino, 3))

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        render_kpi_card("Profit Factor", format_number(profit_factor, 3))
    with col6:
        render_kpi_card("Max Drawdown", format_pct(max_dd * 100 if isinstance(max_dd, (int, float)) and abs(max_dd) <= 1.0 else max_dd))
    with col7:
        render_kpi_card("Win Rate", format_pct(win_rate))
    with col8:
        render_kpi_card("Closed Trades", str(tot_trades))
