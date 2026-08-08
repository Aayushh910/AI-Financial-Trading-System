# dashboard/views/risk.py

import streamlit as st

from dashboard.components.cards import render_section_header, render_kpi_card, render_status_badge
from dashboard.components.charts import create_volatility_chart, create_drawdown_chart
from dashboard.components.tables import render_risk_metrics_table
from dashboard.utils.formatting import format_pct, format_number


def render_risk_page():
    st.title("⚠ Risk Management & Volatility Terminal")
    st.caption("Quantitative risk monitoring, volatility regime detection, maximum drawdown containment, and risk filter analytics.")

    is_dark = st.session_state.get("theme_mode", "Dark") == "Dark"
    df = st.session_state.get("market_data")
    bt_results = st.session_state.get("backtest_results")

    # Determine Risk Status
    vol_current = df["Volatility"].iloc[-1] if df is not None and "Volatility" in df.columns else 0.0
    vol_thresh = bt_results.get("volatility_threshold", 0.02) if bt_results else 0.02

    if vol_current <= vol_thresh:
        risk_status = "LOW RISK — ACTIVE SIGNALS"
        risk_level = "low"
    elif vol_current <= vol_thresh * 1.5:
        risk_status = "MEDIUM RISK — CAUTION"
        risk_level = "medium"
    else:
        risk_status = "HIGH RISK — ALL SIGNALS FILTERED"
        risk_level = "high"

    # 1. Risk Status Header Banner
    render_section_header("🛡 Current Market Risk Status & Regime Classification")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.markdown("#### Volatility Regime:")
        render_status_badge(risk_status, level=risk_level)
    with col_r2:
        render_kpi_card("Current 20d Volatility", format_pct(vol_current * 100))
    with col_r3:
        render_kpi_card("Volatility Threshold (60th %)", format_pct(vol_thresh * 100))
    with col_r4:
        atr_val = df["ATR"].iloc[-1] if df is not None and "ATR" in df.columns else "N/A"
        render_kpi_card("Current ATR (14d)", format_number(atr_val, 4))

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Risk Charts (Volatility Curve & Drawdown Curve)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        render_section_header("📉 Volatility Regime Curve")
        if df is not None:
            fig_vol = create_volatility_chart(df.tail(252), is_dark=is_dark)
            st.plotly_chart(fig_vol, width="stretch")
        else:
            st.info("Market data unavailable for volatility curve.")

    with col_c2:
        render_section_header("🔻 Strategy Drawdown Curve")
        if bt_results and "equity_curve" in bt_results:
            fig_dd = create_drawdown_chart(bt_results["equity_curve"], dates=bt_results.get("dates"), is_dark=is_dark)
            st.plotly_chart(fig_dd, width="stretch")
        else:
            st.info("Backtest equity curve unavailable. Run backtest first.")

    with st.expander("💡 Risk Management & Volatility Filtering Guidelines"):
        st.markdown("""
        **Risk Control Architecture:**
        - **RandomForest Volatility Filter**: Trained on trailing feature data to forecast future market volatility.
        - **60th Percentile Quantile Threshold**: Calculated strictly from training historical data to prevent lookahead bias. Signals generated during market regimes above this threshold are automatically overridden to `0 (HOLD)` to protect capital during crashes.
        - **Position Exposure Cap**: Maximum position size is strictly capped at `95%` of total portfolio balance, leaving a 5% cash buffer.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Risk Metrics Table
    render_section_header("📋 Quantitative Risk Parameters Breakdown Table")
    
    max_dd = bt_results.get("max_drawdown") if bt_results else "N/A"
    sharpe = bt_results.get("sharpe") if bt_results else "N/A"
    sortino = bt_results.get("sortino") if bt_results else "N/A"
    
    risk_metrics = [
        {"Metric": "20-Day Rolling Volatility", "Value": format_pct(vol_current * 100), "Status": "Normal" if vol_current <= vol_thresh else "Elevated"},
        {"Metric": "Volatility Risk Filter Threshold", "Value": format_pct(vol_thresh * 100), "Status": "Target Threshold"},
        {"Metric": "Average True Range (ATR)", "Value": format_number(atr_val, 4), "Status": "Active"},
        {"Metric": "Maximum Historical Drawdown", "Value": format_pct(max_dd * 100 if isinstance(max_dd, (int, float)) and abs(max_dd) <= 1.0 else max_dd), "Status": "Controlled" if isinstance(max_dd, (int, float)) and abs(max_dd) < 0.2 else "Alert"},
        {"Metric": "Annualized Sharpe Ratio", "Value": format_number(sharpe, 3), "Status": "Positive" if isinstance(sharpe, (int, float)) and sharpe > 0 else "Neutral"},
        {"Metric": "Annualized Sortino Ratio", "Value": format_number(sortino, 3), "Status": "Positive" if isinstance(sortino, (int, float)) and sortino > 0 else "Neutral"},
        {"Metric": "Transaction Fee per Trade", "Value": "0.10% (10 bps)", "Status": "Active Deduction"},
        {"Metric": "Maximum Position Exposure Size", "Value": "95.0% Equity", "Status": "Capped"}
    ]
    render_risk_metrics_table(risk_metrics)
