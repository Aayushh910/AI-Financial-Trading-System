# dashboard/views/backtesting.py

import streamlit as st

from dashboard.components.cards import render_section_header, render_kpi_card
from dashboard.components.metrics import render_backtest_metrics_grid
from dashboard.components.charts import create_equity_curve_chart, create_drawdown_chart
from dashboard.components.tables import render_trade_history_table
from dashboard.utils.loaders import run_backtest_pipeline
from dashboard.utils.formatting import format_pct, format_number


def render_backtesting_page():
    st.title("📈 Quantitative Backtesting Terminal")
    st.caption("Historical strategy simulation, transaction fee friction, benchmark comparison, and trade performance breakdown.")

    asset = st.session_state.get("selected_asset", "AAPL")
    is_dark = st.session_state.get("theme_mode", "Dark") == "Dark"
    bt_results = st.session_state.get("backtest_results")

    # Top Controls Bar
    col_b1, col_b2, col_b3, col_b4 = st.columns([2, 2, 2, 1])
    with col_b1:
        st.markdown(f"**Asset Target:** `{asset}`")
    with col_b2:
        st.markdown(f"**Strategy Architecture:** `XGBoost + Volatility Filter`")
    with col_b3:
        st.markdown(f"**Initial Portfolio Capital:** `$10,000.00`")
    with col_b4:
        run_bt_clicked = st.button("🚀 Run Backtest", width="stretch")

    st.markdown("---")

    # Trigger backtest if button clicked or if model trained but no backtest
    if run_bt_clicked:
        if st.session_state.get("ml_predictions") is None or st.session_state.get("market_data") is None:
            st.warning("Please train the XGBoost model first by clicking 'Train XGBoost Model' in the sidebar.")
            return
        with st.spinner("Running historical strategy backtest with transaction fee friction..."):
            try:
                data = st.session_state.market_data
                preds = st.session_state.ml_predictions["Predicted"].values
                split_idx = int(len(data) * 0.8)
                res = run_backtest_pipeline(data, preds, split_idx)
                st.session_state.backtest_results = res
                bt_results = res
                st.success("Backtest executed successfully!")
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    if bt_results is not None:
        # 1. Benchmark Comparison Cards
        render_section_header("📊 Strategy vs Benchmark Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi_card("AI Strategy Return", format_pct(bt_results["total_return"] * 100))
        with col2:
            render_kpi_card("Buy & Hold Benchmark", format_pct(bt_results["buy_hold_return"] * 100))
        with col3:
            render_kpi_card("Outperformance Alpha", format_pct(bt_results["outperformance"] * 100, include_sign=True))
        with col4:
            render_kpi_card("Calmar Ratio", format_number(bt_results["calmar"], 3))

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Comprehensive Metrics Grid
        render_section_header("🎯 Complete Performance Risk & Return Metrics")
        render_backtest_metrics_grid(bt_results)

        with st.expander("💡 Backtest Metrics Interpretation Guide"):
            st.markdown("""
            **Key Quantitative Financial Ratios:**
            - **CAGR (Compound Annual Growth Rate)**: Geometric annual return rate over historical period.
            - **Sharpe Ratio**: Excess return per unit of total risk (standard deviation).
            - **Sortino Ratio**: Excess return per unit of **downside** risk only. Punishes negative volatility while ignoring positive upside variance.
            - **Calmar Ratio**: Ratio of CAGR to Maximum Drawdown. Measures return per unit of drawdown risk (`> 1.0` is strong).
            - **Transaction Friction**: A realistic 10 basis points (`0.10%`) cost is deducted on every position entry and exit to ensure backtest realism.
            """)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Interactive Plotly Equity Curve Chart
        render_section_header("📈 Strategy Cumulative Equity Growth Curve")
        fig_equity = create_equity_curve_chart(
            bt_results["equity_curve"],
            bt_results["buy_hold_equity"],
            dates=bt_results.get("dates"),
            is_dark=is_dark
        )
        st.plotly_chart(fig_equity, width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. Drawdown Curve
        render_section_header("🔻 Historical Portfolio Drawdown Containment")
        fig_dd = create_drawdown_chart(bt_results["equity_curve"], dates=bt_results.get("dates"), is_dark=is_dark)
        st.plotly_chart(fig_dd, width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)

        # 5. Trade Execution History Table
        render_section_header("📋 Detailed Trade Execution Log Table")
        render_trade_history_table(bt_results.get("trade_df"))
    else:
        st.info("""
        ℹ No backtest results generated for the current session.
        
        Click **⚡ Train XGBoost Model** or **🚀 Run Backtest** in the sidebar to simulate historical strategy performance.
        """)
