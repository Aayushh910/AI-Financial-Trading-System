# dashboard/views/home.py

import streamlit as st
import datetime

from dashboard.components.cards import render_signal_card, render_section_header
from dashboard.components.metrics import render_home_kpis
from dashboard.components.charts import create_price_line_chart, create_volume_chart
from dashboard.utils.loaders import load_market_data


def render_home_page():
    st.title("🚀 AI Financial Trading System")
    st.caption("AI-powered market prediction, portfolio optimization, risk management, backtesting, and reinforcement learning terminal.")
    
    asset = st.session_state.get("selected_asset", "AAPL")
    start_date = st.session_state.get("start_date", datetime.date(2015, 1, 1))
    end_date = st.session_state.get("end_date", datetime.date.today())
    last_updated = st.session_state.get("last_updated", "Not Synced")
    is_dark = st.session_state.get("theme_mode", "Dark") == "Dark"

    # Status Banner
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"**Selected Asset:** `{asset}`")
    with col_b:
        st.markdown(f"**Market Status:** <span style='color:#3fb950;'>● Active Trading</span>", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"**Last Model Sync:** `{last_updated}`")
        
    st.markdown("---")

    # Load market data automatically if not in session state
    if st.session_state.get("market_data") is None:
        with st.spinner(f"Loading market data for {asset}..."):
            try:
                df = load_market_data(asset, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                st.session_state.market_data = df
            except Exception as e:
                st.error(f"Error loading market data: {e}")

    df = st.session_state.get("market_data")
    bt_results = st.session_state.get("backtest_results")

    # 1. KPI Cards Grid (11 KPIs)
    render_section_header("📊 Key Performance Metrics Overview")
    render_home_kpis(results=bt_results, market_data=df, ticker=asset)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Prominent Signal Badge & Interpretation Box
    render_section_header("🎯 Latest AI Trading Signal")
    if bt_results and "signals" in bt_results and len(bt_results["signals"]) > 0:
        sig_raw = bt_results["signals"][-1]
        sig_text = "BUY" if sig_raw == 1 else ("SELL" if sig_raw == -1 else "HOLD")
        pred_ret = bt_results.get("pred_df")["Predicted"].iloc[-1] if "pred_df" in bt_results and not bt_results["pred_df"].empty else 0.0
        price = df["Close"].iloc[-1] if df is not None and not df.empty else 0.0
        timestamp = df.index[-1].strftime("%Y-%m-%d") if df is not None and not df.empty else None
        render_signal_card(signal=sig_text, predicted_return=pred_ret, current_price=price, timestamp=timestamp, model="XGBoost + Volatility Filter")
    else:
        st.info("ℹ No active prediction model results generated yet. Click 'Train XGBoost Model' in the sidebar to generate live trading signals.")

    with st.expander("💡 Signal Threshold Logic & Output Interpretation Guide"):
        st.markdown("""
        **Trading Signal Generation Logic:**
        - **BUY Signal (`+1`)**: Generated when Predicted Return exceeds `+0.1 * std(preds)` AND rolling volatility is below the 60th percentile risk threshold.
        - **SELL Signal (`-1`)**: Generated when Predicted Return is below `-0.1 * std(preds)` AND rolling volatility is below the 60th percentile risk threshold.
        - **HOLD Signal (`0`)**: Triggered when predicted return is neutral OR when high volatility risk filter overrides signal to protect capital.
        
        **Key Metrics Benchmarks:**
        - **Sharpe Ratio**: Measures return per unit of total risk (`> 1.0` is good, `> 2.0` is excellent).
        - **Profit Factor**: Ratio of gross profits to gross losses (`> 1.5` indicates robust strategy edge).
        - **Max Drawdown**: Peak-to-trough decline (`< 15%` indicates superior risk containment).
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Interactive Price Chart with Indicator Toggles
    render_section_header("📈 Interactive Market Price Terminal")
    if df is not None and not df.empty:
        col_t1, col_t2 = st.columns([1, 4])
        with col_t1:
            st.markdown("#### Overlay Toggles")
            show_ema = st.checkbox("EMA 10/20/50", value=True)
            show_sma = st.checkbox("SMA 10/20/50", value=False)
            show_bb = st.checkbox("Bollinger Bands", value=True)
            
            overlays = []
            if show_ema:
                overlays.extend(["EMA_10", "EMA_20", "EMA_50"])
            if show_sma:
                overlays.extend(["SMA_10", "SMA_20", "SMA_50"])
            if show_bb:
                overlays.append("Bollinger Bands")

        with col_t2:
            fig_price = create_price_line_chart(df.tail(252), ticker=asset, overlays=overlays, is_dark=is_dark)
            st.plotly_chart(fig_price, width="stretch")

        # 4. Volume Chart
        fig_vol = create_volume_chart(df.tail(252), is_dark=is_dark)
        st.plotly_chart(fig_vol, width="stretch")
    else:
        st.warning("Market data unavailable. Please verify internet connection or select a valid asset symbol.")

    # 5. Recent Performance Summary Table
    render_section_header("📝 Recent Model Predictions & Executed Signals Log")
    if bt_results and "trade_df" in bt_results and not bt_results["trade_df"].empty:
        st.dataframe(bt_results["trade_df"].tail(10), width="stretch", hide_index=True)
    else:
        st.info("No backtested trade history recorded yet. Run backtest in the sidebar to populate trade log.")
