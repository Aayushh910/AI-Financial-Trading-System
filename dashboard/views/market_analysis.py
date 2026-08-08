# dashboard/views/market_analysis.py

import streamlit as st
import datetime

from dashboard.components.cards import render_section_header
from dashboard.components.charts import (
    create_candlestick_chart,
    create_rsi_chart,
    create_macd_chart,
    create_volatility_chart,
    create_volume_chart,
    create_correlation_heatmap
)
from dashboard.utils.loaders import load_market_data, load_multi_asset_data


def render_market_analysis_page():
    st.title("📊 Technical Market Analysis Terminal")
    st.caption("Deep-dive technical indicators, candlestick charts, volatility regime tracking, and cross-asset correlations.")

    asset = st.session_state.get("selected_asset", "AAPL")
    start_date = st.session_state.get("start_date", datetime.date(2015, 1, 1))
    end_date = st.session_state.get("end_date", datetime.date.today())
    is_dark = st.session_state.get("theme_mode", "Dark") == "Dark"

    # Top Controls Bar
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.markdown(f"**Symbol:** `{asset}`")
    with c2:
        st.markdown(f"**Period:** `{start_date}` to `{end_date}`")
    with c3:
        overlays_selected = st.multiselect(
            "Candlestick Technical Overlays",
            ["EMA 10", "EMA 20", "EMA 50", "SMA 20", "Bollinger Bands"],
            default=["EMA 20", "Bollinger Bands"],
            help="Select moving averages or volatility envelopes to overlay directly onto the OHLC candlestick chart."
        )

    st.markdown("---")

    # Load data
    df = st.session_state.get("market_data")
    if df is None:
        with st.spinner(f"Loading market data for {asset}..."):
            try:
                df = load_market_data(asset, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                st.session_state.market_data = df
            except Exception as e:
                st.error(f"Error loading market data: {e}")

    if df is not None and not df.empty:
        # 1. Candlestick Terminal Chart
        render_section_header("🕯 OHLC Candlestick Terminal")
        fig_candle = create_candlestick_chart(df.tail(200), ticker=asset, overlays=overlays_selected, is_dark=is_dark)
        st.plotly_chart(fig_candle, width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Technical Indicator Sub-Panels (Tabs for clean navigation)
        render_section_header("📉 Technical Oscillators & Volatility Panels")
        tab_rsi, tab_macd, tab_vol, tab_volu = st.tabs(["RSI Oscillator", "MACD Indicator", "Volatility & ATR", "Volume Analysis"])

        with tab_rsi:
            fig_rsi = create_rsi_chart(df.tail(252), is_dark=is_dark)
            st.plotly_chart(fig_rsi, width="stretch")

        with tab_macd:
            fig_macd = create_macd_chart(df.tail(252), is_dark=is_dark)
            st.plotly_chart(fig_macd, width="stretch")

        with tab_vol:
            fig_vol = create_volatility_chart(df.tail(252), is_dark=is_dark)
            st.plotly_chart(fig_vol, width="stretch")

        with tab_volu:
            fig_volu = create_volume_chart(df.tail(252), is_dark=is_dark)
            st.plotly_chart(fig_volu, width="stretch")

        with st.expander("💡 How to Read Technical Indicators & Trading Guidelines"):
            st.markdown("""
            **Technical Analysis Guidance:**
            - **RSI (Relative Strength Index)**:
              - `RSI > 70`: Overbought territory — price may be due for mean reversion or pullbacks.
              - `RSI < 30`: Oversold territory — price may be oversold and due for a bullish bounce.
            - **MACD (Moving Average Convergence Divergence)**:
              - **Bullish Signal**: MACD Line crosses above Signal Line with positive histogram bars.
              - **Bearish Signal**: MACD Line crosses below Signal Line with negative histogram bars.
            - **ATR (Average True Range)**:
              - Quantifies trailing 14-day dollar price volatility. High ATR indicates wider price swings.
            - **Bollinger Bands**:
              - Envelopes placed 2 standard deviations above and below a 20-day SMA. Band squeeze predicts imminent volatility breakouts.
            """)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Cross-Asset Correlation Matrix
        render_section_header("🔗 Cross-Asset Return Correlation Heatmap")
        corr_assets = st.multiselect(
            "Select assets for cross-correlation matrix",
            ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA"],
            default=["AAPL", "MSFT", "GOOG", "AMZN"],
            help="Computes Pearson pairwise correlation coefficients of daily return series."
        )

        if len(corr_assets) > 1:
            with st.spinner("Downloading cross-asset price history..."):
                try:
                    _, returns_portfolio = load_multi_asset_data(
                        corr_assets,
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d")
                    )
                    if returns_portfolio is not None and not returns_portfolio.empty:
                        fig_corr = create_correlation_heatmap(returns_portfolio, is_dark=is_dark)
                        st.plotly_chart(fig_corr, width="stretch")
                        
                        with st.expander("💡 Understanding Cross-Asset Correlation"):
                            st.markdown("""
                            - **Correlation `+1.0`**: Assets move in perfect lockstep. High concentration risk if held together.
                            - **Correlation `0.0`**: Asset returns are uncorrelated, providing diversification benefit.
                            - **Correlation `-1.0`**: Assets move in opposite directions, offering natural portfolio hedging.
                            """)
                    else:
                        st.warning("Could not calculate correlation matrix for selected assets.")
                except Exception as e:
                    st.error(f"Error computing correlations: {e}")
        else:
            st.info("Select at least 2 assets to compute cross-correlation heatmap.")
    else:
        st.warning("No market data available for analysis.")
