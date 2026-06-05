import streamlit as st
from PIL import Image
import os

st.title("📊 Backtesting Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Return",
        "20.19%"
    )

with col2:
    st.metric(
        "Sharpe Ratio",
        "0.053"
    )

with col3:
    st.metric(
        "Max Drawdown",
        "-9.12%"
    )

st.divider()

st.subheader("📈 Strategy Performance")

chart_path = "outputs/charts/strategy_vs_buy_hold.png"

if os.path.exists(chart_path):

    img = Image.open(chart_path)

    st.image(
        img,
        width="stretch"
    )

else:

    st.warning(
        "Run main.py first to generate charts."
    )

st.divider()

st.subheader("📝 Performance Summary")

st.success("""
✅ Strategy generated positive returns.

✅ Portfolio risk remained controlled.

✅ Maximum drawdown stayed below 10%.

✅ Outperformed many basic trading approaches.
""")

st.info("""
Backtesting evaluates how the trading strategy would
have performed on historical market data before
deploying it in a live environment.
""")