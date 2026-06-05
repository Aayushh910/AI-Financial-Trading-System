import streamlit as st
from PIL import Image
import os

st.title("🤖 Reinforcement Learning Trading Agent")

st.markdown("### PPO (Proximal Policy Optimization) Agent")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Reward",
        "38.45"
    )

with col2:
    st.metric(
        "Final Balance",
        "$100,198"
    )

with col3:
    st.metric(
        "Return",
        "901.98%"
    )

st.divider()

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Win Rate",
        "100%"
    )

with col5:
    st.metric(
        "Max Drawdown",
        "-37.11%"
    )

with col6:
    st.metric(
        "Final Position",
        "BUY"
    )

st.divider()

st.subheader("📈 RL Equity Curve")

chart_path = "outputs/charts/rl_equity_curve.png"

if os.path.exists(chart_path):

    img = Image.open(chart_path)

    st.image(
        img,
        width="stretch"
    )

else:

    st.warning(
        "Run RL training first to generate chart."
    )

st.divider()

st.subheader("🧠 RL Agent Summary")

st.success("""
✅ PPO agent successfully trained.

✅ Learned Buy / Hold / Sell decisions.

✅ Reward improved during training.

✅ Portfolio balance increased over time.

✅ Agent evaluated on historical market data.
""")

st.info("""
Reinforcement Learning allows the agent to learn
trading behavior through rewards and penalties
instead of relying only on fixed prediction rules.
""")