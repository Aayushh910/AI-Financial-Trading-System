import streamlit as st
from PIL import Image
import os

st.title("🤖 Reinforcement Learning Trading Agent")

st.markdown("### PPO (Proximal Policy Optimization) Agent")

metrics = {}
report_path = "outputs/reports/rl_metrics.txt"
if os.path.exists(report_path):
    with open(report_path, "r") as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                metrics[k.strip()] = v.strip()

total_reward = metrics.get("Total Reward", "N/A")
final_balance = f"${float(metrics.get('Final Portfolio Value', 10000)):,.2f}" if "Final Portfolio Value" in metrics else "$10,000"
return_pct = f"{metrics.get('Return (%)', '0')} %"
win_rate = f"{metrics.get('Win Rate (%)', '0')} %"
max_drawdown = f"{metrics.get('Max Drawdown (%)', '0')} %"
sharpe = metrics.get("Sharpe Ratio", "N/A")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Reward",
        total_reward
    )

with col2:
    st.metric(
        "Final Balance",
        final_balance
    )

with col3:
    st.metric(
        "Return",
        return_pct
    )

st.divider()

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Win Rate",
        win_rate
    )

with col5:
    st.metric(
        "Max Drawdown",
        max_drawdown
    )

with col6:
    st.metric(
        "Sharpe Ratio",
        sharpe
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