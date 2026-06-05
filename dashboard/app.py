import streamlit as st

st.set_page_config(
    page_title="AI Financial Trading System",
    page_icon="📈",
    layout="wide"
)

st.title("🚀 AI Financial Trading System")

st.markdown("""
Welcome to the AI-powered trading platform.

### Modules

- 📈 Market Prediction
- 💼 Portfolio Optimization
- 🛡️ Risk Management
- 📊 Backtesting
- 🤖 Reinforcement Learning
- 📋 Reports & Analytics
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("ML Model", "XGBoost")

with col2:
    st.metric("RL Agent", "PPO")

with col3:
    st.metric("Data Source", "Yahoo Finance")

st.info(
    "Use the sidebar to navigate through different modules."
)