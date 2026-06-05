import streamlit as st

st.set_page_config(
    page_title="AI Financial Trading System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Financial Trading System")

st.markdown("""
### Intelligent Trading Platform using Machine Learning & Reinforcement Learning
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Assets", "3+")

with col2:
    st.metric("ML Model", "XGBoost")

with col3:
    st.metric("RL Agent", "PPO")

with col4:
    st.metric("Risk Module", "Active")

st.divider()

st.subheader("🎯 Project Objective")

st.write("""
Build an AI-powered trading system capable of:

- Predicting market movements
- Managing trading risk
- Optimizing portfolios
- Training autonomous trading agents
- Evaluating performance through backtesting
""")

st.divider()

st.subheader("⚙️ System Workflow")

st.code("""
Market Data
    ↓
Feature Engineering
    ↓
XGBoost Prediction
    ↓
Signal Generation
    ↓
Risk Filtering
    ↓
Backtesting
    ↓
Portfolio Optimization
    ↓
PPO Reinforcement Learning Agent
    ↓
Performance Dashboard
""")

st.divider()

st.subheader("🛠 Technologies Used")

st.markdown("""
- Python
- Pandas & NumPy
- XGBoost
- Scikit-Learn
- Stable-Baselines3
- Gymnasium
- Optuna
- Streamlit
- Yahoo Finance API
""")

st.divider()

st.subheader("👨‍💻 Team Members")

st.markdown("""
- Aayush Savaliya
- Jeel
""")

st.divider()

st.success("✅ Project Dashboard Ready")