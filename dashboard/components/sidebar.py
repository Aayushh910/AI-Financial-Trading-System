# dashboard/components/sidebar.py

import streamlit as st
import datetime
from dashboard.utils.loaders import (
    load_market_data,
    run_ml_pipeline,
    run_backtest_pipeline,
    run_portfolio_pipeline,
    run_rl_pipeline
)

PAGES = [
    "🏠 Home",
    "📊 Market Analysis",
    "🤖 Machine Learning",
    "💼 Portfolio Optimization",
    "⚠ Risk Management",
    "📈 Backtesting",
    "🧠 Reinforcement Learning",
    "ℹ About"
]

ASSETS_AVAILABLE = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA"]


def render_sidebar():
    """Render an organized, professional sidebar control panel."""
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #58a6ff;'>📈 AI Trading System</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #8b949e;'>Quantitative Terminal v2.0</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        # 1. Navigation Menu
        st.markdown("<div class='sidebar-section-title'>📍 Terminal Navigation</div>", unsafe_allow_html=True)
        selected_page = st.radio(
            "Select Page View",
            PAGES,
            index=PAGES.index(st.session_state.get("selected_page", "🏠 Home")),
            label_visibility="collapsed"
        )
        st.session_state.selected_page = selected_page
        
        st.markdown("---")
        
        # 2. Asset & Period Settings
        st.markdown("<div class='sidebar-section-title'>📊 Asset & Timeframe</div>", unsafe_allow_html=True)
        selected_asset = st.selectbox(
            "Target Equity Symbol",
            ASSETS_AVAILABLE,
            index=ASSETS_AVAILABLE.index(st.session_state.get("selected_asset", "AAPL")),
            help="Select the primary stock ticker symbol for analysis, machine learning prediction, and strategy backtesting."
        )
        st.session_state.selected_asset = selected_asset
        
        col1, col2 = st.columns(2)
        with col1:
            start_d = st.date_input(
                "Start Date",
                value=st.session_state.get("start_date", datetime.date(2015, 1, 1)),
                help="Historical data training start date."
            )
        with col2:
            end_d = st.date_input(
                "End Date",
                value=st.session_state.get("end_date", datetime.date.today()),
                help="Historical data end date."
            )
            
        if start_d >= end_d:
            st.error("⚠ Start Date must be earlier than End Date.")
        else:
            st.session_state.start_date = start_d
            st.session_state.end_date = end_d
            
        st.markdown("---")
        
        # 3. Model & Theme Settings
        st.markdown("<div class='sidebar-section-title'>🤖 Model & Interface</div>", unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Prediction Algorithm",
            ["XGBoost Regressor", "Random Forest — (Coming Soon)", "LightGBM — (Coming Soon)"],
            index=0,
            help="Supervised learning algorithm used for 1-day future return prediction."
        )
        st.session_state.selected_model_name = "XGBoost Regressor"
        
        theme_choice = st.radio(
            "Theme Color Mode",
            ["Dark", "Light"],
            index=0 if st.session_state.get("theme_mode", "Dark") == "Dark" else 1,
            horizontal=True
        )
        st.session_state.theme_mode = theme_choice
        
        st.markdown("---")
        
        # 4. Backend Actions & Triggers
        st.markdown("<div class='sidebar-section-title'>⚡ Execution Actions</div>", unsafe_allow_html=True)
        
        if st.button("📥 Load Market Data", width="stretch", help="Download historical price & volume data from Yahoo Finance API"):
            with st.spinner(f"Fetching data for {selected_asset}..."):
                try:
                    df = load_market_data(
                        selected_asset,
                        start_d.strftime("%Y-%m-%d"),
                        end_d.strftime("%Y-%m-%d")
                    )
                    st.session_state.market_data = df
                    st.session_state.last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success(f"Loaded {len(df)} bars for {selected_asset}!")
                except Exception as e:
                    st.error(f"Failed to load data: {e}")

        if st.button("⚡ Train XGBoost Model", width="stretch", help="Train XGBoost model, calculate features, and compute signal predictions"):
            with st.spinner(f"Training XGBoost model on {selected_asset}..."):
                try:
                    ml_res = run_ml_pipeline(
                        selected_asset,
                        start_d.strftime("%Y-%m-%d"),
                        end_d.strftime("%Y-%m-%d")
                    )
                    st.session_state.ml_predictions = ml_res["pred_df"]
                    st.session_state.ml_metrics = {
                        "mae": ml_res["mae"],
                        "mse": ml_res["mse"],
                        "rmse": ml_res["rmse"],
                        "r2": ml_res["r2"],
                        "directional_acc": ml_res["directional_acc"]
                    }
                    st.session_state.ml_feature_importance = ml_res["importance"]
                    st.session_state.market_data = ml_res["data"]
                    
                    # Auto-run backtest after model training
                    bt_res = run_backtest_pipeline(
                        ml_res["data"],
                        ml_res["pred_df"]["Predicted"].values,
                        ml_res["split_idx"]
                    )
                    st.session_state.backtest_results = bt_res
                    st.session_state.last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("XGBoost trained & strategy backtested!")
                except Exception as e:
                    st.error(f"Model training failed: {e}")

        if st.button("📊 Run Strategy Backtest", width="stretch", help="Simulate historical strategy returns with volatility risk filter and transaction costs"):
            if st.session_state.get("ml_predictions") is None or st.session_state.get("market_data") is None:
                st.warning("Please train the model first by clicking 'Train XGBoost Model'.")
            else:
                with st.spinner("Running historical backtest..."):
                    try:
                        data = st.session_state.market_data
                        preds = st.session_state.ml_predictions["Predicted"].values
                        split_idx = int(len(data) * 0.8)
                        
                        bt_res = run_backtest_pipeline(data, preds, split_idx)
                        st.session_state.backtest_results = bt_res
                        st.session_state.last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success("Backtest simulation completed!")
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")

        if st.button("🧠 Train PPO Agent", width="stretch", help="Train Proximal Policy Optimization RL agent for 40,000 steps"):
            with st.status("🧠 Training PPO Agent (40k steps)...", expanded=True) as status:
                st.write("⏳ **Training 40,000 timesteps.**")
                st.warning("⏱️ **PLEASE WAIT AT LEAST 3-4 MINUTES.** PPO agent is optimizing neural network parameters.")
                try:
                    rl_metrics = run_rl_pipeline(
                        selected_asset,
                        start_d.strftime("%Y-%m-%d"),
                        end_d.strftime("%Y-%m-%d")
                    )
                    st.session_state.rl_results = rl_metrics
                    st.session_state.last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    status.update(label="✅ PPO RL Agent Trained Successfully!", state="complete", expanded=False)
                    st.success("PPO Agent trained & evaluated!")
                except Exception as e:
                    status.update(label="❌ Training Failed", state="error")
                    st.error(f"RL Training failed: {e}")

        if st.button("🔄 Refresh Dashboard Cache", width="stretch", help="Clear Streamlit cache memory"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")
            st.rerun()

        # System Readiness Indicators Summary
        st.markdown("---")
        st.markdown("<div class='sidebar-section-title'>📌 System State Tracker</div>", unsafe_allow_html=True)
        has_data = "✅ Loaded" if st.session_state.get("market_data") is not None else "⚪ Pending"
        has_model = "✅ Trained" if st.session_state.get("ml_predictions") is not None else "⚪ Pending"
        has_bt = "✅ Completed" if st.session_state.get("backtest_results") is not None else "⚪ Pending"
        has_rl = "✅ Trained" if st.session_state.get("rl_results") is not None else "⚪ Pending"
        
        st.markdown(f"- **Data State:** {has_data}")
        st.markdown(f"- **ML Model:** {has_model}")
        st.markdown(f"- **Backtest:** {has_bt}")
        st.markdown(f"- **RL Agent:** {has_rl}")

        return selected_page
