# dashboard/utils/state.py
import streamlit as st
import datetime

def init_session_state():
    """Initialize all session state variables for the financial platform."""
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "🏠 Home"
    
    if "selected_asset" not in st.session_state:
        st.session_state.selected_asset = "AAPL"
        
    if "selected_assets_portfolio" not in st.session_state:
        st.session_state.selected_assets_portfolio = ["AAPL", "MSFT", "GOOG"]
        
    if "start_date" not in st.session_state:
        st.session_state.start_date = datetime.date(2015, 1, 1)
        
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.date.today()
        
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = "XGBoost"
        
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Dark"
        
    # Data states
    if "market_data" not in st.session_state:
        st.session_state.market_data = None
        
    if "ml_predictions" not in st.session_state:
        st.session_state.ml_predictions = None
        
    if "ml_metrics" not in st.session_state:
        st.session_state.ml_metrics = None
        
    if "ml_feature_importance" not in st.session_state:
        st.session_state.ml_feature_importance = None
        
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None
        
    if "portfolio_results" not in st.session_state:
        st.session_state.portfolio_results = None
        
    if "rl_results" not in st.session_state:
        st.session_state.rl_results = None
        
    if "last_updated" not in st.session_state:
        st.session_state.last_updated = None
