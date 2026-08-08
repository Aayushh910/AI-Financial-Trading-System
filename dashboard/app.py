# dashboard/app.py

import os
import sys
import streamlit as st

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.styles.theme import apply_theme
from dashboard.utils.state import init_session_state
from dashboard.components.sidebar import render_sidebar

from dashboard.views.home import render_home_page
from dashboard.views.market_analysis import render_market_analysis_page
from dashboard.views.machine_learning import render_machine_learning_page
from dashboard.views.portfolio import render_portfolio_page
from dashboard.views.risk import render_risk_page
from dashboard.views.backtesting import render_backtesting_page
from dashboard.views.reinforcement_learning import render_reinforcement_learning_page
from dashboard.views.about import render_about_page

st.set_page_config(
    page_title="AI Financial Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Initialize session state
init_session_state()

# 2. Render Centralized Sidebar Control Panel & Navigation
selected_page = render_sidebar()

# 3. Apply Theme CSS
is_dark = st.session_state.get("theme_mode", "Dark") == "Dark"
apply_theme(is_dark=is_dark)

# 4. Dynamic Lazy Router (Only renders selected page)
if selected_page == "🏠 Home":
    render_home_page()
elif selected_page == "📊 Market Analysis":
    render_market_analysis_page()
elif selected_page == "🤖 Machine Learning":
    render_machine_learning_page()
elif selected_page == "💼 Portfolio Optimization":
    render_portfolio_page()
elif selected_page == "⚠ Risk Management":
    render_risk_page()
elif selected_page == "📈 Backtesting":
    render_backtesting_page()
elif selected_page == "🧠 Reinforcement Learning":
    render_reinforcement_learning_page()
elif selected_page == "ℹ About":
    render_about_page()