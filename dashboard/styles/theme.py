# dashboard/styles/theme.py

DARK_THEME_CSS = """
<style>
    /* Main Background & Text Color */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: #0d1117;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #58a6ff;
    }

    /* Sidebar Section Headers */
    .sidebar-section-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 16px;
        margin-bottom: 8px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 4px;
    }

    /* Card Containers */
    div.metric-card-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div.metric-card-container:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-card-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .metric-card-sub {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Signal Card Custom Styling */
    .signal-card {
        background: linear-gradient(135deg, #161b22 0%, #1f242d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    .signal-badge-buy {
        background-color: rgba(46, 160, 67, 0.18);
        color: #3fb950;
        border: 1px solid #2ea043;
        padding: 8px 24px;
        border-radius: 24px;
        font-weight: 800;
        font-size: 1.5rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(46, 160, 67, 0.2);
    }
    .signal-badge-sell {
        background-color: rgba(248, 81, 73, 0.18);
        color: #f85149;
        border: 1px solid #f85149;
        padding: 8px 24px;
        border-radius: 24px;
        font-weight: 800;
        font-size: 1.5rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(248, 81, 73, 0.2);
    }
    .signal-badge-hold {
        background-color: rgba(210, 153, 34, 0.18);
        color: #d29922;
        border: 1px solid #d29922;
        padding: 8px 24px;
        border-radius: 24px;
        font-weight: 800;
        font-size: 1.5rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(210, 153, 34, 0.2);
    }

    /* Status Badges */
    .badge-low-risk {
        color: #3fb950;
        background: rgba(46, 160, 67, 0.12);
        border: 1px solid #2ea043;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-med-risk {
        color: #d29922;
        background: rgba(210, 153, 34, 0.12);
        border: 1px solid #d29922;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-high-risk {
        color: #f85149;
        background: rgba(248, 81, 73, 0.12);
        border: 1px solid #f85149;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
    }

    /* Info & Tip Expanders */
    .stExpander {
        background: #161b22;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        margin-top: 12px;
    }
    .stExpander summary {
        font-weight: 600;
        color: #58a6ff !important;
    }

    /* Custom Headers & Section Dividers */
    .section-header {
        color: #f0f6fc;
        font-size: 1.25rem;
        font-weight: 700;
        border-bottom: 2px solid #30363d;
        padding-bottom: 8px;
        margin-top: 24px;
        margin-bottom: 16px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #363b42;
        border-radius: 6px;
        font-weight: 600;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #30363d;
        color: #58a6ff;
        border-color: #58a6ff;
    }

    /* Tables & Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Metric Overrides */
    div[data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-weight: 600;
    }
</style>
"""

LIGHT_THEME_CSS = """
<style>
    /* Main Background & Text Color */
    .stApp {
        background-color: #f6f8fa;
        color: #24292f;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #d0d7de;
    }

    /* Card Containers */
    div.metric-card-container {
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #57606a;
        text-transform: uppercase;
    }
    .metric-card-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0969da;
    }

    /* Signal Card */
    .signal-card {
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
    }
    .signal-badge-buy {
        background-color: #dafbe1;
        color: #1a7f37;
        border: 1px solid #2da44e;
        padding: 8px 24px;
        border-radius: 24px;
        font-weight: 800;
        font-size: 1.5rem;
    }
    .signal-badge-sell {
        background-color: #ffebe9;
        color: #cf222e;
        border: 1px solid #cf222e;
        padding: 8px 24px;
        border-radius: 24px;
        font-weight: 800;
        font-size: 1.5rem;
    }
    .signal-badge-hold {
        background-color: #fff8c5;
        color: #9a6700;
        border: 1px solid #bf8700;
        padding: 8px 24px;
        border-radius: 24px;
        font-weight: 800;
        font-size: 1.5rem;
    }
</style>
"""


def apply_theme(is_dark=True):
    import streamlit as st
    if is_dark:
        st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)
