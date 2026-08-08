# dashboard/components/cards.py

import streamlit as st

def render_kpi_card(title, value, sub_text=None, delta_color="neutral"):
    """Render sleek modern KPI card container."""
    sub_html = f'<div class="metric-card-sub">{sub_text}</div>' if sub_text else ''
    
    html = f"""
    <div class="metric-card-container">
        <div class="metric-card-title">{title}</div>
        <div class="metric-card-value">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_signal_card(signal="BUY", predicted_return=0.0, current_price=0.0, timestamp=None, model="XGBoost"):
    """Render large prominent trading signal badge component."""
    sig_str = str(signal).upper()
    
    if sig_str in ["BUY", "1", "STRONG BUY"]:
        badge_class = "signal-badge-buy"
        sig_text = "BUY ⬆"
    elif sig_str in ["SELL", "-1", "STRONG SELL"]:
        badge_class = "signal-badge-sell"
        sig_text = "SELL ⬇"
    else:
        badge_class = "signal-badge-hold"
        sig_text = "HOLD ⏸"

    time_str = timestamp if timestamp else "Latest Market Close"
    price_str = f"${current_price:,.2f}" if isinstance(current_price, (int, float)) and current_price > 0 else "N/A"
    ret_str = f"{predicted_return:+.4f}" if isinstance(predicted_return, (int, float)) else "N/A"

    html = f"""
    <div class="signal-card">
        <div style="font-size:0.9rem; font-weight:600; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">
            Current AI Trading Recommendation ({model})
        </div>
        <div class="{badge_class}">
            {sig_text}
        </div>
        <div style="display:flex; justify-content:space-around; margin-top:16px; border-top:1px solid #30363d; padding-top:12px;">
            <div>
                <span style="color:#8b949e; font-size:0.8rem;">Current Price</span><br>
                <strong style="color:#f0f6fc; font-size:1.1rem;">{price_str}</strong>
            </div>
            <div>
                <span style="color:#8b949e; font-size:0.8rem;">Predicted Return</span><br>
                <strong style="color:#f0f6fc; font-size:1.1rem;">{ret_str}</strong>
            </div>
            <div>
                <span style="color:#8b949e; font-size:0.8rem;">Timestamp</span><br>
                <strong style="color:#f0f6fc; font-size:1.1rem;">{time_str}</strong>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_status_badge(text, level="low"):
    """Render risk level badge (Low/Medium/High)."""
    lvl = str(level).lower()
    if lvl == "low":
        css_class = "badge-low-risk"
    elif lvl in ["medium", "med"]:
        css_class = "badge-med-risk"
    else:
        css_class = "badge-high-risk"
        
    html = f'<span class="{css_class}">{text}</span>'
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title):
    """Render custom section header divider."""
    html = f'<div class="section-header">{title}</div>'
    st.markdown(html, unsafe_allow_html=True)
