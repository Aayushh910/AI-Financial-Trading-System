# dashboard/views/portfolio.py

import streamlit as st
import datetime
import numpy as np
import plotly.graph_objects as go

from dashboard.components.cards import render_section_header, render_kpi_card
from dashboard.components.charts import create_portfolio_pie_chart, apply_trading_chart_theme
from dashboard.components.tables import render_portfolio_table
from dashboard.utils.loaders import run_portfolio_pipeline
from dashboard.utils.formatting import format_pct, format_number


def render_portfolio_page():
    st.title("💼 Portfolio Optimization Terminal")
    st.caption("Modern Portfolio Theory (MPT) asset allocation, Sharpe ratio maximization, risk diversification, and efficient frontier.")

    start_date = st.session_state.get("start_date", datetime.date(2015, 1, 1))
    end_date = st.session_state.get("end_date", datetime.date.today())
    is_dark = st.session_state.get("theme_mode", "Dark") == "Dark"

    # Top Controls Bar
    col_p1, col_p2, col_p3 = st.columns([2, 2, 1])
    with col_p1:
        selected_assets = st.multiselect(
            "Select Portfolio Universe",
            ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA"],
            default=st.session_state.get("selected_assets_portfolio", ["AAPL", "MSFT", "GOOG"]),
            help="Select assets to include in mean-variance covariance optimization."
        )
        st.session_state.selected_assets_portfolio = selected_assets
    with col_p2:
        opt_method = st.selectbox(
            "Optimization Model",
            ["Maximum Sharpe Ratio (SLSQP)", "Minimum Variance", "Equal Weight Benchmark", "Risk Parity — (Coming Soon)"],
            help="Mathematical objective function for capital allocation."
        )
    with col_p3:
        st.markdown("<br>", unsafe_allow_html=True)
        recalc_clicked = st.button("🚀 Optimize Portfolio", width="stretch")

    st.markdown("---")

    # Run optimization if button clicked or if result doesn't exist
    if recalc_clicked or st.session_state.get("portfolio_results") is None:
        if len(selected_assets) < 2:
            st.warning("Please select at least 2 assets to run portfolio optimization.")
            return
        with st.spinner("Executing SLSQP Mean-Variance Portfolio Optimization..."):
            try:
                res = run_portfolio_pipeline(
                    assets=selected_assets,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d")
                )
                st.session_state.portfolio_results = res
            except Exception as e:
                st.error(f"Portfolio optimization failed: {e}")

    port_res = st.session_state.get("portfolio_results")

    if port_res is not None:
        # 1. KPI Header
        render_section_header("📊 Optimized Portfolio Analytics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi_card("Annualized Expected Return", format_pct(port_res["expected_return"] * 100))
        with col2:
            render_kpi_card("Annualized Volatility (Risk)", format_pct(port_res["portfolio_risk"] * 100))
        with col3:
            render_kpi_card("Optimal Sharpe Ratio", format_number(port_res["sharpe_ratio"], 3))
        with col4:
            render_kpi_card("Asset Count", str(len(port_res["assets"])))

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Allocation Charts (Donut + Table)
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            render_section_header("🍩 Optimal Asset Weights")
            fig_pie = create_portfolio_pie_chart(port_res["weights_df"], is_dark=is_dark)
            st.plotly_chart(fig_pie, width="stretch")

        with col_w2:
            render_section_header("📋 Asset Allocation Breakdown Table")
            render_portfolio_table(port_res["weights_df"])

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Risk vs Return Scatter Plot & Efficient Frontier Visualization
        render_section_header("📈 Asset Risk vs Return Profile & Diversification Frontier")
        
        returns_mat = port_res["returns_matrix"]
        annual_returns = returns_mat.mean() * 252
        annual_risks = returns_mat.std() * np.sqrt(252)

        fig_scat = go.Figure()
        
        # Individual Assets
        fig_scat.add_trace(go.Scatter(
            x=annual_risks.values * 100,
            y=annual_returns.values * 100,
            mode="markers+text",
            text=returns_mat.columns,
            textposition="top center",
            name="Individual Assets",
            marker=dict(size=12, color="#58a6ff")
        ))
        
        # Optimized Portfolio Point
        fig_scat.add_trace(go.Scatter(
            x=[port_res["portfolio_risk"] * 100],
            y=[port_res["expected_return"] * 100],
            mode="markers+text",
            text=["Optimal Portfolio"],
            textposition="top right",
            name="Optimal Portfolio (Max Sharpe)",
            marker=dict(size=16, color="#2ea043", symbol="star")
        ))

        fig_scat.update_layout(
            xaxis_title="Annualized Risk / Volatility (%)",
            yaxis_title="Annualized Expected Return (%)"
        )
        fig_scat = apply_trading_chart_theme(fig_scat, is_dark=is_dark, title="Risk vs Return Scatter Plot")
        st.plotly_chart(fig_scat, width="stretch")

        with st.expander("💡 Modern Portfolio Theory (MPT) & Sharpe Optimization Guide"):
            st.markdown("""
            **Understanding Mean-Variance Portfolio Optimization:**
            - **Objective (SLSQP)**: Finds exact portfolio weights $w$ that maximize the risk-adjusted Sharpe Ratio $\\frac{E[R_p]}{\\sigma_p}$ subject to $\\sum w_i = 1$ and $w_i \\ge 0$.
            - **Diversification Benefit**: Notice how the Optimal Portfolio star typically achieves higher return per unit of risk than individual holdings due to covariance cancellation.
            - **Constraint**: Long-only constrained weights ($0 \\le w_i \\le 1$).
            """)
    else:
        st.info("Click 'Optimize Portfolio' to generate portfolio allocations.")
