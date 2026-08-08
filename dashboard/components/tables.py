# dashboard/components/tables.py

import streamlit as st
import pandas as pd


def render_signals_table(pred_df, signals=None):
    """Render table of generated model signals."""
    if pred_df is None or pred_df.empty:
        st.info("No predictions or signals available yet.")
        return
        
    df_display = pred_df.copy()
    if signals is not None and len(signals) == len(df_display):
        labels = ["BUY" if s == 1 else ("SELL" if s == -1 else "HOLD") for s in signals]
        df_display["Signal"] = labels
        
    st.dataframe(
        df_display.tail(20),
        width="stretch",
        hide_index=True
    )


def render_trade_history_table(trade_df):
    """Render table of trade execution history."""
    if trade_df is None or trade_df.empty:
        st.info("No trade history available. Run backtest first.")
        return

    st.dataframe(
        trade_df,
        width="stretch",
        hide_index=True
    )


def render_portfolio_table(weights_df):
    """Render table of optimal portfolio weights."""
    if weights_df is None or weights_df.empty:
        st.info("No portfolio weights available. Run optimization first.")
        return

    st.dataframe(
        weights_df,
        width="stretch",
        hide_index=True
    )


def render_risk_metrics_table(metrics_list):
    """Render structured table of risk parameters."""
    if not metrics_list:
        st.info("No risk metrics computed yet.")
        return

    df = pd.DataFrame(metrics_list)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True
    )
