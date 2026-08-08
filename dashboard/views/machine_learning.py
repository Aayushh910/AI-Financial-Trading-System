# dashboard/views/machine_learning.py

import streamlit as st

from dashboard.components.cards import render_section_header, render_kpi_card
from dashboard.components.metrics import render_ml_metrics_grid
from dashboard.components.charts import (
    create_actual_vs_predicted_chart,
    create_feature_importance_chart,
    create_residual_chart
)
from dashboard.components.tables import render_signals_table


def render_machine_learning_page():
    st.title("🤖 Machine Learning Research Terminal")
    st.caption("XGBoost market direction prediction, feature importance ranking, residual analytics, and signal evaluation.")

    is_dark = st.session_state.get("theme_mode", "Dark") == "Dark"

    pred_df = st.session_state.get("ml_predictions")
    metrics_dict = st.session_state.get("ml_metrics")
    importance_df = st.session_state.get("ml_feature_importance")
    bt_results = st.session_state.get("backtest_results")

    # Top Section: Model Information Card
    render_section_header("ℹ Model Architecture & Specifications")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Model Architecture", "XGBoost Regressor")
    with col2:
        render_kpi_card("Target Variable", "Next-Day Return")
    with col3:
        render_kpi_card("Feature Count", "21 Technical Features")
    with col4:
        render_kpi_card("Train / Test Split", "80% / 20%")
    with col5:
        samples_count = str(len(pred_df)) if pred_df is not None else "N/A"
        render_kpi_card("Test Sample Count", samples_count)

    st.markdown("<br>", unsafe_allow_html=True)

    if pred_df is not None and metrics_dict is not None:
        # 1. Evaluation Metrics Grid
        render_section_header("📊 Machine Learning Performance Metrics")
        render_ml_metrics_grid(metrics_dict)

        with st.expander("💡 Machine Learning Metrics Guidance & Benchmarks"):
            st.markdown("""
            **Understanding ML Regression & Classification Metrics:**
            - **MAE (Mean Absolute Error)**: Average magnitude of prediction errors. Lower values indicate closer alignment to true returns.
            - **RMSE (Root Mean Squared Error)**: Penalizes larger prediction errors more heavily than MAE.
            - **R² Score (Coefficient of Determination)**: Proportion of return variance explained by features (`> 0.0` beats naive mean prediction).
            - **Directional Accuracy (%)**: Percentage of trading days where the model correctly predicted the sign (+ or -) of market return (`> 52-54%` is statistically significant in financial time series).
            """)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Actual vs Predicted Chart
        render_section_header("📈 Actual Target Return vs Model Prediction")
        fig_act_pred = create_actual_vs_predicted_chart(pred_df.tail(150), is_dark=is_dark)
        st.plotly_chart(fig_act_pred, width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Residuals & Feature Importance (2 Columns)
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            render_section_header("🎯 Feature Importance Ranking")
            if importance_df is not None:
                fig_imp = create_feature_importance_chart(importance_df, is_dark=is_dark)
                st.plotly_chart(fig_imp, width="stretch")
            else:
                st.info("Feature importance data unavailable.")

        with col_f2:
            render_section_header("📉 Residual Scatter Plot")
            fig_res = create_residual_chart(pred_df, is_dark=is_dark)
            st.plotly_chart(fig_res, width="stretch")

        with st.expander("💡 Feature Importance & Residual Diagnostics"):
            st.markdown("""
            - **Feature Importance**: Higher Gini gain importance signifies that the XGBoost tree nodes frequently use that technical indicator to split return distributions.
            - **Residual Scatter Plot**: Residuals (`Actual - Predicted`) should be randomly centered around zero (`0`). Funnel shapes or systematic patterns indicate potential heteroscedasticity or unmodeled regime shifts.
            """)

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. Generated Signals Dataframe
        render_section_header("📋 Recent Generated Trading Signals Table")
        signals = bt_results.get("signals") if bt_results else None
        render_signals_table(pred_df, signals=signals)
    else:
        st.info("""
        ℹ XGBoost model has not been trained yet for the current session.
        
        Go to the sidebar on the left and click **⚡ Train XGBoost Model** to run feature engineering, train the model, and view predictions.
        """)
