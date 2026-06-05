import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )
)

import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import plotly.graph_objects as go

from features.feature_engineering import create_features

st.title("🔮 Live Market Prediction")

st.markdown("""
Predict the next-day market movement using the trained
XGBoost model and technical indicators.
""")

ticker = st.text_input(
    "📌 Enter Stock Symbol",
    value="AAPL"
)

if st.button("Predict Market Direction"):

    try:

        with st.spinner("Fetching market data..."):

            data = yf.download(
                ticker,
                period="1y",
                auto_adjust=False
            )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = create_features(data)

        features = [
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Lag_5",
            "Momentum",
            "Rolling_STD",
            "RSI",
            "MACD",
            "MACD_SIGNAL",
            "BB_HIGH",
            "BB_LOW",
            "ATR",
            "Volume_Change",
            "EMA_10",
            "EMA_20",
            "EMA_50",
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "Price_Range",
            "Volume_MA"
        ]

        latest = data[features].tail(1)

        scaler = joblib.load(
            "saved_models/scaler.pkl"
        )

        model = joblib.load(
            "saved_models/xgb_model.pkl"
        )

        latest_scaled = scaler.transform(
            latest
        )

        prediction = model.predict(
            latest_scaled
        )[0]

        st.divider()

        st.subheader("📈 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Return",
                f"{prediction:.5f}"
            )

        with col2:
            current_price = round(
                data["Close"].iloc[-1],
                2
            )

            st.metric(
                "Current Price",
                f"${current_price}"
            )

        if prediction > 0.001:

            st.success(
                "📈 BUY Signal Generated"
            )

            signal = "BUY"

        elif prediction < -0.001:

            st.error(
                "📉 SELL Signal Generated"
            )

            signal = "SELL"

        else:

            st.warning(
                "⏸ HOLD Signal Generated"
            )

            signal = "HOLD"

        st.divider()

        st.subheader("📊 Latest Market Indicators")

        indicator_df = pd.DataFrame({
            "Indicator": [
                "RSI",
                "MACD",
                "ATR",
                "Momentum"
            ],
            "Value": [
                round(data["RSI"].iloc[-1], 2),
                round(data["MACD"].iloc[-1], 4),
                round(data["ATR"].iloc[-1], 4),
                round(data["Momentum"].iloc[-1], 4)
            ]
        })

        st.dataframe(
            indicator_df,
            width="stretch"
        )

        st.divider()

        st.subheader("📉 Price Chart")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data.index[-100:],
                y=data["Close"].tail(100),
                mode="lines",
                name="Close Price"
            )
        )

        fig.update_layout(
            height=500,
            title=f"{ticker} Price Trend",
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.divider()

        st.subheader("📝 AI Recommendation")

        st.info(
            f"""
            Ticker: {ticker}

            Signal: {signal}

            Predicted Return: {prediction:.5f}

            Recommendation generated using the
            trained XGBoost prediction model.
            """
        )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )