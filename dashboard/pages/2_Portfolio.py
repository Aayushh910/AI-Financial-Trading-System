import streamlit as st
import pandas as pd

st.title("💼 Portfolio Optimization")

st.markdown("### Optimized Portfolio Allocation")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Expected Return",
        "0.104%"
    )

with col2:
    st.metric(
        "Portfolio Risk",
        "1.55%"
    )

with col3:
    st.metric(
        "Sharpe Ratio",
        "0.067"
    )

st.divider()

st.subheader("📊 Asset Allocation")

weights = pd.DataFrame(
    {
        "Asset": ["AAPL", "MSFT", "GOOG"],
        "Weight (%)": [36.3, 25.2, 38.4]
    }
)

st.dataframe(
    weights,
    width="stretch"
)

st.bar_chart(
    weights.set_index("Asset")
)

st.divider()

st.subheader("📈 Portfolio Summary")

st.success("""
The optimizer distributes capital across multiple assets
to maximize expected return while controlling portfolio risk.

Current Allocation:

• AAPL → 36.3%

• MSFT → 25.2%

• GOOG → 38.4%
""")

st.info("""
Portfolio optimization uses historical returns and covariance
to determine the most efficient asset allocation.
""")