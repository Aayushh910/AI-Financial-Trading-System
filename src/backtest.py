import pandas as pd

def run_backtest(
    data,
    model,
    X_test
):

    predictions = model.predict(X_test)

    test_data = data.iloc[-len(X_test):].copy()

    test_data['Prediction'] = predictions

    # Trading signal
    test_data['Signal'] = (
        test_data['Prediction'] > 0
    ).astype(int)

    # Strategy returns
    test_data['Strategy_Returns'] = (
        test_data['Signal'].shift(1) *
        test_data['Returns']
    )

    # Equity curve
    test_data['Equity_Curve'] = (
        1 +
        test_data['Strategy_Returns']
    ).cumprod()

    return test_data