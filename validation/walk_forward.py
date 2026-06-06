import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


def walk_forward_validation(data, features):

    window_size = 252 * 4
    test_size = 252

    maes = []
    mses = []
    returns_list = []
    sharpe_list = []

    start = 0

    while start + window_size + test_size < len(data):

        train = data.iloc[
            start:start + window_size
        ]

        test = data.iloc[
            start + window_size:
            start + window_size + test_size
        ]

        X_train = train[features]
        y_train = train["Target"]

        X_test = test[features]
        y_test = test["Target"]

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror"
        )

        model.fit(
            X_train,
            y_train
        )

        preds = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            preds
        )

        mse = mean_squared_error(
            y_test,
            preds
        )

        maes.append(mae)
        mses.append(mse)

        signals = np.where(
            preds > 0,
            1,
            -1
        )

        strategy_returns = (
            y_test.reset_index(drop=True)
            * signals
        )

        equity_curve = (
            1 + strategy_returns
        ).cumprod()

        total_return = (
            equity_curve.iloc[-1] - 1
        )

        sharpe = (
            strategy_returns.mean()
            /
            (
                strategy_returns.std()
                + 1e-8
            )
        )

        returns_list.append(
            total_return
        )

        sharpe_list.append(
            sharpe
        )

        start += test_size

    results = {

        "MAE": np.mean(maes),

        "MSE": np.mean(mses),

        "Return": np.mean(
            returns_list
        ),

        "Sharpe": np.mean(
            sharpe_list
        )
    }

    print("\n===== Walk Forward Validation =====")

    for k, v in results.items():

        print(
            f"{k}: {round(v, 6)}"
        )

    return results