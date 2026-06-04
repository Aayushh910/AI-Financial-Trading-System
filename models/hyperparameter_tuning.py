import optuna

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


def objective(
    trial,
    X_train,
    y_train,
    X_test,
    y_test
):

    model = XGBRegressor(
        n_estimators=trial.suggest_int(
            "n_estimators",
            100,
            1000
        ),

        max_depth=trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        learning_rate=trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True
        ),

        subsample=trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        colsample_bytree=trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        min_child_weight=trial.suggest_int(
            "min_child_weight",
            1,
            10
        ),

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

    mse = mean_squared_error(
        y_test,
        preds
    )

    return mse


def tune_model(
    X_train,
    y_train,
    X_test,
    y_test
):

    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            X_test,
            y_test
        ),
        n_trials=50
    )

    print("\n===== Best XGBoost Parameters =====")
    print(study.best_params)

    print(
        "\nBest MSE:",
        study.best_value
    )

    return study.best_params