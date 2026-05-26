import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def objective(trial, X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 50, 200),
        max_depth=trial.suggest_int("max_depth", 3, 10)
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return mean_squared_error(y_test, preds)

def tune_model(X_train, y_train, X_test, y_test):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X_train, y_train, X_test, y_test), n_trials=20)
    return study.best_params