from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

FEATURES = [
    'SMA_20',
    'EMA_20',
    'RSI',
    'Volatility',
    'Momentum',
    'Volume_Change'
]

def train_model(data):

    X = data[FEATURES]
    y = data['Target']

    split = int(len(data) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    print("MSE:", mse)

    joblib.dump(
        model,
        "models/random_forest.pkl"
    )

    return model, X_test