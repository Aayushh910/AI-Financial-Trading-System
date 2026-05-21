from src.data_loader import download_data
from src.indicators import add_indicators
from src.feature_engineering import create_features
from src.train_model import train_model
from src.backtest import run_backtest
from src.evaluate import evaluate_strategy
from src.utils import plot_equity_curve

def main():

    # Step 1 — Download data
    data = download_data()

    # Step 2 — Indicators
    data = add_indicators(data)

    # Step 3 — Features
    data = create_features(data)

    # Save processed data
    data.to_csv(
        "data/processed/processed_data.csv"
    )

    # Step 4 — Train model
    model, X_test = train_model(data)

    # Step 5 — Backtest
    results = run_backtest(
        data,
        model,
        X_test
    )

    # Step 6 — Evaluate
    evaluate_strategy(results)

    # Step 7 — Plot
    plot_equity_curve(results)

    # Save predictions
    results.to_csv(
        "outputs/predictions.csv"
    )

if __name__ == "__main__":
    main()