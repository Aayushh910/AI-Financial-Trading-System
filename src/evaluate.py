import numpy as np

def evaluate_strategy(data):

    sharpe = (
        data['Strategy_Returns'].mean() /
        data['Strategy_Returns'].std()
    ) * np.sqrt(252)

    cumulative = data['Equity_Curve']

    rolling_max = cumulative.cummax()

    drawdown = (
        cumulative - rolling_max
    ) / rolling_max

    max_drawdown = drawdown.min()

    total_return = (
        cumulative.iloc[-1] - 1
    )

    print("Sharpe Ratio:", sharpe)
    print("Max Drawdown:", max_drawdown)
    print("Total Return:", total_return)

    with open(
        "outputs/metrics.txt",
        "w"
    ) as f:

        f.write(
            f"Sharpe Ratio: {sharpe}\n"
        )

        f.write(
            f"Max Drawdown: {max_drawdown}\n"
        )

        f.write(
            f"Total Return: {total_return}\n"
        )