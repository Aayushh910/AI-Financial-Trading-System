import numpy as np
from scipy.optimize import minimize


def optimize_portfolio(returns):

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(mean_returns)

    def portfolio_performance(weights):

        portfolio_return = np.dot(
            weights,
            mean_returns
        )

        portfolio_risk = np.sqrt(
            np.dot(
                weights.T,
                np.dot(
                    cov_matrix,
                    weights
                )
            )
        )

        sharpe = portfolio_return / (
            portfolio_risk + 1e-8
        )

        return (
            portfolio_return,
            portfolio_risk,
            sharpe
        )

    def negative_sharpe(weights):

        return -portfolio_performance(
            weights
        )[2]

    constraints = (
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1
        },
    )

    bounds = tuple(
        (0, 1)
        for _ in range(num_assets)
    )

    initial_weights = np.ones(
        num_assets
    ) / num_assets

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    weights = result.x

    portfolio_return, portfolio_risk, sharpe = (
        portfolio_performance(weights)
    )

    print("\n===== Portfolio Optimization =====")
    print("Weights:", weights)
    print("Expected Return:", portfolio_return)
    print("Risk:", portfolio_risk)
    print("Sharpe:", sharpe)

    return (
        weights,
        portfolio_return,
        portfolio_risk
    )