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

    portfolio_return, portfolio_risk, _ = portfolio_performance(weights)

    annual_return = portfolio_return * 252
    annual_risk = portfolio_risk * np.sqrt(252)
    annual_sharpe = annual_return / (annual_risk + 1e-8)

    print("\n===== Portfolio Optimization =====")
    print("Weights:", np.round(weights, 4))
    print("Annualized Return:", round(annual_return, 4))
    print("Annualized Risk:", round(annual_risk, 4))
    print("Annualized Sharpe:", round(annual_sharpe, 4))

    return (
        weights,
        annual_return,
        annual_risk
    )