import numpy as np

def optimize_portfolio(returns):
    cov_matrix = returns.cov()
    mean_returns = returns.mean()

    weights = np.ones(len(mean_returns)) / len(mean_returns)

    portfolio_return = np.dot(weights, mean_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    return weights, portfolio_return, portfolio_risk