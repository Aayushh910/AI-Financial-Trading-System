# rl_agents/evaluate_rl.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from stable_baselines3 import PPO
from rl_agents.trading_env import TradingEnv


def evaluate_agent(data):

    env = TradingEnv(data)

    model = PPO.load(
        "ppo_trading_agent"
    )

    obs, _ = env.reset()

    total_reward = 0.0

    portfolio_values = []
    positions = []

    initial_balance = env.initial_balance

    # Track discrete trade round-trips for realistic metrics
    trade_returns = []
    in_trade = False
    entry_price = 0.0
    prev_pos = 0

    while True:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        curr_pos = info["position"]
        curr_price = info["price"]

        # Position entered (0 -> 1)
        if prev_pos == 0 and curr_pos == 1:
            in_trade = True
            entry_price = curr_price * (1 + env.slippage)

        # Position exited (1 -> 0)
        elif prev_pos == 1 and curr_pos == 0 and in_trade:
            exit_price = curr_price * (1 - env.slippage)
            net_trade_ret = ((exit_price * (1 - env.transaction_cost)) /
                             (entry_price * (1 + env.transaction_cost))) - 1.0
            trade_returns.append(net_trade_ret)
            in_trade = False

        prev_pos = curr_pos

        total_reward += reward

        portfolio_values.append(
            info["portfolio_value"]
        )

        positions.append(
            info["position"]
        )

        if terminated or truncated:
            # Handle open position at the end of episode
            if in_trade:
                exit_price = curr_price * (1 - env.slippage)
                net_trade_ret = ((exit_price * (1 - env.transaction_cost)) /
                                 (entry_price * (1 + env.transaction_cost))) - 1.0
                trade_returns.append(net_trade_ret)
            break

    final_portfolio_value = portfolio_values[-1]

    total_return = (
        (
            final_portfolio_value /
            initial_balance
        ) - 1
    ) * 100

    max_portfolio_value = max(
        portfolio_values
    )

    min_portfolio_value = min(
        portfolio_values
    )

    equity_curve = np.array(
        portfolio_values
    )

    running_max = np.maximum.accumulate(
        equity_curve
    )

    drawdown = (
        equity_curve /
        running_max
    ) - 1

    max_drawdown = (
        drawdown.min() * 100
    )

    # Accurate Trade Metrics
    total_trades = len(trade_returns)
    winning_trades = len([r for r in trade_returns if r > 0])
    losing_trades = len([r for r in trade_returns if r <= 0])

    win_rate = (
        (winning_trades / total_trades * 100)
        if total_trades > 0
        else 0.0
    )

    avg_trade_return = (
        np.mean(trade_returns)
        if total_trades > 0
        else 0.0
    )

    # Daily Return Metrics
    step_returns = np.diff(portfolio_values) / (np.array(portfolio_values[:-1]) + 1e-8)
    sharpe = (np.mean(step_returns) / (np.std(step_returns) + 1e-8)) * np.sqrt(252)

    downside = step_returns[step_returns < 0]
    sortino = (np.mean(step_returns) / (np.std(downside) + 1e-8)) * np.sqrt(252) if len(downside) > 0 else 0.0

    gross_profit = sum([r for r in trade_returns if r > 0])
    gross_loss = abs(sum([r for r in trade_returns if r < 0]))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    years = max(len(portfolio_values) / 252.0, 0.1)
    cagr = (((final_portfolio_value / initial_balance) ** (1 / years)) - 1) * 100

    print("\n===== RL Evaluation =====")

    print(
        "Total Reward:",
        round(total_reward, 4)
    )

    print(
        "Final Portfolio Value:",
        round(final_portfolio_value, 2)
    )

    print(
        "Profit/Loss:",
        round(
            final_portfolio_value -
            initial_balance,
            2
        )
    )

    print(
        "Return (%):",
        round(total_return, 2)
    )

    print(
        "CAGR (%):",
        round(cagr, 2)
    )

    print(
        "Sharpe Ratio:",
        round(sharpe, 4)
    )

    print(
        "Sortino Ratio:",
        round(sortino, 4)
    )

    print(
        "Max Drawdown (%):",
        round(max_drawdown, 2)
    )

    print(
        "Total Closed Trades:",
        total_trades
    )

    print(
        "Winning Trades:",
        winning_trades
    )

    print(
        "Losing Trades:",
        losing_trades
    )

    print(
        "Win Rate (%):",
        round(win_rate, 2)
    )

    print(
        "Profit Factor:",
        round(profit_factor, 4)
    )

    print(
        "Average Trade Return (%):",
        round(avg_trade_return * 100, 4)
    )

    print(
        "Final Position:",
        info["position"]
    )

    print(
        "Max Portfolio Value:",
        round(max_portfolio_value, 2)
    )

    print(
        "Min Portfolio Value:",
        round(min_portfolio_value, 2)
    )

    os.makedirs(
        "outputs/charts",
        exist_ok=True
    )

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    # Save RL metrics report
    with open("outputs/reports/rl_metrics.txt", "w") as f:
        f.write("===== RL AGENT EVALUATION METRICS =====\n")
        f.write(f"Total Reward: {total_reward:.4f}\n")
        f.write(f"Final Portfolio Value: {final_portfolio_value:.2f}\n")
        f.write(f"Return (%): {total_return:.2f}\n")
        f.write(f"CAGR (%): {cagr:.2f}\n")
        f.write(f"Sharpe Ratio: {sharpe:.4f}\n")
        f.write(f"Sortino Ratio: {sortino:.4f}\n")
        f.write(f"Max Drawdown (%): {max_drawdown:.2f}\n")
        f.write(f"Total Closed Trades: {total_trades}\n")
        f.write(f"Winning Trades: {winning_trades}\n")
        f.write(f"Losing Trades: {losing_trades}\n")
        f.write(f"Win Rate (%): {win_rate:.2f}\n")
        f.write(f"Profit Factor: {profit_factor:.4f}\n")

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        portfolio_values,
        label="RL Equity Curve",
        color="blue"
    )

    plt.title(
        "RL Agent Portfolio Value"
    )

    plt.xlabel(
        "Steps"
    )

    plt.ylabel(
        "Portfolio Value"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "outputs/charts/rl_equity_curve.png",
        bbox_inches="tight"
    )

    plt.close()

    return {
        "total_reward": total_reward,
        "final_portfolio_value": final_portfolio_value,
        "return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr
    }