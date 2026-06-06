# rl_agents/evaluate_rl.py

from stable_baselines3 import PPO
from rl_agents.trading_env import TradingEnv

import matplotlib.pyplot as plt
import numpy as np
import os


def evaluate_agent(data):

    env = TradingEnv(data)

    model = PPO.load(
        "ppo_trading_agent"
    )

    obs, _ = env.reset()

    total_reward = 0

    portfolio_values = []
    positions = []

    initial_balance = env.initial_balance

    while True:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        portfolio_values.append(
            info["portfolio_value"]
        )

        positions.append(
            info["position"]
        )

        if terminated or truncated:
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

    trades = [
        p for p in positions
        if p != 0
    ]

    winning_trades = len([
        x for x in trades
        if x == 1
    ])

    win_rate = (
        winning_trades /
        len(trades) * 100
        if len(trades) > 0
        else 0
    )

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
        "Max Drawdown (%):",
        round(max_drawdown, 2)
    )

    print(
        "Win Rate (%):",
        round(win_rate, 2)
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
        "win_rate": win_rate
    }