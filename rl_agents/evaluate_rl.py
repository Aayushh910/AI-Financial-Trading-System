from stable_baselines3 import PPO
from rl_agents.trading_env import TradingEnv

import matplotlib.pyplot as plt
import numpy as np


def evaluate_agent(data):

    env = TradingEnv(data)

    model = PPO.load(
        "ppo_trading_agent"
    )

    obs, _ = env.reset()

    total_reward = 0

    balances = []
    positions = []

    initial_balance = 10000

    while True:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        balances.append(
            info["balance"]
        )

        positions.append(
            info["position"]
        )

        if terminated or truncated:
            break

    final_balance = balances[-1]

    total_return = (
        (final_balance / initial_balance) - 1
    ) * 100

    max_balance = max(balances)
    min_balance = min(balances)

    equity_curve = np.array(balances)

    running_max = np.maximum.accumulate(
        equity_curve
    )

    drawdown = (
        equity_curve / running_max
    ) - 1

    max_drawdown = drawdown.min() * 100

    trades = [
        p for p in positions
        if p != 0
    ]

    win_rate = (
        len([x for x in trades if x == 1])
        / len(trades)
        * 100
        if len(trades) > 0
        else 0
    )

    print("\n===== RL Evaluation =====")

    print(
        "Total Reward:",
        round(total_reward, 4)
    )

    print(
        "Final Balance:",
        round(final_balance, 2)
    )

    print(
        "Profit/Loss:",
        round(
            final_balance - initial_balance,
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
        "Max Balance:",
        round(max_balance, 2)
    )

    print(
        "Min Balance:",
        round(min_balance, 2)
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        balances,
        label="RL Equity Curve",
        color="blue"
    )

    plt.title(
        "RL Agent Portfolio Value"
    )

    plt.xlabel("Steps")
    plt.ylabel("Balance")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        "rl_equity_curve.png"
    )

    plt.show()  