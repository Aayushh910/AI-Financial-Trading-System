# rl_agents/trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):

    def __init__(self, data):
        super().__init__()

        self.data = data.reset_index(drop=True)

        self.initial_balance = 10000
        self.transaction_cost = 0.001
        self.slippage = 0.0005
        self.position_size = 0.95

        self.action_space = spaces.Discrete(3)
        # 0 = Hold, 1 = Buy, 2 = Sell

        self.feature_columns = [
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Lag_5",
            "Momentum",
            "Rolling_STD",
            "RSI",
            "MACD",
            "MACD_SIGNAL",
            "BB_HIGH",
            "BB_LOW",
            "ATR",
            "Volume_Change",
            "EMA_10",
            "EMA_20",
            "EMA_50",
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "Price_Range",
            "Volume_MA"
        ]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.cash = self.initial_balance
        self.shares = 0.0

        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance

        self.position = 0
        self.entry_price = 0.0

        return self._next_observation(), {}

    def _next_observation(self):

        row = self.data.iloc[self.current_step]

        obs = [row[col] for col in self.feature_columns]

        obs.extend([
            self.cash,
            self.shares,
            self.portfolio_value
        ])

        return np.array(obs, dtype=np.float32)

    def step(self, action):

        current_price = float(
            self.data["Close"].iloc[self.current_step]
        )

        if self.current_step > 0:
            previous_price = float(
                self.data["Close"].iloc[self.current_step - 1]
            )
        else:
            previous_price = current_price

        trade_cost = 0.0

        # BUY
        if action == 1 and self.cash > 0:

            invest_amount = self.cash * self.position_size

            execution_price = current_price * (
                1 + self.slippage
            )

            shares_bought = (
                invest_amount / execution_price
            )

            trade_cost = invest_amount * (
                self.transaction_cost
            )

            self.cash -= (
                invest_amount + trade_cost
            )

            self.shares += shares_bought

            self.position = 1
            self.entry_price = execution_price

        # SELL
        elif action == 2 and self.shares > 0:

            execution_price = current_price * (
                1 - self.slippage
            )

            sell_value = (
                self.shares * execution_price
            )

            trade_cost = sell_value * (
                self.transaction_cost
            )

            self.cash += (
                sell_value - trade_cost
            )

            self.shares = 0.0

            self.position = 0
            self.entry_price = 0.0

        # Portfolio value
        self.portfolio_value = (
            self.cash +
            self.shares * current_price
        )

        # Reward = portfolio growth
        reward = (
            self.portfolio_value -
            self.prev_portfolio_value
        ) / (
            self.prev_portfolio_value + 1e-8
        )

        # Small penalty for trading
        reward -= (
            trade_cost /
            (self.prev_portfolio_value + 1e-8)
        )

        reward = float(
            np.clip(reward, -1, 1)
        )

        self.prev_portfolio_value = (
            self.portfolio_value
        )

        self.current_step += 1

        terminated = (
            self.current_step >= len(self.data) - 1
        )

        truncated = False

        if terminated:
            obs = np.zeros(
                self.observation_space.shape,
                dtype=np.float32
            )
        else:
            obs = self._next_observation()

        info = {
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "position": self.position
        }

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )