import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):

    def __init__(self, data):
        super().__init__()

        self.data = data.reset_index(drop=True)

        self.initial_balance = 10000

        self.action_space = spaces.Discrete(3)

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
        self.shares = 0

        self.position = 0
        self.entry_price = 0

        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance
        self.max_balance = self.initial_balance

        return self._next_observation(), {}

    def _next_observation(self):

        row = self.data.iloc[self.current_step]

        obs = np.array([
            row["Lag_1"],
            row["Lag_2"],
            row["Lag_3"],
            row["Lag_5"],
            row["Momentum"],
            row["Rolling_STD"],
            row["RSI"],
            row["MACD"],
            row["MACD_SIGNAL"],
            row["BB_HIGH"],
            row["BB_LOW"],
            row["ATR"],
            row["Volume_Change"],
            row["EMA_10"],
            row["EMA_20"],
            row["EMA_50"],
            row["SMA_10"],
            row["SMA_20"],
            row["SMA_50"],
            row["Price_Range"],
            row["Volume_MA"],
            self.cash,
            self.shares,
            self.portfolio_value
        ], dtype=np.float32)

        return obs

    def step(self, action):

        current_price = float(
            self.data["Close"].iloc[self.current_step]
        )

        transaction_cost = 0.001
        slippage = 0.0005
        position_size = 0.95

        trade_cost = 0

        # BUY
        if action == 1 and self.cash > 0:

            invest_amount = (
                self.cash * position_size
            )

            executed_price = (
                current_price * (1 + slippage)
            )

            bought_shares = (
                invest_amount / executed_price
            )

            self.shares += bought_shares

            self.cash -= invest_amount

            trade_cost = (
                invest_amount * transaction_cost
            )

            self.cash -= trade_cost

            self.position = 1
            self.entry_price = executed_price

        # SELL
        elif action == 2 and self.shares > 0:

            executed_price = (
                current_price * (1 - slippage)
            )

            sell_value = (
                self.shares * executed_price
            )

            trade_cost = (
                sell_value * transaction_cost
            )

            self.cash += (
                sell_value - trade_cost
            )

            self.shares = 0

            self.position = 0

        self.portfolio_value = (
            self.cash +
            self.shares * current_price
        )

        reward = (
            self.portfolio_value -
            self.prev_portfolio_value
        ) / (
            self.prev_portfolio_value + 1e-8
        )

        reward = np.clip(
            reward,
            -1,
            1
        )

        self.max_balance = max(
            self.max_balance,
            self.portfolio_value
        )

        self.prev_portfolio_value = (
            self.portfolio_value
        )

        self.current_step += 1

        terminated = (
            self.current_step >= len(self.data) - 1
        )

        truncated = False

        obs = self._next_observation()

        info = {
            "cash": round(self.cash, 2),
            "shares": round(self.shares, 4),
            "portfolio_value": round(
                self.portfolio_value, 2
            ),
            "position": self.position,
            "max_balance": round(
                self.max_balance, 2
            )
        }

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )