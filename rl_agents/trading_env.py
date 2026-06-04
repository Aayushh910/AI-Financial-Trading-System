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
            shape=(21,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.max_balance = self.initial_balance

        self.current_step = 0

        self.position = 0
        self.entry_price = 0

        return self._next_observation(), {}

    def _next_observation(self):

        row = self.data.iloc[self.current_step]

        obs = np.array([
            row['Lag_1'],
            row['Lag_2'],
            row['Lag_3'],
            row['Lag_5'],
            row['Momentum'],
            row['Rolling_STD'],
            row['RSI'],
            row['MACD'],
            row['MACD_SIGNAL'],
            row['BB_HIGH'],
            row['BB_LOW'],
            row['ATR'],
            row['Volume_Change'],
            row['EMA_10'],
            row['EMA_20'],
            row['EMA_50'],
            row['SMA_10'],
            row['SMA_20'],
            row['SMA_50'],
            row['Price_Range'],
            row['Volume_MA']
        ], dtype=np.float32)

        return obs

    def step(self, action):

        current_price = self.data['Close'].iloc[self.current_step]

        reward = 0

        transaction_cost = 0.0005

        stop_loss = 0.03
        take_profit = 0.06

        if action == 1:

            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                reward -= transaction_cost

        elif action == 2:

            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                reward -= transaction_cost

        if self.position == 1:

            pnl = (
                current_price - self.entry_price
            ) / self.entry_price

            reward += pnl

            if pnl <= -stop_loss or pnl >= take_profit:
                self.balance *= (1 + pnl)
                self.position = 0

        elif self.position == -1:

            pnl = (
                self.entry_price - current_price
            ) / self.entry_price

            reward += pnl

            if pnl <= -stop_loss or pnl >= take_profit:
                self.balance *= (1 + pnl)
                self.position = 0

        reward = np.clip(reward, -1, 1)

        self.max_balance = max(
            self.max_balance,
            self.balance
        )

        self.current_step += 1

        terminated = (
            self.current_step >= len(self.data) - 1
        )

        truncated = False

        obs = self._next_observation()

        info = {
            "balance": self.balance,
            "position": self.position,
            "max_balance": self.max_balance
        }

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )