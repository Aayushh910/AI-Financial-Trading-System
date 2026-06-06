import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):

    def __init__(self, data):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = 10000
        self.cash = self.initial_balance
        self.shares = 0
        self.position = 0

        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance

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

        self.cash = self.initial_balance
        self.shares = 0

        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance
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
            row['Volume_MA'],
            self.cash,
            self.shares,
            self.portfolio_value
        ], dtype=np.float32)

        return obs

    def step(self, action):
        reward = 0.0
        current_price = self.data['Close'].iloc[self.current_step]

        transaction_cost = 0.001
        slippage = 0.0005

        trade_cost = 0

        position_size = 0.95

        transaction_cost = 0.0005

        stop_loss = 0.03
        take_profit = 0.06

        if action == 1 and self.cash > 0:

            invest_amount = self.cash * position_size

            self.shares += invest_amount / current_price

            self.cash -= invest_amount

            trade_cost = invest_amount * (
                transaction_cost + slippage
            )

            self.position = 1

        elif action == 2 and self.shares > 0:

            sell_value = self.shares * current_price

            self.cash += sell_value

            self.shares = 0

            trade_cost = sell_value * (
                transaction_cost + slippage
            )

            self.position = -1


        else:
            pass

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

        reward -= trade_cost / (
            self.prev_portfolio_value + 1e-8
        )

        reward -= (
            abs(self.shares) * 0.00001
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