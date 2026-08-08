# rl_agents/trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.preprocessing import StandardScaler


class TradingEnv(gym.Env):

    def __init__(self, data):
        super().__init__()

        self.data = data.reset_index(drop=True)

        self.initial_balance = 10000.0
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

        # Extract NumPy arrays for high-performance indexing
        raw_features = self.data[self.feature_columns].values.astype(np.float32)
        scaler = StandardScaler()
        self.features_np = scaler.fit_transform(raw_features).astype(np.float32)
        self.prices_np = self.data["Close"].values.astype(np.float32)
        self.num_steps = len(self.data)

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(24,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.cash = float(self.initial_balance)
        self.shares = 0.0

        self.portfolio_value = float(self.initial_balance)
        self.prev_portfolio_value = float(self.initial_balance)

        self.position = 0
        self.entry_price = 0.0

        return self._next_observation(), {}

    def _next_observation(self):
        if self.current_step >= self.num_steps:
            return np.zeros(24, dtype=np.float32)

        feat = self.features_np[self.current_step]
        cash_ratio = float(self.cash / (self.portfolio_value + 1e-8))
        return_ratio = float((self.portfolio_value - self.initial_balance) / self.initial_balance)
        pos_ratio = float(self.position)

        obs = np.empty(24, dtype=np.float32)
        obs[:21] = feat
        obs[21] = pos_ratio
        obs[22] = cash_ratio
        obs[23] = return_ratio
        return obs

    def step(self, action):
        current_price = float(self.prices_np[self.current_step])
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

        # Update Portfolio value
        self.portfolio_value = (
            self.cash +
            (self.shares * current_price)
        )

        step_return = (
            self.portfolio_value - self.prev_portfolio_value
        ) / (
            self.prev_portfolio_value + 1e-8
        )

        cost_penalty = trade_cost / (self.prev_portfolio_value + 1e-8)
        reward = step_return - cost_penalty

        reward = float(
            np.clip(reward, -1.0, 1.0)
        )

        self.prev_portfolio_value = (
            self.portfolio_value
        )

        self.current_step += 1

        terminated = (
            self.current_step >= self.num_steps - 1
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