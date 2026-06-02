import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):

    def __init__(self, data):
        super().__init__()

        self.data = data.reset_index(drop=True)

        # Portfolio Settings
        self.initial_balance = 10000
        self.balance = self.initial_balance

        self.current_step = 0
        self.position = 0

        # Actions:
        # 0 = Hold
        # 1 = Buy
        # 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )

    # Reset Environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0

        return self._next_observation(), {}

    # Observation Function
    def _next_observation(self):

        obs = np.array([

            # Basic Features
            self.data['Lag_1'].iloc[self.current_step],
            self.data['Lag_2'].iloc[self.current_step],
            self.data['Momentum'].iloc[self.current_step],
            self.data['Rolling_STD'].iloc[self.current_step],

            # RSI
            self.data['RSI'].iloc[self.current_step],

            # MACD
            self.data['MACD'].iloc[self.current_step],
            self.data['MACD_SIGNAL'].iloc[self.current_step],

            # Bollinger Bands
            self.data['BB_HIGH'].iloc[self.current_step],
            self.data['BB_LOW'].iloc[self.current_step],

            # ATR
            self.data['ATR'].iloc[self.current_step],

            # Volume Feature
            self.data['Volume_Change'].iloc[self.current_step]

        ], dtype=np.float32)

        return obs

    # Step Function
    def step(self, action):

        current_return = self.data['Target'].iloc[self.current_step]

        reward = 0

        # Transaction Cost
        transaction_cost = 0.001

        # BUY
        if action == 1:
            self.position = 1
            reward = current_return - transaction_cost

        # SELL
        elif action == 2:
            self.position = -1
            reward = -current_return - transaction_cost

        # HOLD
        else:
            reward = -0.0001

        # Update Portfolio Balance
        self.balance *= (1 + reward)

        # Move to next timestep
        self.current_step += 1

        # Episode End
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        # Next Observation
        obs = self._next_observation()

        # Extra Info
        info = {
            "balance": self.balance,
            "position": self.position
        }

        return obs, reward, terminated, truncated, info