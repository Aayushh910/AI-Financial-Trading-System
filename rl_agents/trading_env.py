import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):

    def __init__(self, data):
        super(TradingEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.current_step = 0

        # Actions:
        # 0 = Hold
        # 1 = Buy
        # 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        return self._next_observation(), {}

    def _next_observation(self):

        obs = np.array([
            self.data['Lag_1'].iloc[self.current_step],
            self.data['Lag_2'].iloc[self.current_step],
            self.data['Momentum'].iloc[self.current_step],
            self.data['Rolling_STD'].iloc[self.current_step]
        ], dtype=np.float32)

        return obs

    def step(self, action):

        reward = self.data['Target'].iloc[self.current_step]

        # Basic reward logic
        if action == 1:  # Buy
            reward = reward

        elif action == 2:  # Sell
            reward = -reward

        else:  # Hold
            reward = 0

        self.current_step += 1

        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        obs = self._next_observation()

        return obs, reward, terminated, truncated, {}