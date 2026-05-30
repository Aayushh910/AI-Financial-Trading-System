from stable_baselines3 import PPO
from rl_agents.trading_env import TradingEnv

def train_agent(data):

    env = TradingEnv(data)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1
    )

    model.learn(total_timesteps=10000)

    return model