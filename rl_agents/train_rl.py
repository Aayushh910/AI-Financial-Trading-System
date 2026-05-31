from stable_baselines3 import PPO
from rl_agents.trading_env import TradingEnv

def train_agent(data):

    env = TradingEnv(data)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64
    )

    model.learn(total_timesteps=20000)

    # Save trained model
    model.save("ppo_trading_agent")

    return model