from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl_agents.trading_env import TradingEnv


def train_agent(data):

    env = DummyVecEnv([
        lambda: Monitor(TradingEnv(data))
    ])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,

        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,

        gamma=0.99,
        gae_lambda=0.95,

        ent_coef=0.01,
        clip_range=0.2,

        n_epochs=10,

        policy_kwargs=dict(
            net_arch=[128, 128]
        )
    )

    model.learn(
        total_timesteps=25000,
        progress_bar=False
    )

    model.save(
        "ppo_trading_agent"
    )

    return model