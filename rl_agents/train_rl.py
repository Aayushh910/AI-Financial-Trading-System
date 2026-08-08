import random
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl_agents.trading_env import TradingEnv


def train_agent(data, seed=42):
    # Set seeds for reproducibility across runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def make_env():
        env_instance = TradingEnv(data)
        env_instance.reset(seed=seed)
        env_instance.action_space.seed(seed)
        env_instance.observation_space.seed(seed)
        return Monitor(env_instance)

    env = DummyVecEnv([make_env])
    env.seed(seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.2,
        max_grad_norm=0.5,
        n_epochs=10,
        seed=seed,
        policy_kwargs=dict(
            net_arch=[128, 128],
            ortho_init=True
        )
    )

    model.learn(
        total_timesteps=40000,
        progress_bar=False
    )

    model.save(
        "ppo_trading_agent"
    )

    return model