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
        verbose=1,

        learning_rate=5e-5,
        n_steps=4096,
        batch_size=256,

        gamma=0.995,
        gae_lambda=0.98,

        ent_coef=0.005,
        clip_range=0.2,

        n_epochs=20,

        policy_kwargs=dict(
            net_arch=[256, 256, 128]
        )
    )

    model.learn(
        total_timesteps=100000,
        progress_bar=True
    )

    model.save(
        "ppo_trading_agent"
    )

    return model