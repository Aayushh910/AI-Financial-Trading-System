from stable_baselines3 import PPO
from rl_agents.trading_env import TradingEnv

def evaluate_agent(data):

    env = TradingEnv(data)

    model = PPO.load("ppo_trading_agent")

    obs, _ = env.reset()

    total_reward = 0

    while True:

        action, _ = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    print("\n===== RL Evaluation =====")
    print("Total Reward:", total_reward)
    print("Final Balance:", round(info["balance"], 2))
    print("Final Position:", info["position"])