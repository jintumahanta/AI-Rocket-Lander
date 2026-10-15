
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Load the trained baseline model
model = PPO.load("ppo_lunar_baseline")

# Create the original, unmodified environment
env = gym.make("LunarLander-v3")

scores = []

# Evaluate over 10 independent episodes
for episode in range(10):
    obs, info = env.reset(seed=episode)
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    scores.append(total_reward)
    print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

env.close()

print("\n--- Baseline Evaluation ---")
print(f"Average reward: {np.mean(scores):.2f}")
print(f"Minimum reward: {np.min(scores):.2f}")
print(f"Maximum reward: {np.max(scores):.2f}")
print(f"Successful target (average >= 200): {np.mean(scores) >= 200}")
