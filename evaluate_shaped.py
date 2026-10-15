
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

model = PPO.load("ppo_lunar_shaped")
env = gym.make("LunarLander-v3")

scores = []
episode_lengths = []

for episode in range(10):
    obs, info = env.reset(seed=episode)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Evaluate using original rewards only
        total_reward += reward
        steps += 1

    scores.append(total_reward)
    episode_lengths.append(steps)

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}, Steps = {steps}"
    )

env.close()

print("\n--- Improved Agent Evaluation ---")
print(f"Average reward: {np.mean(scores):.2f}")
print(f"Minimum reward: {np.min(scores):.2f}")
print(f"Maximum reward: {np.max(scores):.2f}")
print(f"Average episode length: {np.mean(episode_lengths):.1f}")
print(f"Target average >= 200: {np.mean(scores) >= 200}")
