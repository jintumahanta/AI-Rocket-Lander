
import gymnasium as gym
from stable_baselines3 import PPO

# Open a visible Lunar Lander window
env = gym.make("LunarLander-v3", render_mode="human")

model = PPO.load("ppo_lunar_reward_hacking")

obs, info = env.reset(seed=42)
total_reward = 0.0
steps = 0
terminated = False
truncated = False

print("Watching the reward-hacking agent...")
print("Close the simulation window or wait for the episode to finish.")

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    steps += 1

print(f"Episode steps: {steps}")
print(f"Original environment reward: {total_reward:.2f}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")

env.close()
