
import gymnasium as gym
from stable_baselines3 import PPO

# Load your improved, trained AI model
model = PPO.load("ppo_lunar_shaped")

# Open the Lunar Lander simulation window
env = gym.make("LunarLander-v3", render_mode="human")

obs, info = env.reset(seed=42)
total_reward = 0.0
steps = 0
terminated = False
truncated = False

print("Watching the improved AI lander...")
print("Watch the simulation window!")

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    steps += 1

print("\n--- Landing Results ---")
print(f"Steps: {steps}")
print(f"Original reward: {total_reward:.2f}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")

env.close()
input("Press Enter to exit...")
