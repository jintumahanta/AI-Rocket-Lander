
import gymnasium as gym
from stable_baselines3 import PPO

# Create the original Lunar Lander environment
env = gym.make("LunarLander-v3")

# Create the PPO agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    device="cpu",
)

# Train the agent
print("Starting baseline PPO training...")
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_lunar_baseline")

env.close()

print("Training complete!")
print("Model saved as ppo_lunar_baseline.zip")
