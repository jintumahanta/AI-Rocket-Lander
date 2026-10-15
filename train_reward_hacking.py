
import gymnasium as gym
from stable_baselines3 import PPO


class HoverRewardWrapper(gym.Wrapper):
    """Add +1 reward for every frame the agent remains active."""

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        # Intentional reward loophole for the experiment
        modified_reward = reward + 1.0

        return (
            observation,
            modified_reward,
            terminated,
            truncated,
            info,
        )


# Create the environment with the modified reward
env = HoverRewardWrapper(gym.make("LunarLander-v3"))

# Train a new agent from scratch
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

print("Starting reward-hacking experiment...")
model.learn(total_timesteps=100_000)

# Save this model separately from the baseline
model.save("ppo_lunar_reward_hacking")

env.close()

print("Reward-hacking training complete!")
print("Model saved as ppo_lunar_reward_hacking.zip")
