
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


class ProgressRewardWrapper(gym.Wrapper):
    """Reward progress toward the landing pad and discourage wasted time."""

    def __init__(self, env):
        super().__init__(env)
        self.gamma = 0.99
        self.previous_potential = 0.0

    def potential(self, observation):
        # The landing pad is at (x, y) = (0, 0).
        x, y = observation[0], observation[1]
        distance = np.sqrt(x**2 + y**2)
        return -float(distance)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        current_potential = self.potential(observation)

        # Potential-based shaping: reward progress, not mere survival.
        progress_reward = (
            self.gamma * current_potential - self.previous_potential
        )

        # Small cost for every frame to discourage needless delay.
        time_penalty = 0.01

        shaped_reward = reward + progress_reward - time_penalty
        self.previous_potential = current_potential

        return (
            observation,
            shaped_reward,
            terminated,
            truncated,
            info,
        )


env = ProgressRewardWrapper(gym.make("LunarLander-v3"))

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

print("Starting progress-shaped PPO training...")
model.learn(total_timesteps=200_000)

model.save("ppo_lunar_shaped")
env.close()

print("Training complete!")
print("Model saved as ppo_lunar_shaped.zip")
