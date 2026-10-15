
import csv
import matplotlib.pyplot as plt

# Recorded original-environment evaluation results
results = {
    "Baseline PPO": {
        "rewards": [-167.20, -41.27, -80.59, -83.49, -28.10,
                    -23.74, 14.51, -39.62, -16.67, 16.61],
        "steps": [None] * 10,
        "training_steps": 100_000,
    },
    "Reward-Hacking PPO": {
        "rewards": [-82.36, -71.65, -131.42, -164.91, -110.37,
                    -52.40, -98.12, -99.74, -134.89, -118.58],
        "steps": [250, 1000, 1000, 1000, 1000,
                  1000, 497, 1000, 590, 704],
        "training_steps": 100_000,
    },
    "Progress-Shaped PPO": {
        "rewards": [254.26, 192.88, 226.65, 216.36, 244.62,
                    250.52, 250.31, 209.33, 244.02, -22.87],
        "steps": [300, 265, 472, 260, 293,
                  286, 252, 279, 221, 171],
        "training_steps": 200_000,
    },
}

# Export episode-level results to CSV
with open("evaluation_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "agent", "episode", "original_reward",
        "episode_steps", "training_timesteps"
    ])

    for agent, data in results.items():
        for i, reward in enumerate(data["rewards"]):
            writer.writerow([
                agent,
                i + 1,
                reward,
                data["steps"][i],
                data["training_steps"],
            ])

# Calculate and print the actual means
agents = list(results.keys())
means = [
    sum(results[agent]["rewards"]) / len(results[agent]["rewards"])
    for agent in agents
]

print("Average original-environment rewards:")
for agent, mean in zip(agents, means):
    print(f"{agent}: {mean:.2f}")

# Plot average rewards
plt.figure(figsize=(10, 6))
bars = plt.bar(agents, means)
plt.axhline(200, linestyle="--", label="Target reward = 200")
plt.axhline(0, linewidth=0.8)

plt.title("Lunar Lander: Reward Design Comparison")
plt.ylabel("Average Original Episode Reward (10 episodes)")
plt.xlabel("Agent")
plt.xticks(rotation=10)
plt.legend()

for bar, mean in zip(bars, means):
    offset = 5 if mean >= 0 else -12
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        mean + offset,
        f"{mean:.2f}",
        ha="center",
        va="bottom" if mean >= 0 else "top",
    )

plt.tight_layout()
plt.savefig("average_reward_comparison.png", dpi=200)
plt.close()

print("\nCreated evaluation_results.csv")
print("Created average_reward_comparison.png")
