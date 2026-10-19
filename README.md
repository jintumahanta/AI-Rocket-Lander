
# Rocket Landing with Reinforcement Learning: Reward Hacking and Reward Shaping

A reinforcement learning project that investigates how reward design influences an AI agent's behavior when learning to land a simulated lunar lander. Using **Proximal Policy Optimization (PPO)** and Gymnasium's `LunarLander-v3` environment, this project compares a baseline agent, an agent trained with a deliberately flawed reward function, and an agent trained using progress-based reward shaping.

The goal is to demonstrate **reward hacking**—when an agent exploits its reward function without achieving the intended objective—and evaluate whether a better-designed reward function can improve landing performance.

## Project Overview

Three PPO agents are trained using different reward strategies:

1. **Baseline PPO:** Uses the environment's original reward function.
2. **Reward-Hacking PPO:** Receives an additional `+1` reward at every simulation step, illustrating how a poorly designed reward function can encourage unnecessary survival and delayed task completion.
3. **Progress-Shaped PPO:** Uses a progress-based shaping term and a small per-step penalty to encourage movement toward the landing area while discouraging unnecessary delays.

All three agents are evaluated using the original `LunarLander-v3` reward function. This makes the comparison more meaningful because artificial training bonuses are excluded from the evaluation scores.

## Objectives

- Implement a reinforcement learning agent using PPO.
- Investigate how reward design affects learned behavior.
- Demonstrate reward hacking using an intentionally flawed reward function.
- Develop and evaluate a progress-based reward-shaping strategy.
- Compare agent performance using episode rewards and episode lengths.
- Analyze the limitations of reward-based optimization in reinforcement learning.

## Technologies Used

- **Language:** Python 3.12
- **Environment:** Gymnasium (`LunarLander-v3`)
- **Reinforcement Learning:** Proximal Policy Optimization (PPO)
- **RL Framework:** Stable-Baselines3
- **Deep Learning:** PyTorch
- **Numerical Computing:** NumPy
- **Physics Simulation:** Box2D

## Experimental Results

Each agent was evaluated over **10 episodes** using the original environment reward function.

| Agent | Average Reward | Average Episode Length |
|---|---:|---:|
| Baseline PPO | -44.96 | Not recorded |
| Reward-Hacking PPO | -106.44 | 804.1 steps |
| Progress-Shaped PPO | 206.61 | 279.9 steps |

### Key Results

- **Progress-Shaped PPO achieved an average reward of 206.61**, exceeding the commonly used 200-point performance benchmark for LunarLander.
- **Reward-Hacking PPO scored -106.44**, performing worse than the baseline despite receiving an additional reward at every simulation step during training.
- The reward-hacking agent averaged 804.1 steps per episode, while the progress-shaped agent averaged 279.9 steps—a reduction of approximately **65.2%** in average episode length.

These results illustrate that maximizing a poorly designed reward signal does not necessarily improve performance on the intended task.

### Experimental Limitation

The agents were trained using different timestep budgets:

- Baseline PPO: 100,000 timesteps
- Reward-Hacking PPO: 100,000 timesteps
- Progress-Shaped PPO: 200,000 timesteps

Because the progress-shaped agent received twice the training budget of the other agents, the results do not isolate the effect of reward design from the effect of additional training. Further experiments using equal training budgets, multiple random seeds, and more evaluation episodes are needed to establish a fairer comparison.

## Project Structure

```text
AI-Rocket-Lander/
├── train_baseline.py
├── evaluate_baseline.py
├── train_reward_hacking.py
├── evaluate_reward_hacking.py
├── watch_reward_hacking.py
├── train_shaped.py
├── evaluate_shaped.py
├── watch_shaped.py
├── requirements.txt
├── .gitignore
└── README.md
```

The training scripts generate the trained model files. Generated models and evaluation outputs may not be included in the repository.

## Installation

### Prerequisites

- Python 3.12
- Git
- A working Python environment with the required dependencies

### 1. Clone the repository

```bash
git clone https://github.com/jintumahanta/AI-Rocket-Lander.git
cd AI-Rocket-Lander
```

### 2. Create a virtual environment

On Windows:

```bash
py -3.12 -m venv .venv
```

### 3. Activate the virtual environment

On Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If installation fails, check the Python version and the compatibility of the installed Gymnasium, Box2D, and Stable-Baselines3 dependencies.

## Running the Project

Run the scripts from the repository's root directory.

### 1. Train and evaluate the baseline agent

Train the baseline PPO agent:

```bash
python train_baseline.py
```

Evaluate the trained baseline agent:

```bash
python evaluate_baseline.py
```

### 2. Train and evaluate the reward-hacking agent

```bash
python train_reward_hacking.py
python evaluate_reward_hacking.py
```

### 3. Train and evaluate the progress-shaped agent

```bash
python train_shaped.py
python evaluate_shaped.py
```

### 4. Visualize agent behavior

Watch the reward-hacking agent:

```bash
python watch_reward_hacking.py
```

Watch the progress-shaped agent:

```bash
python watch_shaped.py
```

**Note:** Run each evaluation or visualization script after the corresponding trained model has been generated and saved in the location expected by that script.

## Key Findings

- **Reward hacking:** A constant survival bonus can encourage an agent to prolong an episode rather than prioritize a successful landing.
- **Reward alignment:** An agent can optimize the reward signal it receives without necessarily optimizing the actual objective.
- **Reward shaping:** Progress-based rewards can provide more useful learning feedback when designed to align with the intended task.
- **Evaluation matters:** Agents should be evaluated using the original task reward rather than artificial bonuses introduced during training.
- **Experimental fairness:** Differences in training duration can influence results, so controlled experiments are essential for reliable comparisons.

## Future Improvements

- Train all agents using equal timestep budgets.
- Repeat experiments across multiple random seeds.
- Increase the number of evaluation episodes to improve statistical reliability.
- Record episode rewards and lengths in CSV files.
- Generate reward curves and comparative performance plots.
- Experiment with alternative reward functions and PPO hyperparameters.
- Measure successful landings, crashes, and time-limit truncations separately.
- Compare the agents using mean reward, landing success rate, and episode length.

## Conclusion

This project demonstrates how reward design can influence the behavior of reinforcement learning agents in a simulated lunar-landing task.

The reward-hacking experiment illustrates that introducing a seemingly beneficial reward can produce undesirable behavior when the reward does not properly reflect the intended objective. In contrast, the progress-shaped agent achieved an average original-environment reward of **206.61 across 10 evaluation episodes**, exceeding the 200-point benchmark.

However, because the progress-shaped agent was trained for twice as many timesteps as the other agents, further controlled experiments are necessary to determine the extent to which reward shaping itself contributed to the improvement.

The project highlights an important principle in reinforcement learning: **an agent's reward function should be designed to encourage the behavior we actually want, not merely behavior that maximizes the numerical reward.**



