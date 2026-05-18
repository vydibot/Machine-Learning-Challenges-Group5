# Challenge 3: PPO vs DQN on ALE/Solaris-v5 - Copilot Agent Guidelines

## Agent Objective
You are an expert reinforcement learning coding agent. Your goal is to complete Challenge 3 by implementing a PPO agent for the Atari `ALE/Solaris-v5` environment, comparing it against a previously trained DQN agent, and generating an IEEE-formatted scientific report.

## 1. Environment & Preprocessing Protocol
You must ensure the `ALE/Solaris-v5` environment perfectly matches the Challenge 1 baseline for a fair comparison. 
* Implement the following wrappers: Grayscale, Resize to 84x84, Frame-stack 4, and Frame-skip 4.
* Do not alter the reward structure. Keep preprocessing constant so performance differences are strictly algorithmic.

## 2. PPO Implementation Requirements
Update `challenge3__<5>/Solaris.py` to include a fully functioning Proximal Policy Optimization (PPO) agent.
* Include on-policy rollout collection for T environment steps.
* Implement Generalised Advantage Estimation (GAE).
* Use a clipped surrogate objective and an entropy bonus to encourage exploration.
* Ensure the architecture uses a shared convolutional backbone with separate actor and critic heads.

## 3. Training & Hyperparameter Sweep (Group 5 Focus)
Run experiments with a strict budget of 5,000,000 environment steps. Focus on the parameters critical for Solaris (multi-stage dynamics):
* Horizon: 2048
* Gamma: 0.995
* GAE Lambda: 0.97
* Clip Epsilon: 0.2
* Entropy Coefficient: 0.01
* Run the best configuration across at least 3 independent random seeds.
* Log all metrics. Save the best models to `models/ppo_solaris`.

## 4. Evaluation & Metrics Comparison
You must compare the PPO runs against the existing DQN baseline using the provided CSV time-series data.
* Load the DQN baseline data from: `time_series/dqn`
* Load the new PPO data from: `time_series/ppo`
* Generate comparative plots for **Learning Curve** (episode return vs. env steps).
* Calculate **Sample Efficiency** (steps needed to reach a specific score threshold above baseline).
* Calculate **Final Performance** (mean and standard deviation over the 3 seeds at the end of training).
* Calculate **Training Stability** (area under the learning curve normalized by total steps).
* Analyze whether DQN's experience replay or PPO's longer trajectory windows work better for the multi-stage nature of Solaris.

## 5. Required Deliverables Generation

**Task A: CHECKLIST.md**
Create a `challenge3__<5>/CHECKLIST.md` file containing:
* The exact command to reproduce the best PPO run.
* The seeds used for the repeated experiments.
* Pointers to the log and figure folders.
* A 200-word comparative summary of DQN vs PPO on Solaris.

**Task B: IEEE Report (.tex)**
Generate an 8-page scientific report in LaTeX format named `challenge3_group5_paper.tex`. You must strictly follow this structure:
* **Abstract:** Three sentences covering context, problem/solution, and key results.
* **IEEEkeywords:** Relevant keywords for RL and Atari.
* **Introduction (1 page):** Context, problem description, challenges of Solaris, and prior work (referencing DQN).
* **Methods and Materials (1-2 pages):** Design choices, PPO architecture, hyperparameter decisions, and justification for the approach. Provide enough detail for reproducibility.
* **Results & Discussion:** Present the comparative evidence using the metrics calculated in Step 4. Include references to the plotted learning curves and tables. Explain the differences linking algorithmic properties (DQN off-policy/replay buffer vs. PPO on-policy/clipped ratio).
* **Conclusions (1-2 paragraphs):** Summarize the work, state why the approach succeeded or failed, and offer suggestions for future work.
* **References:** Standard IEEE bibliography referencing the original PPO, GAE, and DQN papers.