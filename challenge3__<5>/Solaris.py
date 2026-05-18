"""Atari PPO — Train and Play with Gymnasium + PyTorch
====================================================
This script trains a Proximal Policy Optimization (PPO) agent on the
Arcade Learning Environment (ALE) version of Solaris and lets you watch
it play.

How it works (high level):
  1. The environment renders raw Atari frames preprocessed to 84x84 grayscale.
  2. The agent stacks the last 4 frames to capture motion and transient events.
  3. A shared CNN backbone produces both a policy distribution and a value estimate.
  4. PPO uses on-policy rollouts, Generalized Advantage Estimation (GAE),
     a clipped surrogate objective, and entropy regularization for stable
     policy improvement.

Usage
-----
  # Train with the default PPO config
  python Solaris.py --mode train --model-path models/ppo_solaris

  # Train a named experiment from the JSON config
  python Solaris.py --mode train --experiment default_ppo_solaris --model-path models/ppo_solaris

  # Run all experiments from the sweep config and keep the best model
  python Solaris.py --mode sweep --sweep-file ppo_sweep_configs.json --model-path models/ppo_solaris

  # Run replicates of the best config with random seeds
  python Solaris.py --mode replicate --n-replicates 3 --model-path models/ppo_solaris

  # Watch a trained policy play Solaris
  python Solaris.py --mode play --model-path models/ppo_solaris --episodes 3

  # Inspect the saved PPO checkpoint hyperparameters
  python Solaris.py --mode inspect --model-path models/ppo_solaris

  # Monitor the tuning runs in TensorBoard
  python -m tensorboard.main --logdir logs/ppo_solaris --port 6006

  # Then open http://localhost:6006

  # Change the game by updating ENV_ID below.

Trying a different Atari game
------------------------------
  Change the ENV_ID constant below. Example values:
    "ALE/Pong-v5"
    "ALE/SpaceInvaders-v5"
    "ALE/MsPacman-v5"
    "ALE/Assault-v5"
    "ALE/Asteroids-v5"
  Full list: https://ale.farama.org/environments/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
import ale_py
gym.register_envs(ale_py)

# Atari environment settings matching Solaris.py.
ENV_ID = "ALE/Solaris-v5"
N_STACK = 4
SEEDS_DIR = Path("seeds")
SEEDS_FILE = SEEDS_DIR / "ppo_experiment_seeds.json"

CONFIG_FILE = Path("ppo_sweep_configs.json")
DEFAULT_MODEL_PATH = Path("models/ppo_solaris")
TENSORBOARD_LOG_DIR = Path("logs/ppo_solaris")


def set_global_seed(seed: int) -> None:
    """Set all relevant PRNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def record_seed(experiment_name: str, seed: int, note: Optional[str] = None) -> None:
    """Record a seed for an experiment in JSON."""
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    if SEEDS_FILE.exists():
        with open(SEEDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    entry = data.get(experiment_name, {})
    seeds = entry.get("seeds", [])
    if seed not in seeds:
        seeds.append(seed)
    entry["seeds"] = seeds
    if note:
        entry.setdefault("note", note)
    data[experiment_name] = entry

    with open(SEEDS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_config(experiment_name: str, config: Dict[str, Any], note: Optional[str] = None) -> None:
    """Add or update a configuration entry in the PPO config JSON."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            configs = json.load(f)
    else:
        configs = []

    existing = next((item for item in configs if item.get("name") == experiment_name), None)
    if existing is not None:
        existing.update(config)
        if note:
            existing.setdefault("note", note)
    else:
        record = {"name": experiment_name, **config}
        if note:
            record["note"] = note
        configs.append(record)

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)


def ensure_default_config_file() -> None:
    """Create the base config file for PPO if it does not exist."""
    if CONFIG_FILE.exists():
        return
    config = [
        {
            "name": "default_ppo_solaris",
            "note": "Initial PPO config based on the suggested search space for Solaris.",
            "timesteps": 2000000,
            "learning_rate": 2.5e-4,
            "horizon": 1024,
            "n_epochs": 6,
            "batch_size": 128,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    ]
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def make_env(env_id: str, seed: int = 0, render_mode: Optional[str] = None):
    """Build the ALE environment with the same preprocessing as Solaris.py."""
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True,
        grayscale_newaxis=False,
    )
    env = FrameStackObservation(env, N_STACK)
    env.reset(seed=seed)
    return env


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantage estimates and returns.

    This implements the standard GAE(λ) estimator:
      A_t = δ_t + γλδ_{t+1} + γ^2λ^2δ_{t+2} + ...
    where δ_t = r_t + γV(s_{t+1}) - V(s_t).

    Args:
        rewards: Rewards collected during the rollout.
        values: Value estimates from the critic for each step.
        dones: Done flags from the environment.
        next_value: Value estimate for the final observation after rollout.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        advantages: Advantage estimates for each rollout step.
        returns: Target values for critic regression.
    """
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    values_t = torch.tensor(values, dtype=torch.float32)
    dones_t = torch.tensor(dones, dtype=torch.float32)

    advantages = torch.zeros_like(rewards_t)
    last_advantage = 0.0
    for step in reversed(range(len(rewards_t))):
        mask = 1.0 - dones_t[step]
        delta = rewards_t[step] + gamma * next_value * mask - values_t[step]
        advantages[step] = last_advantage = delta + gamma * gae_lambda * mask * last_advantage
        next_value = values_t[step]

    returns = advantages + values_t
    return advantages, returns


class AtariActorCritic(nn.Module):
    """Shared CNN backbone with separate actor and critic heads."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(N_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_output_size = 64 * 7 * 7
        self.actor = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.cnn(x)
        return self.actor(features), self.critic(features).squeeze(-1)


def save_model(model: AtariActorCritic, model_path: str, hparams: Dict[str, Any]) -> None:
    """Save model weights and hyperparameters in a checkpoint."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hparams": hparams,
    }, f"{model_path}.pth")


def load_model(model_path: str, n_actions: int, device: torch.device) -> AtariActorCritic:
    """Load a saved PPO model checkpoint."""
    checkpoint = torch.load(f"{model_path}.pth", map_location=device)
    model = AtariActorCritic(n_actions).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def inspect_model(model_path: str) -> None:
    """Print saved PPO hyperparameters stored in the checkpoint."""
    checkpoint_path = f"{model_path}.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"\nSaved PPO model: {checkpoint_path}")
    for key, value in checkpoint.get("hparams", {}).items():
        print(f"{key}: {value}")
    print()


def train_ppo(
    model_path: str,
    timesteps: int,
    seed: int,
    hparams: Optional[Dict[str, Any]] = None,
    experiment_name: str = "default_ppo_solaris",
    tensorboard_log: str = str(TENSORBOARD_LOG_DIR),
) -> float:
    """Train PPO on Solaris, save the model, and record the config and seed.

    The training loop implements the key PPO elements:
      - On-policy rollout collection for T environment steps.
      - Advantage estimation using GAE with γ and λ.
      - Clipped surrogate objective with clip_eps.
      - Value-function loss for critic regression.
      - Entropy bonus to encourage exploration.
      - Multiple epochs over the same rollout batch.

    Args:
        model_path: Base path for saving the checkpoint (without .pth).
        timesteps: Total number of environment steps to train.
        seed: Random seed for reproducibility.
        hparams: Optional PPO hyperparameters dictionary.
        experiment_name: Name used to record config and seeds.

    Returns:
        Mean return over the last 10 completed episodes.
    """
    default_hparams = {
        "timesteps": timesteps,
        "learning_rate": 2.5e-4,
        "horizon": 1024,
        "n_epochs": 6,
        "batch_size": 128,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    if hparams is None:
        hparams = default_hparams
    else:
        default_hparams.update(hparams)
        hparams = default_hparams

    set_global_seed(seed)
    device = torch.device("cuda:0")  # Force GPU 0
    env = make_env(ENV_ID, seed=seed, render_mode=None)
    n_actions = env.action_space.n
    model = AtariActorCritic(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"])

    run_name = f"{experiment_name}_seed_{seed}_{int(time.time())}"
    run_dir = Path(tensorboard_log) / run_name
    writer = SummaryWriter(log_dir=str(run_dir))

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    all_returns: List[float] = []
    steps = 0

    while steps < timesteps:
        rollout_length = min(hparams["horizon"], timesteps - steps)
        obs_buf: List[torch.Tensor] = []
        act_buf: List[torch.Tensor] = []
        logp_buf: List[torch.Tensor] = []
        rew_buf: List[float] = []
        done_buf: List[bool] = []
        val_buf: List[float] = []

        # --- rollout collection ---
        # Collect T steps of on-policy experience from the current policy.
        for _ in range(rollout_length):
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value = model(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()

            obs_buf.append(obs_tensor.squeeze(0).cpu())
            act_buf.append(action.cpu())
            logp_buf.append(dist.log_prob(action).cpu())
            val_buf.append(value.squeeze().cpu().item())

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = bool(terminated or truncated)
            rew_buf.append(float(reward))
            done_buf.append(done)
            episode_return += float(reward)
            steps += 1

            if done:
                all_returns.append(episode_return)
                # Learning curve: episode return versus environment steps.
                writer.add_scalar("training/episode_return", episode_return, steps)
                remaining = timesteps - steps
                print(f"Step {steps}/{timesteps} (done: {steps}, remaining: {remaining}) | episode_return: {episode_return:.2f}")
                episode_return = 0.0
                obs, _ = env.reset()

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            _, next_value = model(obs_tensor)

        # --- advantage estimation ---
        advantages, returns = compute_gae(
            rew_buf,
            val_buf,
            done_buf,
            next_value.item(),
            hparams["gamma"],
            hparams["gae_lambda"],
        )

        obs_batch = torch.stack(obs_buf).to(device)
        act_batch = torch.stack(act_buf).to(device)
        logp_batch = torch.stack(logp_buf).to(device)
        adv_batch = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).to(device)
        ret_batch = returns.to(device)

        # --- PPO update ---
        policy_losses: List[float] = []
        value_losses: List[float] = []
        permutation = torch.randperm(rollout_length)
        for _ in range(hparams["n_epochs"]):
            for start in range(0, rollout_length, hparams["batch_size"]):
                batch_idx = permutation[start:start + hparams["batch_size"]]
                logits, values = model(obs_batch[batch_idx])
                dist = Categorical(logits=logits)
                logp_new = dist.log_prob(act_batch[batch_idx])
                entropy = dist.entropy().mean()
                ratio = (logp_new - logp_batch[batch_idx]).exp()

                surr1 = ratio * adv_batch[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - hparams["clip_eps"], 1.0 + hparams["clip_eps"]) * adv_batch[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (values - ret_batch[batch_idx]).pow(2).mean()
                loss = policy_loss + hparams["vf_coef"] * value_loss - hparams["ent_coef"] * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), hparams["max_grad_norm"])
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        mean_policy_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
        mean_value_loss = float(np.mean(value_losses)) if value_losses else 0.0
        mean_return = float(np.mean(all_returns[-10:])) if all_returns else 0.0

        writer.add_scalar("train/policy_loss", mean_policy_loss, steps)
        writer.add_scalar("train/value_loss", mean_value_loss, steps)
        writer.add_scalar("train/mean_return_10", mean_return, steps)

    writer.close()
    env.close()
    print(f"Training complete. Total steps: {steps}")
    save_model(model, model_path, {**hparams, "seed": seed})
    append_config(experiment_name, {**hparams, "timesteps": timesteps}, note="PPO Solaris run")
    record_seed(experiment_name, seed, note="PPO Solaris replicate")

    return float(np.mean(all_returns[-10:])) if all_returns else 0.0


def run_sweep(
    sweep_path: str,
    default_timesteps: int,
    seed: int,
    base_log_dir: str,
    best_model_path: str,
) -> None:
    """Run all experiments defined in a JSON config file and save the best model.

    Each experiment uses the ``timesteps`` value from its JSON entry. If the
    entry omits ``timesteps``, the value from ``default_timesteps`` is used.
    TensorBoard logs for every run are written to
    ``<base_log_dir>/sweep/<experiment_name>/`` so all runs are visible
    together by pointing TensorBoard at ``<base_log_dir>/sweep``.

    After all experiments finish, only the model with the highest mean final
    episode reward is kept at ``best_model_path``. All intermediate models are
    deleted to save disk space.

    Args:
        sweep_path:        Path to the JSON file containing experiment configs.
        default_timesteps: Fallback timestep budget for experiments that do not
                           define ``timesteps`` in their JSON entry.
        seed:              Random seed applied to every experiment.
        base_log_dir:      Root TensorBoard log directory.
        best_model_path:   Where to save the winning model (without .pth).
    """
    with open(sweep_path) as f:
        configs = json.load(f)

    tmp_model_dir = Path("models") / "_ppo_sweep_tmp"
    tmp_model_dir.mkdir(parents=True, exist_ok=True)

    results: List[Tuple[str, float]] = []
    total = len(configs)

    for idx, cfg in enumerate(configs, start=1):
        name = cfg.get("name", f"exp_{idx:02d}")
        note = cfg.get("note", "")
        # Per-experiment timestep budget; falls back to the default value.
        exp_timesteps = cfg.get("timesteps", default_timesteps)

        # Ensure each experiment gets a reproducible but different seed.
        experiment_seed = seed + idx

        print(f"Experiment {idx}/{total}: {name}  ({exp_timesteps:,} steps, seed={experiment_seed})")
        if note:
            print(f"  {note}")
        print(f"{'='*60}")

        # Build the hparams dict expected by train_ppo.
        hparams = {k: v for k, v in cfg.items() if k not in {"name", "note", "timesteps"}}

        model_path = str(tmp_model_dir / name)
        log_dir = f"{base_log_dir}/sweep/{name}"

        score = train_ppo(
            model_path=model_path,
            timesteps=exp_timesteps,
            seed=experiment_seed,
            hparams=hparams,
            experiment_name=name,
            tensorboard_log=log_dir,
        )
        results.append((name, score))
        print(f"  → final mean reward: {score:.2f}")

    # Summary
    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = results[0]

    print(f"\n{'='*60}")
    print("Sweep complete — results ranked by final mean reward:")
    for rank, (name, score) in enumerate(results, start=1):
        marker = "  BEST" if rank == 1 else ""
        print(f"  {rank:2d}. {name:<35s}  {score:7.2f}{marker}")
    print(f"{'='*60}")

    # Save best, clean up 
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(tmp_model_dir / f"{best_name}.pth"), f"{best_model_path}.pth")
    shutil.rmtree(tmp_model_dir)

    print(f"\nBest model ({best_name}, score={best_score:.2f}) saved into {best_model_path}.pth")
    print(f"TensorBoard logs for all runs: {base_log_dir}/sweep/")


def run_replicates(
    sweep_path: str,
    n_replicates: int,
    base_log_dir: str,
    model_base_path: str,
) -> None:
    """Run the best config from ppo_sweep_configs.json multiple times with different random seeds.

    This is useful for validating the stability of the best hyperparameters found by sweep.
    Each replicate uses a random seed and records it in experiment_seeds.json.

    Args:
        sweep_path:      Path to the JSON file containing experiment configs.
        n_replicates:    Number of times to run the best config.
        base_log_dir:    Root TensorBoard log directory.
        model_base_path: Base path for saving replicate models (without .pth).
    """
    try:
        with open(sweep_path) as f:
            configs = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading sweep file {sweep_path}: {e}")
        print("Make sure to run a sweep first to generate configurations.")
        return

    # Ensure configs is a list
    if isinstance(configs, dict):
        configs = [configs]

    if not configs:
        print("No configs found in sweep file.")
        return

    # Assume the last config is the best
    best_config = configs[-1]
    best_name = best_config["name"]

    print(f"Running {n_replicates} replicates of best config: {best_name}")
    print(f"  {best_config.get('note', '')}")
    print(f"{'='*60}")

    for i in range(1, n_replicates + 1):
        replicate_seed = random.randint(0, 100000)

        experiment_name = f"{best_name}_replicate_{i}"

        print(f"Replicate {i}/{n_replicates}: {experiment_name} (seed={replicate_seed})")

        # Filter hparams
        hparams = {k: v for k, v in best_config.items() if k not in ["name", "note"]}

        model_path = f"{model_base_path}_replicate_{i}"
        log_dir = f"{base_log_dir}/replicates/{experiment_name}"

        timesteps = hparams.get("timesteps", 2000000)

        score = train_ppo(
            model_path=model_path,
            timesteps=timesteps,
            seed=replicate_seed,
            hparams=hparams,
            experiment_name=experiment_name,
            tensorboard_log=log_dir,
        )
        print(f"  → final mean reward: {score:.2f}")

    print(f"\n{'='*60}")
    print(f"Replication complete. Models saved as {model_base_path}_replicate_1.pth etc.")
    print(f"TensorBoard logs: {base_log_dir}/replicates/")


def play_agent(model_path: str, episodes: int = 3, seed: int = 42) -> None:
    """Play a saved PPO agent in a rendered Atari window."""
    checkpoint_path = f"{model_path}.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    env = make_env(ENV_ID, seed=seed, render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, env.action_space.n, device)
    model.eval()

    completed = 0
    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0

    while completed < episodes:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(obs_tensor)
            action = Categorical(logits=logits).sample().item()

        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += float(reward)
        if terminated or truncated:
            completed += 1
            print(f"Episode {completed} reward={episode_reward:.1f}")
            episode_reward = 0.0
            obs, _ = env.reset()

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, inspect, or play PPO on Atari Solaris.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "play", "inspect", "sweep", "replicate"], required=True)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Model path (without .pth).")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training environment steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--experiment", default="default_ppo_solaris", help="Experiment name for config and seed records.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play in play mode.")
    parser.add_argument("--tensorboard-log", default=str(TENSORBOARD_LOG_DIR), help="Directory for TensorBoard logs.")
    parser.add_argument("--sweep-file", default="ppo_sweep_configs.json", help="Path to JSON file with experiment configs (used by --mode sweep).")
    parser.add_argument("--n-replicates", type=int, default=3, help="Number of replicates for replicate mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_default_config_file()

    hparams = None
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            configs = json.load(f)
        # Ensure configs is a list
        if isinstance(configs, dict):
            configs = [configs]
        experiment = next((cfg for cfg in configs if cfg.get("name") == args.experiment), None)
        if experiment is not None:
            hparams = {k: v for k, v in experiment.items() if k not in {"name", "note"}}

    if args.mode == "train":
        best_return = train_ppo(
            model_path=args.model_path,
            timesteps=args.timesteps,
            seed=args.seed,
            hparams=hparams,
            experiment_name=args.experiment,
            tensorboard_log=args.tensorboard_log,
        )
        print(f"\nTraining complete. Mean return (last 10 episodes): {best_return:.2f}")
        print(f"Model saved to {args.model_path}.pth")
        print(f"Configs stored in {CONFIG_FILE}")
        print(f"Seeds stored in {SEEDS_FILE}")

    elif args.mode == "play":
        play_agent(model_path=args.model_path, episodes=args.episodes, seed=args.seed)

    elif args.mode == "inspect":
        inspect_model(model_path=args.model_path)

    elif args.mode == "sweep":
        run_sweep(
            sweep_path=args.sweep_file,
            default_timesteps=args.timesteps or 2000000,
            seed=args.seed,
            base_log_dir=args.tensorboard_log,
            best_model_path=args.model_path,
        )

    elif args.mode == "replicate":
        run_replicates(
            sweep_path=args.sweep_file,
            n_replicates=args.n_replicates,
            base_log_dir=args.tensorboard_log,
            model_base_path=args.model_path,
        )


if __name__ == "__main__":
    main()