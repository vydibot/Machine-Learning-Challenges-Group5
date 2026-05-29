"""GAIL for Atari Solaris — Learning from Demonstration with Adversarial Training
================================================================================
Generative Adversarial Imitation Learning (GAIL) agent for Solaris.
Combines Behavioral Cloning baseline, discriminator adversarial training, and PPO.

This implementation follows the Challenge 4 specification:
  1. Demonstration collection from trained PPO checkpoints (Challenge 3)
  2. Behavioral Cloning (BC) baseline — supervised learning lower bound
  3. GAIL training loop — alternating discriminator (binary cross-entropy) and 
     PPO updates using adversarial reward signal instead of environment reward
  4. Demonstration quality study — best PPO vs mid-training PPO checkpoints
  5. TensorBoard logging for all training metrics and discriminator dynamics

Usage examples:
  # Step 1: Collect demonstrations from best PPO checkpoint
  python SolarisGAIL.py --mode collect-demos --checkpoint-path models/ppo_solaris.pth \\
    --n-steps 50000 --output demonstrations/demos_best_ppo.npz

  # Step 2: Train Behavioral Cloning baseline
  python SolarisGAIL.py --mode train-bc --demos demonstrations/demos_best_ppo.npz --model-path models/bc_solaris

  # Step 3: Train GAIL (adversarial imitation learning) - automatically generates random seed
  python SolarisGAIL.py --mode train-gail --demos demonstrations/demos_best_ppo.npz \\
    --experiment gail_from_best_ppo --model-path models/gail_solaris

  # Step 4: Play trained GAIL agent
  python SolarisGAIL.py --mode play --model-path models/gail_solaris --episodes 3

  # Step 5: Inspect GAIL checkpoint hyperparameters
  python SolarisGAIL.py --mode inspect --model-path models/gail_solaris

  # Monitor training live in TensorBoard:
  python -m tensorboard.main --logdir logs/gail_solaris --port 6006


"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import ale_py

gym.register_envs(ale_py)

# Atari environment settings matching Solaris.py and Challenge 3
ENV_ID = "ALE/Solaris-v5"
N_STACK = 4
SEEDS_DIR = Path("seeds")
SEEDS_FILE = SEEDS_DIR / "gail_experiment_seeds.json"

CONFIG_FILE = Path("gail_sweep_configs.json")
DEFAULT_MODEL_PATH = Path("models/gail_solaris")
TENSORBOARD_LOG_DIR = Path("logs/gail_solaris")
DEMOS_DIR = Path("demonstrations")


def set_global_seed(seed: Optional[int]) -> int:
    """Set all relevant PRNGs for reproducibility and return the resolved seed."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


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
    """Add or update a configuration entry in the GAIL config JSON."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            configs = json.load(f)
    else:
        configs = []

    if not isinstance(configs, list):
        configs = [configs]

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
    """Create the base config file for GAIL if it does not exist."""
    if CONFIG_FILE.exists():
        return
    # Default GAIL config based on Challenge 4 specification (2M timesteps)
    config = [
        {
            "name": "gail_baseline",
            "note": "Baseline GAIL configuration for Solaris (Group 5)",
            "total_timesteps": 2000000,
            "horizon": 2048,
            "n_ppo_epochs": 4,
            "batch_size": 128,
            "lr_policy": 2.5e-4,
            "lr_disc": 3e-4,
            "disc_updates_per_rollout": 5,
            "gamma": 0.995,
            "gae_lambda": 0.97,
            "clip_eps": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    ]
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def make_env(env_id: str, seed: int = 0, render_mode: Optional[str] = None):
    """Build the ALE environment with identical preprocessing to Challenge 3."""
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


class AtariActorCritic(nn.Module):
    """Shared CNN backbone with separate actor and critic heads (PPO-style)."""

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
        """Returns (logits, value)."""
        features = self.cnn(x)
        return self.actor(features), self.critic(features).squeeze(-1)


class GAILDiscriminator(nn.Module):
    """Discriminator network for GAIL.
    
    Takes stacked-frame observations (and optionally one-hot actions) and 
    outputs P(expert | s, a) ∈ (0, 1).
    """

    def __init__(self, n_actions: int, use_action: bool = False):
        super().__init__()
        self.use_action = use_action
        
        # Shared CNN backbone (same as policy)
        self.cnn = nn.Sequential(
            nn.Conv2d(N_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        cnn_out = 64 * 7 * 7  # 3136
        fc_in = cnn_out + n_actions if use_action else cnn_out

        self.fc = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        self.n_actions = n_actions

    def forward(self, obs: torch.Tensor, actions_onehot: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass. Returns probability of being expert trajectory."""
        feats = self.cnn(obs)
        if self.use_action and actions_onehot is not None:
            feats = torch.cat([feats, actions_onehot], dim=-1)
        return self.fc(feats).squeeze(-1)


def save_model(model: AtariActorCritic, model_path: str, hparams: Dict[str, Any]) -> None:
    """Save model weights and hyperparameters."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hparams": hparams,
    }, f"{model_path}.pth")


def load_model(model_path: str, n_actions: int, device: torch.device) -> AtariActorCritic:
    """Load a saved GAIL policy checkpoint."""
    checkpoint = torch.load(f"{model_path}.pth", map_location=device)
    model = AtariActorCritic(n_actions).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def inspect_model(model_path: str) -> None:
    """Print saved GAIL hyperparameters from checkpoint."""
    checkpoint_path = f"{model_path}.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"\nSaved GAIL model: {checkpoint_path}")
    for key, value in checkpoint.get("hparams", {}).items():
        print(f"{key}: {value}")
    print()


def load_expert_dataset(path: str, max_samples: Optional[int] = None):
    """Load expert dataset from .npz file with 'obs' and 'acts' arrays.
    
    Handles both (N, C, H, W) and (N, H, W, C) shapes.
    Converts to float32 in [0, 1] range.
    """
    data = np.load(path)
    obs = data["obs"]
    acts = data["acts"]
    
    # Convert HWC -> CHW if needed
    if obs.ndim == 4 and obs.shape[-1] == N_STACK:
        obs = np.transpose(obs, (0, 3, 1, 2))
    
    # Normalize to [0, 1]
    if obs.dtype == np.uint8:
        obs = obs.astype(np.float32) / 255.0
    else:
        obs = obs.astype(np.float32)
    
    acts = acts.astype(np.int64)
    
    if max_samples is not None:
        idx = np.random.choice(len(obs), size=min(max_samples, len(obs)), replace=False)
        obs = obs[idx]
        acts = acts[idx]
    
    return obs, acts


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).
    
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    Returns (advantages, returns).
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


def collect_demonstrations(
    env_id: str,
    checkpoint_path: str,
    n_steps: int = 50000,
    seed: int = 42,
    output_path: str = "demonstrations/demos.npz",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, np.ndarray]:
    """Collect demonstrations by rolling out a saved policy checkpoint.
    
    Loads a trained PPO policy and records (observation, action) tuples.
    Saves to .npz file with keys 'obs' and 'acts'.
    
    Args:
        env_id: Gymnasium environment ID
        checkpoint_path: Path to saved model checkpoint
        n_steps: Number of environment steps to collect
        seed: Random seed
        output_path: Where to save demonstrations
        device: torch device
        
    Returns:
        Dictionary with 'obs' and 'acts' numpy arrays
    """
    seed = set_global_seed(seed)
    
    env = make_env(env_id, seed=seed, render_mode=None)
    n_actions = env.action_space.n
    device_obj = torch.device(device)
    
    model = AtariActorCritic(n_actions).to(device_obj)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    
    # Handle both our format and standard PyTorch format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    obs_buf, act_buf = [], []
    obs, _ = env.reset(seed=seed)
    
    print(f"Collecting {n_steps} demonstration steps from {checkpoint_path}...")
    for step in range(n_steps):
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device_obj)
        with torch.no_grad():
            logits, _ = model(obs_t)
            # Greedy (deterministic) action selection
            action = logits.argmax(dim=-1).item()

        obs_buf.append(obs)
        act_buf.append(action)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        
        if (step + 1) % 10000 == 0:
            print(f"  {step + 1}/{n_steps} steps collected")

    env.close()
    
    demos = {
        "obs": np.array(obs_buf, dtype=np.float32),
        "acts": np.array(act_buf, dtype=np.int64),
    }
    
    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **demos)
    
    print(f"✓ Saved {n_steps} demonstration steps to {output_path}")
    print(f"  obs shape: {demos['obs'].shape}, acts shape: {demos['acts'].shape}")
    print(f"  Mean action distribution: {np.bincount(act_buf)}")
    
    return demos


def train_bc(
    env_id: str,
    demos_path: str,
    n_epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-4,
    model_path: str = "models/bc_solaris",
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> AtariActorCritic:
    """Train Behavioral Cloning baseline.
    
    Minimizes cross-entropy between demonstrations and policy outputs.
    Establishes supervised learning lower bound.
    
    Args:
        env_id: Gymnasium environment ID
        demos_path: Path to .npz demonstration file
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        model_path: Where to save model
        seed: Random seed
        device: torch device
        
    Returns:
        Trained policy model
    """
    set_global_seed(seed)
    device_obj = torch.device(device)
    
    print(f"\n{'='*60}")
    print("Training Behavioral Cloning (BC) Baseline")
    print(f"{'='*60}")
    
    # Load demonstrations
    data = np.load(demos_path)
    obs_t = torch.from_numpy(data["obs"]).to(device_obj)
    act_t = torch.from_numpy(data["acts"]).to(device_obj)
    
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create environment to get n_actions
    env = make_env(env_id)
    n_actions = env.action_space.n
    env.close()
    
    model = AtariActorCritic(n_actions).to(device_obj)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        for obs_b, act_b in loader:
            logits, _ = model(obs_b)
            loss = criterion(logits, act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"BC epoch {epoch + 1}/{n_epochs} | loss = {avg_loss:.4f}")
    
    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{model_path}.pth")
    print(f"✓ Saved BC model to {model_path}.pth\n")
    
    return model


def train_gail(
    env_id: str,
    demos_path: str,
    total_timesteps: int = 2000000,
    horizon: int = 2048,
    n_ppo_epochs: int = 4,
    batch_size: int = 128,
    lr_policy: float = 2.5e-4,
    lr_disc: float = 3e-4,
    disc_updates_per_rollout: int = 5,
    gamma: float = 0.995,
    gae_lambda: float = 0.97,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: Optional[int] = None,
    experiment_name: str = "gail_baseline",
    tensorboard_log: str = str(TENSORBOARD_LOG_DIR),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[AtariActorCritic, GAILDiscriminator, List[float]]:
    """Train GAIL: adversarial imitation learning with PPO.
    
    Alternates between:
      1. Discriminator update: binary cross-entropy on expert vs. agent (s,a)
      2. PPO update: using adversarial reward r_adv = log D(s,a)
    
    The environment reward is completely replaced by r_adv during training.
    
    Args:
        env_id: Gymnasium environment ID
        demos_path: Path to .npz demonstration file
        total_timesteps: Total environment steps for training (default 2M)
        horizon: PPO rollout horizon
        n_ppo_epochs: PPO epochs per rollout
        batch_size: PPO batch size
        lr_policy: Policy learning rate
        lr_disc: Discriminator learning rate
        disc_updates_per_rollout: Discriminator updates per PPO rollout
        gamma, gae_lambda, clip_eps, ent_coef, vf_coef, max_grad_norm: PPO hyperparams
        seed: Random seed (None generates random seed, otherwise use provided)
        experiment_name: Name for logging/config
        tensorboard_log: Base directory for TensorBoard logs
        device: torch device
        
    Returns:
        (policy, discriminator, episode_returns)
    """
    # Generate random seed if not provided
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    
    set_global_seed(seed)
    device_obj = torch.device(device)
    
    print(f"\n{'='*70}")
    print("Training GAIL (Generative Adversarial Imitation Learning)")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Seed: {seed}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Device: {device_obj}")
    print(f"{'='*70}\n")
    
    # Load demonstrations
    demo_obs, demo_acts = load_expert_dataset(demos_path)
    demo_obs = torch.from_numpy(demo_obs).to(device_obj)
    demo_acts = torch.from_numpy(demo_acts).to(device_obj)
    n_demos = len(demo_obs)
    print(f"Loaded {n_demos} expert demonstrations")
    
    env = make_env(env_id, seed=seed, render_mode=None)
    n_actions = env.action_space.n
    
    # Initialize policy and discriminator
    policy = AtariActorCritic(n_actions).to(device_obj)
    disc = GAILDiscriminator(n_actions, use_action=False).to(device_obj)
    
    opt_policy = optim.Adam(policy.parameters(), lr=lr_policy)
    opt_disc = optim.Adam(disc.parameters(), lr=lr_disc)
    bce = nn.BCELoss()
    
    # TensorBoard logging
    run_name = f"{experiment_name}_seed_{seed}_{int(time.time())}"
    run_dir = Path(tensorboard_log) / run_name
    writer = SummaryWriter(log_dir=str(run_dir))
    
    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    all_returns: List[float] = []
    steps = 0
    
    print(f"TensorBoard logs: {run_dir}")
    print(f"Starting GAIL training with {total_timesteps:,} total timesteps...\n")
    
    while steps < total_timesteps:
        # --- Rollout collection ---
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf = [], [], []
        
        rollout_len = min(horizon, total_timesteps - steps)
        
        for _ in range(rollout_len):
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device_obj)
            with torch.no_grad():
                logits, value = policy(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()

            obs_buf.append(obs_t.squeeze(0))
            act_buf.append(action)
            logp_buf.append(dist.log_prob(action))
            val_buf.append(value.squeeze())

            obs, env_reward, terminated, truncated, _ = env.step(action.item())
            done = bool(terminated or truncated)
            done_buf.append(done)
            episode_return += float(env_reward)
            steps += 1

            if done:
                all_returns.append(episode_return)
                writer.add_scalar("train/episode_return", episode_return, steps)
                episode_return = 0.0
                obs, _ = env.reset()
        
        # --- Adversarial reward computation ---
        obs_stack = torch.stack(obs_buf).to(device_obj)
        with torch.no_grad():
            d_scores = disc(obs_stack)  # P(expert | s)
            # Reward: log D(s) encourages agent to match expert
            adv_rewards = torch.log(d_scores + 1e-8).cpu()
            rew_buf = adv_rewards.tolist()
        
        # --- GAE computation ---
        with torch.no_grad():
            obs_next = torch.from_numpy(obs).unsqueeze(0).to(device_obj)
            _, next_value = policy(obs_next)
            advantages, returns = compute_gae(
                rew_buf, val_buf, done_buf, next_value.item(), gamma, gae_lambda
            )
        
        # --- Discriminator update ---
        act_t = torch.stack(act_buf).to(device_obj)
        disc_losses = []
        
        for _ in range(disc_updates_per_rollout):
            # Sample expert batch
            idx_e = torch.randint(0, n_demos, (batch_size,))
            e_obs = demo_obs[idx_e]
            
            # Sample agent batch
            idx_a = torch.randint(0, rollout_len, (batch_size,))
            a_obs = obs_stack[idx_a]
            
            d_expert = disc(e_obs)
            d_agent = disc(a_obs)
            
            # Binary cross-entropy: expert=1, agent=0
            loss_disc = bce(d_expert, torch.ones_like(d_expert)) + \
                       bce(d_agent, torch.zeros_like(d_agent))
            
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            disc_losses.append(loss_disc.item())
        
        mean_disc_loss = float(np.mean(disc_losses)) if disc_losses else 0.0
        writer.add_scalar("train/disc_loss", mean_disc_loss, steps)
        
        # Discriminator accuracy
        with torch.no_grad():
            d_expert_acc = (d_expert > 0.5).float().mean().item()
            d_agent_acc = (d_agent < 0.5).float().mean().item()
            disc_acc = (d_expert_acc + d_agent_acc) / 2
        writer.add_scalar("train/disc_acc", disc_acc, steps)
        writer.add_scalar("train/disc_expert_p", d_expert.mean().item(), steps)
        writer.add_scalar("train/disc_agent_p", d_agent.mean().item(), steps)
        
        # --- PPO update ---
        logp_t = torch.stack(logp_buf).detach().to(device_obj)
        adv_t = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).to(device_obj)
        ret_t = returns.to(device_obj)
        
        policy_losses = []
        value_losses = []
        idx = torch.randperm(rollout_len)
        
        for _ in range(n_ppo_epochs):
            for start in range(0, rollout_len, batch_size):
                mb_idx = idx[start : start + batch_size]
                logits, values = policy(obs_stack[mb_idx])
                dist = Categorical(logits=logits)
                logp_new = dist.log_prob(act_t[mb_idx])
                entropy = dist.entropy().mean()
                
                ratio = (logp_new - logp_t[mb_idx]).exp()
                surr1 = ratio * adv_t[mb_idx]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[mb_idx]
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (values - ret_t[mb_idx]).pow(2).mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
                
                opt_policy.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt_policy.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        
        mean_policy_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
        mean_value_loss = float(np.mean(value_losses)) if value_losses else 0.0
        mean_return = float(np.mean(all_returns[-10:])) if all_returns else 0.0
        
        writer.add_scalar("train/policy_loss", mean_policy_loss, steps)
        writer.add_scalar("train/value_loss", mean_value_loss, steps)
        writer.add_scalar("train/mean_return_10ep", mean_return, steps)
        
        # Progress reporting - every 5 rollouts (~10k steps with horizon=2048)
        progress_percent = (steps / total_timesteps) * 100
        filled = int(40 * steps / total_timesteps)
        bar = '█' * filled + '░' * (40 - filled)
        print(f"\r[{bar}] {progress_percent:5.1f}% ({steps:,}/{total_timesteps:,} steps) | "
              f"Ret: {mean_return:7.2f} | Disc loss: {mean_disc_loss:.4f} | Disc acc: {disc_acc:.3f}", 
              end='', flush=True)
    
    writer.close()
    env.close()
    
    print(f"\n\n{'='*60}")
    print("GAIL training complete")
    print(f"{'='*70}")
    print(f"Final mean return (last 10 episodes): {mean_return:.2f}")
    print(f"Seed: {seed}")
    print(f"Logs: {run_dir}")
    print(f"Seeds file: {SEEDS_FILE}")
    print(f"{'='*70}\n")
    
    # Save model
    Path(model_path := str(DEFAULT_MODEL_PATH)).parent.mkdir(parents=True, exist_ok=True)
    save_model(policy, model_path, {
        "total_timesteps": total_timesteps,
        "horizon": horizon,
        "n_ppo_epochs": n_ppo_epochs,
        "batch_size": batch_size,
        "lr_policy": lr_policy,
        "lr_disc": lr_disc,
        "disc_updates_per_rollout": disc_updates_per_rollout,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_eps": clip_eps,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
    })
    
    append_config(experiment_name, {
        "total_timesteps": total_timesteps,
        "horizon": horizon,
        "n_ppo_epochs": n_ppo_epochs,
        "batch_size": batch_size,
        "lr_policy": lr_policy,
        "lr_disc": lr_disc,
        "disc_updates_per_rollout": disc_updates_per_rollout,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_eps": clip_eps,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
    }, note=f"GAIL {experiment_name} training")
    
    record_seed(experiment_name, seed, note="GAIL training run")
    
    return policy, disc, all_returns


def play_agent(model_path: str, episodes: int = 3, seed: int = 42) -> None:
    """Play a trained GAIL agent in rendered environment."""
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
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(obs_t)
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
        description="GAIL on Atari Solaris (Challenge 4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--mode",
        choices=["collect-demos", "train-bc", "train-gail", "play", "inspect"],
        required=True,
        help="Mode: collect demonstrations, train BC baseline, train GAIL, play, or inspect model"
    )
    
    # Demonstration collection args
    parser.add_argument(
        "--checkpoint-path",
        help="Path to DQN/PPO checkpoint (for collect-demos mode)"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50000,
        help="Number of demonstration steps to collect"
    )
    
    # BC training args
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=25,
        help="Number of epochs for BC training"
    )
    
    # GAIL training args
    parser.add_argument(
        "--experiment",
        default="gail_baseline",
        help="Experiment name (used for config and logging)"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2000000,
        help="Total environment steps for GAIL training (default 2M)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2048,
        help="PPO rollout horizon"
    )
    parser.add_argument(
        "--disc-lr",
        type=float,
        default=3e-4,
        help="Discriminator learning rate"
    )
    parser.add_argument(
        "--disc-updates",
        type=int,
        default=5,
        help="Discriminator updates per PPO rollout"
    )
    
    # Common args
    parser.add_argument(
        "--demos",
        type=str,
        default="demonstrations/demos.npz",
        help="Path to demonstrations .npz file"
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Model path (without .pth extension)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demonstrations/demos.npz",
        help="Output path for collected demonstrations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (None = generate random seed automatically)"
    )
    parser.add_argument(
        "--run-seeds",
        type=int,
        default=1,
        help="Number of times to run experiment with different random seeds (default 1)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
        help="Number of episodes to play (play mode)"
    )
    parser.add_argument(
        "--tensorboard-log",
        default=str(TENSORBOARD_LOG_DIR),
        help="Directory for TensorBoard logs"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_default_config_file()

    if args.mode == "collect-demos":
        if not args.checkpoint_path:
            raise ValueError("collect-demos mode requires --checkpoint-path")
        collect_demonstrations(
            env_id=ENV_ID,
            checkpoint_path=args.checkpoint_path,
            n_steps=args.n_steps,
            seed=args.seed,
            output_path=args.output,
        )

    elif args.mode == "train-bc":
        if not os.path.exists(args.demos):
            raise FileNotFoundError(f"Demonstrations file not found: {args.demos}")
        train_bc(
            env_id=ENV_ID,
            demos_path=args.demos,
            n_epochs=args.n_epochs,
            model_path=args.model_path,
            seed=args.seed,
        )

    elif args.mode == "train-gail":
        if not os.path.exists(args.demos):
            raise FileNotFoundError(f"Demonstrations file not found: {args.demos}")
        
        # Load experiment config if it exists
        hparams = None
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                configs = json.load(f)
            if not isinstance(configs, list):
                configs = [configs]
            experiment = next((cfg for cfg in configs if cfg.get("name") == args.experiment), None)
            if experiment is not None:
                hparams = {k: v for k, v in experiment.items() if k not in {"name", "note", "demo_steps", "demo_quality", "ppo_checkpoint_source"}}
        
        # Use config or command-line args
        if hparams is None:
            hparams = {
                "total_timesteps": args.total_timesteps,
                "horizon": args.horizon,
                "lr_disc": args.disc_lr,
                "disc_updates_per_rollout": args.disc_updates,
            }
        
        # Run multiple seeds if requested
        num_seeds = args.run_seeds
        for seed_run in range(num_seeds):
            print(f"\n{'='*70}")
            print(f"Running seed {seed_run + 1} of {num_seeds}")
            print(f"{'='*70}\n")
            
            policy, disc, returns = train_gail(
                env_id=ENV_ID,
                demos_path=args.demos,
                **hparams,
                experiment_name=args.experiment,
                tensorboard_log=args.tensorboard_log,
                seed=args.seed,
            )
        
        print(f"\nAll {num_seeds} seed(s) complete!")
        print(f"Model saved to {args.model_path}.pth")
        print(f"Configs stored in {CONFIG_FILE}")
        print(f"Seeds stored in {SEEDS_FILE}")
        print(f"TensorBoard logs: {args.tensorboard_log}")

    elif args.mode == "play":
        play_agent(model_path=args.model_path, episodes=args.episodes, seed=args.seed)

    elif args.mode == "inspect":
        inspect_model(model_path=args.model_path)


if __name__ == "__main__":
    main()
