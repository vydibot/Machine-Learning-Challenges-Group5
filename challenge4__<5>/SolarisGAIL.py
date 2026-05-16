"""GAIL for Atari Solaris — imitation learning using expert trajectories.

This script implements a minimal GAIL loop reusing the Atari preprocessing
from the PPO script. It expects an expert dataset saved as a NumPy `.npz`
with arrays `obs` (uint8 or float32) and `acts` (int actions).

Usage examples:
  python SolarisGAIL.py --mode train --expert expert_dataset.npz --model-path models/gail_solaris
  python SolarisGAIL.py --mode play --model-path models/gail_solaris --episodes 3
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

# Basic Atari settings compatible with Solaris.py
ENV_ID = "ALE/Solaris-v5"
N_STACK = 4


def make_env(env_id: str, seed: int = 0, render_mode: Optional[str] = None):
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
        self.actor = nn.Sequential(nn.Linear(cnn_output_size, 512), nn.ReLU(), nn.Linear(512, n_actions))
        self.critic = nn.Sequential(nn.Linear(cnn_output_size, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.cnn(x)
        return self.actor(features), self.critic(features).squeeze(-1)


class Discriminator(nn.Module):
    """Predicts probability that (s,a) comes from expert dataset."""

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
        feat_size = 64 * 7 * 7
        self.net = nn.Sequential(nn.Linear(feat_size + n_actions, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, obs: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(obs)
        acts_onehot = torch.nn.functional.one_hot(acts.long(), num_classes=self.net[0].in_features - feats.shape[1])
        x = torch.cat([feats, acts_onehot.float()], dim=1)
        return self.net(x).squeeze(-1)


def load_expert_dataset(path: str, max_samples: Optional[int] = None):
    """Load expert dataset saved as a .npz with `obs` and `acts` arrays.

    obs should be shaped like (N, C, H, W) or (N, H, W, C). Function
    converts to float32 and channels-first if needed.
    """
    data = np.load(path)
    obs = data["obs"]
    acts = data["acts"]
    if obs.ndim == 4 and obs.shape[-1] == N_STACK:
        # convert HWC -> CHW
        obs = np.transpose(obs, (0, 3, 1, 2))
    obs = obs.astype(np.float32) / 255.0 if obs.dtype == np.uint8 else obs.astype(np.float32)
    if max_samples is not None:
        idx = np.random.choice(len(obs), size=min(max_samples, len(obs)), replace=False)
        obs = obs[idx]
        acts = acts[idx]
    return obs, acts


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    values_t = torch.tensor(values, dtype=torch.float32)
    dones_t = torch.tensor(dones, dtype=torch.float32)

    advantages = torch.zeros_like(rewards_t)
    last_adv = 0.0
    for step in reversed(range(len(rewards_t))):
        mask = 1.0 - dones_t[step]
        delta = rewards_t[step] + gamma * next_value * mask - values_t[step]
        advantages[step] = last_adv = delta + gamma * gae_lambda * mask * last_adv
        next_value = values_t[step]
    returns = advantages + values_t
    return advantages, returns


def train_gail(
    model_path: str,
    expert_path: str,
    timesteps: int = 200000,
    seed: int = 42,
    device: Optional[torch.device] = None,
):
    set_seed(seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    env = make_env(ENV_ID, seed=seed, render_mode=None)
    n_actions = env.action_space.n

    # models
    policy = AtariActorCritic(n_actions).to(device)
    disc = Discriminator(n_actions).to(device)

    optim_policy = optim.Adam(policy.parameters(), lr=2.5e-4)
    optim_disc = optim.Adam(disc.parameters(), lr=1e-4)

    # load expert data
    expert_obs, expert_acts = load_expert_dataset(expert_path)
    expert_obs = torch.from_numpy(expert_obs).to(device)
    expert_acts = torch.from_numpy(expert_acts).to(device)

    writer = SummaryWriter(log_dir=f"logs/gail_solaris/{int(time.time())}")

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    all_returns: List[float] = []
    steps = 0

    while steps < timesteps:
        # collect rollout
        rollout = []
        rollout_len = min(1024, timesteps - steps)
        for _ in range(rollout_len):
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value = policy(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample().item()

            rollout.append((obs.copy(), action))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            episode_return += float(reward)
            steps += 1
            if done:
                all_returns.append(episode_return)
                writer.add_scalar("train/episode_return", episode_return, steps)
                episode_return = 0.0
                obs, _ = env.reset()

        # --- update discriminator ---
        # prepare tensors
        policy_obs = np.stack([x[0] for x in rollout]).astype(np.float32)
        if policy_obs.ndim == 4 and policy_obs.shape[-1] == N_STACK:
            policy_obs = np.transpose(policy_obs, (0, 3, 1, 2))
        policy_obs = torch.from_numpy(policy_obs).to(device)
        policy_acts = torch.tensor([x[1] for x in rollout], dtype=torch.int64).to(device)

        # train discriminator with balanced batches
        bsize = 64
        perm_e = torch.randperm(len(expert_obs))
        perm_p = torch.randperm(len(policy_obs))
        total_steps = max(len(perm_e), len(perm_p)) // bsize + 1
        disc_losses = []
        for i in range(total_steps):
            e_idx = perm_e[i * bsize : (i + 1) * bsize] if i * bsize < len(perm_e) else perm_e[:bsize]
            p_idx = perm_p[i * bsize : (i + 1) * bsize] if i * bsize < len(perm_p) else perm_p[:bsize]
            e_obs = expert_obs[e_idx]
            e_acts = expert_acts[e_idx]
            p_obs = policy_obs[p_idx]
            p_acts = policy_acts[p_idx]

            pred_e = disc(e_obs, e_acts)
            pred_p = disc(p_obs, p_acts)
            # labels: expert=1, policy=0
            loss = -torch.log(pred_e + 1e-8).mean() - torch.log(1 - pred_p + 1e-8).mean()

            optim_disc.zero_grad()
            loss.backward()
            optim_disc.step()
            disc_losses.append(loss.item())

        writer.add_scalar("train/disc_loss", float(np.mean(disc_losses)), steps)

        # --- compute imitation rewards and run PPO-style update ---
        with torch.no_grad():
            d_vals = disc(policy_obs, policy_acts)
            # GAIL reward: -log(1 - D(s,a))
            imp_rewards = -torch.log(1 - d_vals + 1e-8).cpu().numpy().tolist()

        # Use PPO-style advantage estimation with imitation rewards
        # Collect values from policy for rollout
        values = []
        for o, a in rollout:
            t = torch.from_numpy(o).unsqueeze(0).to(device)
            with torch.no_grad():
                _, v = policy(t)
            values.append(v.item())

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            _, next_value = policy(obs_tensor)

        advantages, returns = compute_gae(imp_rewards, values, [False] * len(imp_rewards), next_value.item(), 0.99, 0.95)
        obs_batch = torch.stack([torch.from_numpy(x[0]) for x in rollout]).to(device)
        act_batch = torch.tensor([x[1] for x in rollout], dtype=torch.int64).to(device)
        logp_old = []
        for o, a in rollout:
            t = torch.from_numpy(o).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = policy(t)
                logp_old.append(Categorical(logits=logits).log_prob(torch.tensor(a).to(device)).cpu())
        logp_batch = torch.stack(logp_old).to(device)

        adv_batch = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).to(device)
        ret_batch = returns.to(device)

        # PPO updates (simplified, few epochs)
        policy_losses = []
        value_losses = []
        rollout_length = len(adv_batch)
        permutation = torch.randperm(rollout_length)
        for _ in range(4):
            for start in range(0, rollout_length, 64):
                idx = permutation[start : start + 64]
                logits, values_pred = policy(obs_batch[idx])
                dist = Categorical(logits=logits)
                logp_new = dist.log_prob(act_batch[idx])
                entropy = dist.entropy().mean()
                ratio = (logp_new - logp_batch[idx]).exp()
                surr1 = ratio * adv_batch[idx]
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv_batch[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (values_pred - ret_batch[idx]).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                optim_policy.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optim_policy.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        writer.add_scalar("train/policy_loss", float(np.mean(policy_losses)), steps)
        writer.add_scalar("train/value_loss", float(np.mean(value_losses)), steps)

    # save policy
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": policy.state_dict()}, f"{model_path}.pth")
    writer.close()
    env.close()


def play_agent(model_path: str, episodes: int = 3, seed: int = 42) -> None:
    env = make_env(ENV_ID, seed=seed, render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n
    policy = AtariActorCritic(n_actions).to(device)
    checkpoint = torch.load(f"{model_path}.pth", map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    completed = 0
    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    while completed < episodes:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = policy(obs_t)
            action = Categorical(logits=logits).sample().item()

        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += float(reward)
        if terminated or truncated:
            completed += 1
            print(f"Episode {completed} reward={episode_reward:.1f}")
            episode_reward = 0.0
            obs, _ = env.reset()
    env.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAIL on Atari Solaris")
    parser.add_argument("--mode", choices=["train", "play"], required=True)
    parser.add_argument("--expert", help="Path to expert .npz file with obs and acts (used in train)")
    parser.add_argument("--model-path", default="models/gail_solaris", help="Model path (without .pth)")
    parser.add_argument("--timesteps", type=int, default=200000, help="Environment steps for training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        if not args.expert:
            raise ValueError("Training requires --expert path to expert .npz dataset")
        train_gail(model_path=args.model_path, expert_path=args.expert, timesteps=args.timesteps, seed=args.seed)
    elif args.mode == "play":
        play_agent(model_path=args.model_path, episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
