# rl/ppo/ppo.py — unified PPO (continuous + discrete)
from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Dict, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from rl.utils.envs import make_vec_env
from rl.policy.networks import (
    ActorCriticContinuous,
    ActorCriticDiscrete,
)


# -----------------------------
# Utils
# -----------------------------

def select_device(name: str) -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# -----------------------------
# Config
# -----------------------------

@dataclass
class PPOConfig:
    env_id: str = "Pendulum-v1"
    hidden_sizes: Tuple[int, ...] = (64, 64)
    seed: int = 1
    total_timesteps: int = 300_000
    num_envs: int = 16
    rollout_steps: int = 256
    n_epochs: int = 10
    minibatch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    anneal_lr: bool = True
    device: str = "auto"
    vector_env: str = "sync"  # "sync" | "async"
    use_amp: bool = False
    # allow either bool or string mode like "max-autotune"
    use_compile: Union[bool, str] = False
    # optional extra perf knob from YAML; ignored if None
    float32_matmul_precision: Optional[str] = None


# -----------------------------
# Buffers
# -----------------------------

class RolloutBuffer:
    def __init__(self, obs_shape, act_dim: int, size: int, num_envs: int, device: torch.device, mode: str):
        self.mode = mode
        self.device = device
        self.size = size
        self.num_envs = num_envs

        self.obs = torch.zeros((size, num_envs) + obs_shape, dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.terminated = torch.zeros((size, num_envs), dtype=torch.bool, device=device)
        self.truncated = torch.zeros((size, num_envs), dtype=torch.bool, device=device)
        self.values = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((size, num_envs), dtype=torch.float32, device=device)

        if mode == "continuous":
            self.actions = torch.zeros((size, num_envs, act_dim), dtype=torch.float32, device=device)
        elif mode == "discrete":
            self.actions = torch.zeros((size, num_envs), dtype=torch.long, device=device)
        else:
            raise ValueError(f"Unknown mode for buffer: {mode}")

    def get_flat(self):
        def flat(x):
            if x.dim() > 2:
                return x.reshape(-1, *x.shape[2:])
            return x.reshape(-1)
        out = {
            "obs": flat(self.obs),
            "logprobs": flat(self.logprobs),
            "returns": flat(self.returns),
            "advantages": flat(self.advantages),
            "values": flat(self.values),
        }
        if self.mode == "continuous":
            out["actions"] = flat(self.actions)
        else:
            out["actions"] = flat(self.actions).long()
        return out


# -----------------------------
# PPO
# -----------------------------

class PPO:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = select_device(cfg.device)

        # Env & spaces
        self.env = make_vec_env(cfg.env_id, cfg.num_envs, cfg.seed, cfg.vector_env)
        obs, _ = self.env.reset(seed=cfg.seed)
        self.obs_shape = obs.shape[1:]  # (obs_dim,)
        single_act = self.env.single_action_space

        # Policy by action space
        if isinstance(single_act, Box):
            self.mode = "continuous"
            self.act_low = torch.as_tensor(single_act.low, device=self.device, dtype=torch.float32)
            self.act_high = torch.as_tensor(single_act.high, device=self.device, dtype=torch.float32)
            self.act_dim = single_act.shape[0]
            hidden = tuple(getattr(cfg, "hidden_sizes", (64, 64)))
            self.net = ActorCriticContinuous(int(np.prod(self.obs_shape)), self.act_dim, hidden=hidden).to(self.device)
        elif isinstance(single_act, Discrete):
            self.mode = "discrete"
            self.n_actions = int(single_act.n)
            hidden = tuple(getattr(cfg, "hidden_sizes", (64, 64)))
            self.net = ActorCriticDiscrete(int(np.prod(self.obs_shape)), self.n_actions, hidden=hidden).to(self.device)
        else:
            raise ValueError("Unsupported action space type.")

        # Optional compile (PyTorch 2)
        if cfg.use_compile and hasattr(torch, "compile"):
            compile_kwargs = {}
            if isinstance(cfg.use_compile, str):
                compile_kwargs["mode"] = cfg.use_compile  # e.g., "max-autotune"
            self.net = torch.compile(self.net, **compile_kwargs)

        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

        # Stats
        self.global_step = 0
        self.episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(cfg.num_envs, dtype=np.int32)
        self.completed_returns = []
        self.completed_lengths = []

    # -----------------
    # Helpers
    # -----------------
    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return x

    def _scale_action(self, a_tanh: torch.Tensor) -> torch.Tensor:
        return self.act_low + (a_tanh + 1.0) * 0.5 * (self.act_high - self.act_low)

    def _compute_gae(self, buffer: RolloutBuffer, last_value: torch.Tensor):
        cfg = self.cfg
        gae = torch.zeros(buffer.rewards.size(1), device=self.device)
        for t in reversed(range(cfg.rollout_steps)):
            not_done = (~buffer.terminated[t]).float()
            next_values = last_value if t == cfg.rollout_steps - 1 else buffer.values[t + 1]
            delta = buffer.rewards[t] + cfg.gamma * next_values * not_done - buffer.values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * not_done * gae
            buffer.advantages[t] = gae
        buffer.returns = buffer.advantages + buffer.values

    # -----------------
    # Update
    # -----------------
    def _update(self, buffer: RolloutBuffer, epoch: int):
        cfg = self.cfg
        B = cfg.rollout_steps * cfg.num_envs
        flat = buffer.get_flat()
        b_inds = np.arange(B)

        adv = flat["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat["advantages"] = adv

        if cfg.anneal_lr:
            frac = 1.0 - (self.global_step / max(cfg.total_timesteps, 1))
            lr = cfg.lr * frac
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        for _ in range(cfg.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, B, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb = slice(start, end)
                obs_b = flat["obs"][b_inds[mb]]
                old_logp_b = flat["logprobs"][b_inds[mb]]
                returns_b = flat["returns"][b_inds[mb]]
                adv_b = flat["advantages"][b_inds[mb]]

                if self.mode == "continuous":
                    actions_b = flat["actions"][b_inds[mb]]  # (MB, act_dim)
                    mean, std = self.net.dist_params(obs_b)
                    normal = torch.distributions.Normal(mean, std)
                    eps = 1e-6
                    a_clipped = torch.clamp(actions_b, -1 + eps, 1 - eps)
                    u = 0.5 * (torch.log1p(a_clipped) - torch.log1p(-a_clipped))  # atanh
                    log_prob_u = normal.log_prob(u).sum(-1)
                    correction = torch.log(1 - a_clipped.pow(2) + eps).sum(-1)
                    logp = log_prob_u - correction
                    entropy = normal.entropy().sum(-1).mean()
                else:
                    actions_b = flat["actions"][b_inds[mb]].long()
                    logits, _ = self.net.forward(obs_b)
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(actions_b)
                    entropy = dist.entropy().mean()

                v = self.net.value(obs_b)
                v_clipped = v + (flat["values"][b_inds[mb]] - v).clamp(-cfg.clip_coef, cfg.clip_coef)
                v_loss_unclipped = (v - returns_b).pow(2)
                v_loss_clipped = (v_clipped - returns_b).pow(2)
                v_loss = 0.5 * torch.maximum(v_loss_unclipped, v_loss_clipped).mean()

                ratio = (logp - old_logp_b).exp()
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

    # -----------------
    # Train
    # -----------------
    def train(self) -> Dict[str, float]:
        cfg = self.cfg
        obs, _ = self.env.reset(seed=cfg.seed)
        buffer = RolloutBuffer(self.obs_shape,
                               getattr(self, "act_dim", getattr(self, "n_actions", 1)),
                               cfg.rollout_steps, cfg.num_envs, self.device, self.mode)

        start_time = time.time()
        while self.global_step < cfg.total_timesteps:
            for t in range(cfg.rollout_steps):
                with torch.no_grad():
                    x = self._obs_to_tensor(obs)
                    if self.mode == "continuous":
                        mean, std = self.net.dist_params(x)
                        normal = torch.distributions.Normal(mean, std)
                        u = normal.rsample()
                        a_tanh = torch.tanh(u)
                        log_prob_u = normal.log_prob(u).sum(-1)
                        correction = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(-1)
                        logp = log_prob_u - correction
                        actions_env = self._scale_action(a_tanh)
                        v = self.net.value(x)
                        buffer.actions[t] = a_tanh
                    else:
                        logits, v = self.net.forward(x)
                        dist = torch.distributions.Categorical(logits=logits)
                        a = dist.sample()
                        logp = dist.log_prob(a)
                        actions_env = a
                        buffer.actions[t] = a

                    buffer.obs[t] = x
                    buffer.logprobs[t] = logp
                    buffer.values[t] = v

                # Step env
                act = actions_env
                if isinstance(act, torch.Tensor):
                    act_np = act.cpu().numpy()
                else:
                    act_np = act
                next_obs, reward, terminated, truncated, _ = self.env.step(act_np)
                buffer.rewards[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
                buffer.terminated[t] = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
                buffer.truncated[t] = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

                self.episode_returns += reward
                self.episode_lengths += 1
                for i in range(cfg.num_envs):
                    if terminated[i] or truncated[i]:
                        self.completed_returns.append(self.episode_returns[i])
                        self.completed_lengths.append(self.episode_lengths[i])
                        self.episode_returns[i] = 0.0
                        self.episode_lengths[i] = 0

                obs = next_obs
                self.global_step += cfg.num_envs
                if self.global_step >= cfg.total_timesteps:
                    break

            with torch.no_grad():
                x = self._obs_to_tensor(obs)
                last_value = self.net.value(x)

            self._compute_gae(buffer, last_value)
            self._update(buffer, epoch=self.global_step // (cfg.rollout_steps * cfg.num_envs))

            if self.completed_returns:
                mean_r = float(np.mean(self.completed_returns[-10:]))
                mean_l = float(np.mean(self.completed_lengths[-10:]))
            else:
                mean_r, mean_l = float("nan"), float("nan")
            elapsed = time.time() - start_time
            fps = int(self.global_step / elapsed) if elapsed > 0 else 0
            print(f"step {self.global_step:>7} | ep_ret {mean_r:8.1f} | ep_len {mean_l:6.1f} | fps {fps}")

        return {
            "steps": self.global_step,
            "mean_return": float(np.mean(self.completed_returns[-100:])) if self.completed_returns else float("nan"),
        }
