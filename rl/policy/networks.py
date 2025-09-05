# rl/policy/networks.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete


# -----------------------------
# Initializers & small utilities
# -----------------------------

def orthogonal_init(m: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


# -----------------------------
# Specs (optional helpers)
# -----------------------------
@dataclass
class PolicySpec:
    obs_is_image: bool
    obs_flat_dim: int
    action_dim: int
    action_discrete: bool


def infer_spec(obs_space, action_space) -> PolicySpec:
    obs_is_image = False
    if isinstance(obs_space, Box):
        if len(obs_space.shape) == 3:
            c, h, w = obs_space.shape
            obs_is_image = True
            obs_flat_dim = c * h * w
        else:
            obs_flat_dim = int(torch.tensor(obs_space.shape).prod().item())
    else:
        raise ValueError("Only Box observation spaces are supported in infer_spec().")

    if isinstance(action_space, Box):
        action_dim = int(torch.tensor(action_space.shape).prod().item())
        action_discrete = False
    elif isinstance(action_space, Discrete):
        action_dim = int(action_space.n)
        action_discrete = True
    else:
        raise ValueError("Only Box or Discrete action spaces are supported in infer_spec().")

    return PolicySpec(obs_is_image, obs_flat_dim, action_dim, action_discrete)


# -----------------------------
# Backbones (MLP / Small CNN)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), activation()]
            last = h
        self.net = nn.Sequential(*layers)
        self.apply(lambda m: orthogonal_init(m, math.sqrt(2.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.flatten = Flatten()
        self._out_dim: Optional[int] = None
        self._proj: Optional[nn.Linear] = None
        self.apply(lambda m: orthogonal_init(m, math.sqrt(2.0)))

    def _ensure_proj(self, x: torch.Tensor):
        if self._out_dim is None:
            with torch.no_grad():
                y = self.flatten(self.conv(x))
                self._out_dim = y.shape[1]
            self._proj = nn.Linear(self._out_dim, 512)
            orthogonal_init(self._proj, math.sqrt(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = self.flatten(z)
        self._ensure_proj(x)
        return self._proj(z)


# -----------------------------
# Heads
# -----------------------------
class ActorCriticContinuous(nn.Module):
    """Actor-Critic for continuous control with a tanh-squashed Gaussian policy."""
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64)):
        super().__init__()
        self.body = MLP(obs_dim, hidden, activation=nn.Tanh)
        self.pi = nn.Linear(hidden[-1], act_dim)   # mean
        self.v = nn.Linear(hidden[-1], 1)          # value
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learnable
        orthogonal_init(self.pi, 0.01)
        orthogonal_init(self.v, 1.0)

    def forward(self, x: torch.Tensor):
        z = self.body(x)
        return self.pi(z), self.v(z).squeeze(-1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        _, v = self.forward(x)
        return v

    def dist_params(self, x: torch.Tensor):
        mean, _ = self.forward(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std


class ActorCriticDiscrete(nn.Module):
    """Actor-Critic for discrete control with a categorical policy."""
    def __init__(self, obs_dim: int, n_actions: int, hidden=(64, 64)):
        super().__init__()
        self.body = MLP(obs_dim, hidden, activation=nn.Tanh)
        self.logits = nn.Linear(hidden[-1], n_actions)
        self.v = nn.Linear(hidden[-1], 1)
        orthogonal_init(self.logits, 0.01)
        orthogonal_init(self.v, 1.0)

    def forward(self, x: torch.Tensor):
        z = self.body(x)
        return self.logits(z), self.v(z).squeeze(-1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        _, v = self.forward(x)
        return v
