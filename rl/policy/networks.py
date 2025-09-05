# rl/policy/networks.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def orthogonal_init(m, gain=math.sqrt(2.0)):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: Tuple[int, ...] = (64, 64)):
        super().__init__()
        layers = []
        last = in_features
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.net = nn.Sequential(*layers)
        self.apply(orthogonal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    """Lightweight CNN for image observations (channels-first)."""
    def __init__(self, in_shape: Tuple[int, int, int]):
        super().__init__()
        c, h, w = in_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        # compute conv output size
        with torch.no_grad():
            o = self.conv(torch.zeros(1, c, h, w))
        self.out_dim = int(o.view(1, -1).shape[1])
        self.fc = nn.Sequential(nn.Linear(self.out_dim, 512), nn.ReLU())
        self.apply(orthogonal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


@dataclass
class PolicySpec:
    obs_is_image: bool
    obs_flat_dim: int
    action_dim: int
    action_discrete: bool


def infer_spec(obs_space, action_space) -> PolicySpec:
    from gymnasium.spaces import Box, Discrete
    # Observation
    if isinstance(obs_space, Box) and len(obs_space.shape) == 3:
        c, h, w = obs_space.shape
        obs_is_image = True
        obs_flat_dim = int(c * h * w)
    else:
        obs_is_image = False
        obs_flat_dim = int(np.prod(obs_space.shape))
    # Action
    if isinstance(action_space, Discrete):
        action_discrete = True
        action_dim = int(action_space.n)
    else:
        action_discrete = False
        action_dim = int(np.prod(action_space.shape))
    return PolicySpec(obs_is_image, obs_flat_dim, action_dim, action_discrete)


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space, hidden=(64, 64)):
        super().__init__()
        self.spec = infer_spec(obs_space, action_space)

        # Base
        if self.spec.obs_is_image:
            c, h, w = obs_space.shape  # expect channels-first
            self.base = SimpleCNN((c, h, w))
            base_out = 512
        else:
            in_dim = self.spec.obs_flat_dim
            self.base = MLP(in_dim, hidden)
            base_out = hidden[-1] if hidden else in_dim

        # Heads
        if self.spec.action_discrete:
            self.actor = nn.Linear(base_out, self.spec.action_dim)
            self.log_std = None
        else:
            self.actor_mean = nn.Linear(base_out, self.spec.action_dim)
            self.log_std = nn.Parameter(torch.zeros(1, self.spec.action_dim))

        self.critic = nn.Linear(base_out, 1)

        # Inits
        self.apply(orthogonal_init)
        if hasattr(self, "actor") and isinstance(self.actor, nn.Linear):
            nn.init.orthogonal_(self.actor.weight, 0.01)
        if hasattr(self, "actor_mean"):
            nn.init.orthogonal_(self.actor_mean.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    # Feature extractor (NCHW if image, else flat)
    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.base(obs)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic(features)

    def act(self, features: torch.Tensor, deterministic: bool = False):
        if self.spec.action_discrete:
            logits = self.actor(features)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.mode if deterministic else dist.sample()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, logprob, entropy
        else:
            mean = self.actor_mean(features)
            std = self.log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            action = mean if deterministic else dist.sample()
            logprob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
            return action, logprob, entropy

    def log_prob(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.spec.action_discrete:
            logits = self.actor(features)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(actions)
        else:
            mean = self.actor_mean(features)
            std = self.log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            return dist.log_prob(actions).sum(-1)
