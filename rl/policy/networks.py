# File: rl/policy/networks.py (The new, corrected version)

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import gymnasium as gym

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a linear layer with orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        
        # Ensure we handle both Box and Discrete spaces correctly for shape info
        obs_shape = obs_space.shape
        obs_size = np.array(obs_shape).prod()
        action_dim = np.array(action_space.shape).prod()

        # --- Critic Network ---
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # --- Actor Network (for continuous actions) ---
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        
        # Learnable parameter for the standard deviation of the action distribution
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        """Returns the value estimate for a given observation."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        """
        Returns the action, its log probability, entropy of the distribution,
        and the value estimate.
        """
        action_mean = self.actor_mean(x)
        
        # If deterministic is True, we don't sample, we just use the mean.
        if deterministic:
            return action_mean, None, None, self.critic(x)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Create a Normal distribution from which to sample actions
        probs = Normal(action_mean, action_std)
        
        # If no action is provided, sample a new one
        if action is None:
            action = probs.sample()
            
        # Calculate log probability and entropy
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        
        return action, log_prob, entropy, self.critic(x)