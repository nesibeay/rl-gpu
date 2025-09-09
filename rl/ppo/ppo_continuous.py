"""
# ppo_continuous.py (Refactored Version)

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# Import your network from networks.py
from networks import ActorCritic 

@dataclass
class PPOConfig:
    """Class to hold all hyperparameters for PPO."""
    # --- Experiment settings ---
    exp_name: str = "PPO_Refactored"
    seed: int = 1
    torch_deterministic: bool = True
    
    # --- Environment settings ---
    env_id: str = "BipedalWalker-v3"
    total_timesteps: int = 2_000_000
    
    # --- Algorithm-specific settings ---
    learning_rate: float = 3e-4
    num_envs: int = 1 # In this baseline, we use 1 environment. Will be crucial for GPU parallelization.
    num_steps: int = 2048 # Steps to run in each environment per policy rollout.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # --- Runtime settings ---
    device: str = "cpu" # This is what we will change to "cuda" for GPU training.

class Agent:
    """
    The Agent class that holds the actor-critic network.
    This is a wrapper around your ActorCritic class from networks.py.
    """
    def __init__(self, envs, device):
        self.device = device
        self.net = ActorCritic(envs).to(device)

    def get_value(self, x):
        # We need to convert numpy arrays to tensors before passing to the network
        return self.net.get_value(torch.Tensor(x).to(self.device))

    def get_action_and_value(self, x, action=None):
        # We need to convert numpy arrays to tensors before passing to the network
        action, log_prob, entropy, value = self.net.get_action_and_value(torch.Tensor(x).to(self.device), action)
        return action, log_prob, entropy, value

def main():
    # 1. Initialize configuration and environment
    config = PPOConfig()
    run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    # 2. Seeding for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # 3. Setup device (CPU or GPU)
    device = torch.device(config.device)

    # 4. Environment setup
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(config.env_id) for _ in range(config.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # 5. Agent and Optimizer setup
    agent = Agent(envs, device)
    optimizer = optim.Adam(agent.net.parameters(), lr=config.learning_rate, eps=1e-5)

    # 6. Storage setup for rollouts
    batch_size = int(config.num_envs * config.num_steps)
    minibatch_size = int(batch_size // config.num_minibatches)

    obs = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)

    # 7. Main training loop
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.num_envs).to(device)
    num_updates = config.total_timesteps // batch_size

    for update in range(1, num_updates + 1):
        # Annealing the learning rate is a common practice, but we'll keep it simple for now.
        
        # --- DATA COLLECTION PHASE ---
        for step in range(0, config.num_steps):
            global_step += 1 * config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.net.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            # This is just for logging the episodic return
            # It's okay for this part to be a bit verbose for now
            if any(done):
                # In a vectorized env, we need to find which env just finished
                for i, d in enumerate(done):
                    if d:
                        # This part is a bit tricky, but it's for logging purposes only
                        # It finds the episode info in the `_info` dict
                        episode_return = envs.call('get_episode_rewards')[i]
                        if episode_return:
                            print(f"global_step={global_step}, episodic_return={episode_return[-1]}")
                            writer.add_scalar("charts/episodic_return", episode_return[-1], global_step)


        # --- LEARNING PHASE ---
        # This is where we call our new, separated learning logic
        train(config, agent, optimizer, obs, actions, logprobs, rewards, dones, values, next_obs, next_done)

        # Log performance metrics
        sps = int(global_step / (time.time() - start_time))
        print(f"Steps per second: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()

def train(config, agent, optimizer, obs, actions, logprobs, rewards, dones, values, next_obs, next_done):
    """This function contains the learning logic separated from the main loop."""
    # 1. Calculate advantages using GAE
    with torch.no_grad():
        next_value = agent.net.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(config.device)
        lastgaelam = 0
        for t in reversed(range(config.num_steps)):
            if t == config.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    # 2. Flatten the batch for training
    b_obs = obs.reshape((-1,) + obs.shape[2:])
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + actions.shape[2:])
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # 3. Optimizing the policy and value network
    b_inds = np.arange(len(b_obs))
    for epoch in range(config.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, len(b_obs), config.num_minibatches):
            end = start + config.num_minibatches
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.net.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            mb_advantages = b_advantages[mb_inds]
            
            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            # Entropy loss
            entropy_loss = entropy.mean()
            
            # Total loss
            loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.net.parameters(), config.max_grad_norm)
            optimizer.step()

if __name__ == "__main__":
    main()
"""