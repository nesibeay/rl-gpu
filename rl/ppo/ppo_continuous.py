# rl/ppo/ppo_continuous.py
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv


@dataclass
class PPOConfig:
    env_id: str
    seed: int
    device: str
    num_envs: int
    total_timesteps: int
    rollout_steps: int
    gamma: float
    gae_lambda: float
    update_epochs: int
    minibatch_size: int
    clip_coef: float
    vf_coef: float
    ent_coef: float
    learning_rate: float
    net_hidden_sizes: list


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.v = nn.Linear(last, 1)

    def forward(self, x):
        h = self.body(x)
        mu = self.mu(h)
        v = self.v(h).squeeze(-1)
        std = torch.exp(self.log_std)
        return mu, std, v

    def act(self, x, action_scale):
        mu, std, v = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()  # reparameterized
        logprob_u = dist.log_prob(u).sum(-1)
        a = torch.tanh(u) * action_scale  # squash to [-1,1] then scale
        return a, u, logprob_u, v

    def evaluate(self, x, u):
        mu, std, v = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        logprob_u = dist.log_prob(u).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob_u, entropy, v


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def ppo_train_continuous(cfg: PPOConfig):
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    # Vectorized envs (Gymnasium 1.2.0: use SyncVectorEnv)
    def make_env(seed_offset: int):
        def _thunk():
            env = gym.make(cfg.env_id)
            env.reset(seed=cfg.seed + seed_offset)
            return env
        return _thunk

    envs = SyncVectorEnv([make_env(i) for i in range(cfg.num_envs)])

    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    assert len(act_space.shape) == 1, "Assumes 1D continuous actions."

    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    action_high = torch.as_tensor(act_space.high, dtype=torch.float32, device=device)
    action_scale = action_high  # assumes symmetric [-high, high]

    policy = ActorCritic(obs_dim, act_dim, cfg.net_hidden_sizes).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    T = cfg.rollout_steps
    updates = math.ceil(cfg.total_timesteps / (T * cfg.num_envs))

    # Buffers
    obs_buf = torch.zeros((T, cfg.num_envs, obs_dim), device=device)
    u_buf = torch.zeros((T, cfg.num_envs, act_dim), device=device)  # pre-tanh samples
    logp_buf = torch.zeros((T, cfg.num_envs), device=device)
    rew_buf = torch.zeros((T, cfg.num_envs), device=device)
    done_buf = torch.zeros((T, cfg.num_envs), device=device)
    val_buf = torch.zeros((T, cfg.num_envs), device=device)

    # Reset and get first obs
    obs_np, _ = envs.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

    global_steps = 0

    # Episode-return trackers (manual, cross-version safe)
    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    finished_returns = []

    for update in range(1, updates + 1):
        policy.eval()
        with torch.no_grad():
            for t in range(T):
                obs_buf[t] = obs
                a, u, logp_u, v = policy.act(obs, action_scale)
                u_buf[t] = u
                logp_buf[t] = logp_u
                val_buf[t] = v

                next_obs, reward, terminated, truncated, infos = envs.step(a.cpu().numpy())
                done = np.logical_or(terminated, truncated)

                rew_buf[t] = torch.tensor(reward, dtype=torch.float32, device=device)
                done_buf[t] = torch.tensor(done, dtype=torch.float32, device=device)

                # Track episodic returns manually
                ep_returns += reward  # vector add per env
                for i, d in enumerate(done):
                    if d:
                        finished_returns.append(ep_returns[i])
                        ep_returns[i] = 0.0

                obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
                global_steps += cfg.num_envs

            # Bootstrap value for the last obs
            _, _, next_v = policy.forward(obs)

        # ------- GAE -------
        adv = torch.zeros_like(rew_buf, device=device)
        lastgaelam = torch.zeros(cfg.num_envs, device=device)
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterm = 1.0 - done_buf[t]
                nextvalues = next_v
            else:
                nextnonterm = 1.0 - done_buf[t + 1]
                nextvalues = val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * nextvalues * nextnonterm - val_buf[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterm * lastgaelam
            adv[t] = lastgaelam
        ret = adv + val_buf

        # Flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_u = u_buf.reshape(-1, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)

        # Advantage norm
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # ------- PPO update -------
        policy.train()
        batch_size = b_obs.shape[0]
        idxs = np.arange(batch_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, cfg.minibatch_size):
                mb = idxs[start:start + cfg.minibatch_size]
                mb_obs = b_obs[mb]
                mb_u = b_u[mb]
                old_logp = b_logp[mb]
                mb_adv = b_adv[mb]
                mb_ret = b_ret[mb]

                new_logp, entropy, v = policy.evaluate(mb_obs, mb_u)
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (mb_ret - v).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_bonus
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        if update == 1 or update % 10 == 0:
            tr = float(np.mean(finished_returns[-10:])) if finished_returns else float("nan")
            print(f"Update {update:4d}/{updates} | Steps {global_steps:7d} | "
                  f"TrainReturn(last10) {tr:7.2f}")

    envs.close()
    print("Training done.")
