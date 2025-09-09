# rl/ppo/ppo.py
import math
import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

# ----------------------------
# Utilities
# ----------------------------

def device_of(cfg: Dict[str, Any]) -> torch.device:
    dev = cfg.get("device", None)
    if dev is not None:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_tf32_guard(cfg: Dict[str, Any]):
    # Only on CUDA
    if torch.cuda.is_available() and cfg.get("float32_matmul_precision", "") == "high":
        try:
            torch.set_float32_matmul_precision("high")  # enables TF32 path where applicable
            print("Set torch.float32 matmul precision -> high")
        except Exception as e:
            print(f"[warn] set_float32_matmul_precision failed: {e}")

def maybe_compile(module: nn.Module, enable: bool) -> nn.Module:
    if enable and torch.cuda.is_available():
        try:
            module = torch.compile(module)
            print("[compile] torch.compile enabled for module")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")
    return module

class RunningNorm:
    """
    Running mean/var normalizer. Keeps buffers on chosen device.
    """
    def __init__(self, shape, eps=1e-8, device="cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(eps, device=device)
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        # x: [B, obs_dim]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        b = x.shape[0]
        if b == 0:
            return
        m = x.mean(0)
        v = x.var(0, unbiased=False)
        tot = self.count + b
        delta = m - self.mean
        new_mean = self.mean + delta * (b / tot)
        new_var = (self.var * self.count + v * b + delta.pow(2) * self.count * b / tot) / tot
        self.mean, self.var, self.count = new_mean, new_var.clamp_min(self.eps), tot

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

# ----------------------------
# Policy / Value Nets
# ----------------------------

def mlp(sizes, activation="tanh", out_act=None):
    acts = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(acts.get(activation, nn.Tanh)())
        elif out_act:
            layers.append(out_act())
    return nn.Sequential(*layers)

class ActorContinuous(nn.Module):
    """
    Tanh-squashed Gaussian with log-prob correction (Appendix C of SAC / PPO conventions).
    """
    def __init__(self, obs_dim, act_dim, hidden, activation="tanh", init_log_std=-0.5):
        super().__init__()
        self.net = mlp([obs_dim] + hidden + [act_dim], activation=activation)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def forward(self, obs):
        mu = self.net(obs)
        std = torch.exp(self.log_std).clamp_min(1e-6)
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)
        # Change of variables (tanh)
        log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)
        return a, log_prob, mu, std

    def log_prob(self, obs, actions):
        # inverse tanh: clip to avoid NaN
        atanh = torch.atanh(actions.clamp(-0.999999, 0.999999))
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        lp = dist.log_prob(atanh) - torch.log(1 - actions.pow(2) + 1e-6)
        return lp.sum(-1)

class ActorDiscrete(nn.Module):
    def __init__(self, obs_dim, n_act, hidden, activation="tanh"):
        super().__init__()
        self.net = mlp([obs_dim] + hidden + [n_act], activation=activation)

    def forward(self, obs):
        logits = self.net(obs)
        return logits

    def sample(self, obs):
        logits = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        return a, log_prob, logits, None

    def log_prob(self, obs, actions):
        logits = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden, activation="tanh"):
        super().__init__()
        self.v = mlp([obs_dim] + hidden + [1], activation=activation)

    def forward(self, obs):
        return self.v(obs).squeeze(-1)

class ActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, cfg):
        super().__init__()
        obs_dim = int(np.prod(obs_space.shape))
        hidden_actor = list(cfg.get("actor_hidden_sizes", [64, 64]))
        hidden_critic = list(cfg.get("critic_hidden_sizes", [64, 64]))
        activation = cfg.get("activation", "tanh")
        self.is_continuous = isinstance(act_space, Box)

        if self.is_continuous:
            act_dim = int(np.prod(act_space.shape))
            self.actor = ActorContinuous(obs_dim, act_dim, hidden_actor, activation, init_log_std=cfg.get("init_log_std", -0.5))
        else:
            assert isinstance(act_space, Discrete), "Only Box or Discrete action spaces are supported"
            self.actor = ActorDiscrete(obs_dim, act_space.n, hidden_actor, activation)
        self.critic = Critic(obs_dim, hidden_critic, activation)

        # action bounds if continuous
        self.action_low = None
        self.action_high = None
        if self.is_continuous:
            # Pendulum: [-2, 2]
            self.action_low = torch.as_tensor(act_space.low, dtype=torch.float32)
            self.action_high = torch.as_tensor(act_space.high, dtype=torch.float32)

    def forward(self, obs):
        raise NotImplementedError

    def act(self, obs):
        if self.is_continuous:
            a, logp, mu, std = self.actor.sample(obs)
            return a, logp
        else:
            a, logp, _, _ = self.actor.sample(obs)
            return a, logp

    def evaluate_actions(self, obs, actions):
        if self.is_continuous:
            logp = self.actor.log_prob(obs, actions)
        else:
            logp = self.actor.log_prob(obs, actions)
        v = self.critic(obs)
        return logp, v

# ----------------------------
# Buffer
# ----------------------------

@dataclass
class RolloutStorage:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor

    def to(self, device):
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.logprobs = self.logprobs.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.values = self.values.to(device)
        return self

# ----------------------------
# Env helpers
# ----------------------------

def make_env(env_id: str, seed: int, idx: int, capture_video: bool = False):
    def thunk():
        env = gym.make(env_id)
        env.reset(seed=seed + idx)
        return env
    return thunk

def action_postproc(ac: torch.Tensor, model: ActorCritic) -> torch.Tensor:
    # scale continuous actions to env bounds; discrete actions pass through
    if model.is_continuous:
        # actions are in [-1, 1] due to tanh; rescale to env range
        if model.action_low is None or model.action_high is None:
            return ac
        low = model.action_low.to(ac.device)
        high = model.action_high.to(ac.device)
        # map [-1,1] -> [low, high]
        return low + (0.5 * (ac + 1.0)) * (high - low)
    else:
        return ac

# ----------------------------
# Training
# ----------------------------

def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main PPO training loop. Expects `cfg` dict with keys such as:
      - env_id, seed, total_timesteps
      - num_envs, rollout_steps, n_epochs, minibatch_size
      - lr, gamma, gae_lambda, clip_coef, vf_clip_coef, vf_coef, ent_coef, max_grad_norm
      - reward_scale, obs_norm, adv_norm, target_kl, lr_anneal
      - use_value_clip, use_tanh_squash (kept for compatibility; squash is on for continuous)
      - use_compile, float32_matmul_precision
    """
    # --- Device & matmul precision guards ---
    dev = device_of(cfg)
    set_tf32_guard(cfg)
    use_compile = bool(cfg.get("use_compile", False)) and torch.cuda.is_available()

    # --- Build vectorized env ---
    env_id = cfg.get("env_id", "Pendulum-v1")
    seed = int(cfg.get("seed", 1))
    num_envs = int(cfg.get("num_envs", 16))
    rollout_steps = int(cfg.get("rollout_steps", 256))
    total_timesteps = int(cfg.get("total_timesteps", 300_000))

    venv = gym.vector.SyncVectorEnv([make_env(env_id, seed, i) for i in range(num_envs)])
    obs_space, act_space = venv.single_observation_space, venv.single_action_space

    # --- Model ---
    ac = ActorCritic(obs_space, act_space, cfg).to(dev)
    ac.actor = maybe_compile(ac.actor, use_compile)
    ac.critic = maybe_compile(ac.critic, use_compile)

    # --- Optimizer ---
    lr = float(cfg.get("lr", 3e-4))
    optimizer = torch.optim.AdamW(ac.parameters(), lr=lr)

    # --- PPO knobs ---
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    clip_coef = float(cfg.get("clip_coef", 0.2))
    vf_clip_coef = float(cfg.get("vf_clip_coef", 0.2))
    vf_coef = float(cfg.get("vf_coef", 0.5))
    ent_coef = float(cfg.get("ent_coef", 0.01))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))

    n_epochs = int(cfg.get("n_epochs", 10))
    minibatch_size = int(cfg.get("minibatch_size", 2048))
    target_kl = float(cfg.get("target_kl", 0.015))
    lr_anneal = str(cfg.get("lr_anneal", ""))  # "linear" or ""

    # --- Stabilizers ---
    reward_scale = float(cfg.get("reward_scale", 1.0))
    use_obs_norm = bool(cfg.get("obs_norm", False))
    adv_norm = bool(cfg.get("adv_norm", True))
    use_value_clip = bool(cfg.get("use_value_clip", True))

    # --- Rollout buffers ---
    obs_shape = (int(np.prod(obs_space.shape)),)
    if isinstance(act_space, Box):
        act_shape = (int(np.prod(act_space.shape)),)
        act_dtype = torch.float32
    else:
        act_shape = ()
        act_dtype = torch.long

    buf_size = rollout_steps * num_envs
    storage = RolloutStorage(
        obs=torch.zeros((rollout_steps, num_envs) + obs_shape, dtype=torch.float32),
        actions=torch.zeros((rollout_steps, num_envs) + act_shape, dtype=act_dtype),
        logprobs=torch.zeros((rollout_steps, num_envs), dtype=torch.float32),
        rewards=torch.zeros((rollout_steps, num_envs), dtype=torch.float32),
        dones=torch.zeros((rollout_steps, num_envs), dtype=torch.float32),
        values=torch.zeros((rollout_steps, num_envs), dtype=torch.float32),
    )

    storage = storage.to(dev)
    obs_norm = RunningNorm(shape=obs_shape, device=dev) if use_obs_norm else None

    # --- Reset envs ---
    next_obs, _ = venv.reset(seed=seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=dev)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=dev)

    # Track episodic returns/lengths
    ep_returns = []
    ep_lengths = []
    ep_ret_env = torch.zeros(num_envs, dtype=torch.float32, device=dev)
    ep_len_env = torch.zeros(num_envs, dtype=torch.float32, device=dev)

    # For FPS
    start_time = time.time()
    global_step = 0
    log_interval = num_envs * rollout_steps  # prints per update

    # ----------------------------
    # Main Loop
    # ----------------------------
    while global_step < total_timesteps:
        # ===== Collect rollout =====
        for t in range(rollout_steps):
            global_step += num_envs

            # Normalize obs if enabled
            flat_obs = next_obs.view(num_envs, -1)
            if obs_norm is not None:
                obs_norm.update(flat_obs)
                flat_obs = obs_norm.normalize(flat_obs)

            with torch.no_grad():
                # value from critic
                v = ac.critic(flat_obs)
                # action + logprob
                if ac.is_continuous:
                    a, logp = ac.act(flat_obs)
                    env_a = action_postproc(a, ac)
                    env_a_np = env_a.detach().cpu().numpy()
                else:
                    a, logp = ac.act(flat_obs)
                    env_a_np = a.detach().cpu().numpy()

            # step
            o2, r, terminated, truncated, infos = venv.step(env_a_np)
            done = np.logical_or(terminated, truncated)

            # Reward scaling (pre-GAE)
            if reward_scale != 1.0:
                r = np.asarray(r, dtype=np.float32) * reward_scale

            # log episodic info
            ep_ret_env += torch.as_tensor(r, device=dev, dtype=torch.float32)
            ep_len_env += torch.ones_like(ep_len_env)
            for i in range(num_envs):
                if done[i]:
                    ep_returns.append(float(ep_ret_env[i].item()))
                    ep_lengths.append(int(ep_len_env[i].item()))
                    ep_ret_env[i] = 0.0
                    ep_len_env[i] = 0.0

            # store
            storage.obs[t] = flat_obs
            if ac.is_continuous:
                storage.actions[t] = a
            else:
                storage.actions[t] = a
            storage.logprobs[t] = logp
            storage.values[t] = v
            storage.rewards[t] = torch.as_tensor(r, device=dev, dtype=torch.float32)
            storage.dones[t] = next_done

            # next
            next_obs = torch.as_tensor(o2, dtype=torch.float32, device=dev)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=dev)

        # print log line
        elapsed = time.time() - start_time
        fps = int(global_step / max(elapsed, 1e-6))
        disp_ret = (np.mean(ep_returns[-10:]) if len(ep_returns) else float("nan"))
        disp_len = (np.mean(ep_lengths[-10:]) if len(ep_lengths) else float("nan"))
        print(f"step {global_step:8d} | ep_ret {disp_ret:8.1f} | ep_len {disp_len:6.1f} | fps {fps}")

        # ===== Bootstrap value =====
        with torch.no_grad():
            flat_obs = next_obs.view(num_envs, -1)
            if obs_norm is not None:
                flat_obs = obs_norm.normalize(flat_obs)
            next_value = ac.critic(flat_obs)

        # ===== Compute GAE & returns =====
        advantages = torch.zeros_like(storage.rewards, device=dev)
        lastgaelam = torch.zeros(num_envs, device=dev)
        for t in reversed(range(rollout_steps)):
            nextnonterminal = 1.0 - storage.dones[t]
            nextv = next_value if t == rollout_steps - 1 else storage.values[t + 1]
            delta = storage.rewards[t] + gamma * nextv * nextnonterminal - storage.values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + storage.values

        # flatten batch
        b_obs = storage.obs.reshape(-1, storage.obs.shape[-1])
        b_actions = storage.actions.reshape(-1, *storage.actions.shape[2:])
        b_logprobs = storage.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = storage.values.reshape(-1)

        # Advantage norm over whole buffer
        if adv_norm:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # ===== PPO Updates =====
        batch_size = b_obs.shape[0]
        idxs = np.arange(batch_size)

        # LR Anneal
        if lr_anneal == "linear":
            frac = 1.0 - (global_step / float(total_timesteps))
            for g in optimizer.param_groups:
                g["lr"] = lr * frac

        for epoch in range(n_epochs):
            np.random.shuffle(idxs)
            kl_exceeded = False
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = idxs[start:end]
                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_oldlog = b_logprobs[mb_idx]
                mb_adv = b_advantages[mb_idx]
                mb_ret = b_returns[mb_idx]
                mb_val = b_values[mb_idx]

                # new logprob & value
                if not ac.is_continuous:
                    # discrete: actions are (N,)
                    newlogprob, newvalue = ac.evaluate_actions(mb_obs, mb_actions)
                else:
                    # continuous: actions are (N, act_dim)
                    newlogprob, newvalue = ac.evaluate_actions(mb_obs, mb_actions)

                logratio = newlogprob - mb_oldlog
                ratio = torch.exp(logratio)

                # Policy loss (clipped surrogate)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (with optional clipping)
                if use_value_clip:
                    v_clipped = mb_val + (newvalue - mb_val).clamp(-vf_clip_coef, vf_clip_coef)
                    v_loss_unclipped = (newvalue - mb_ret).pow(2)
                    v_loss_clipped = (v_clipped - mb_ret).pow(2)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * (newvalue - mb_ret).pow(2).mean()

                # Entropy bonus
                if ac.is_continuous:
                    # approximate entropy from std (works fine)
                    _, std = ac.actor(mb_obs)
                    entropy = (0.5 * (1.0 + math.log(2 * math.pi)) + torch.log(std)).sum(-1).mean()
                else:
                    logits = ac.actor(mb_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    entropy = dist.entropy().mean()

                loss = pg_loss + v_loss * vf_coef - ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                optimizer.step()

                # Target KL early stop
                with torch.no_grad():
                    approx_kl = (ratio.log() - (ratio - 1)).mean().abs()
                if approx_kl > target_kl:
                    kl_exceeded = True
                    break

            if kl_exceeded:
                break

    # End training — summarize
    mean_return = float(np.mean(ep_returns[-100:])) if len(ep_returns) else float("nan")
    result = {"steps": global_step, "mean_return": mean_return}

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "ppo_final.pt")
    torch.save(
        {
            "model_state_dict": ac.state_dict(),
            "cfg": cfg,
            "obs_norm": {
                "mean": (obs_norm.mean.detach().cpu() if obs_norm else None),
                "var": (obs_norm.var.detach().cpu() if obs_norm else None),
                "count": (obs_norm.count.detach().cpu() if obs_norm else None),
            } if obs_norm else None,
        },
        ckpt_path,
    )
    print(f"Training finished: {result}")
    print(f"Saved checkpoint to {ckpt_path}")
    return result
