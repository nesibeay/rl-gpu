# rl/ppo/ppo.py
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

# ----------------------------
# Config (expands train.py whitelist)
# ----------------------------
@dataclass
class PPOConfig:
    # env / run
    env_id: str = "Pendulum-v1"
    seed: int = 1
    total_timesteps: int = 300_000
    device: Optional[str] = None

    # vec env params
    num_envs: int = 16
    rollout_steps: int = 256

    # optimizer / PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    n_epochs: int = 10
    minibatch_size: int = 2048
    target_kl: float = 0.015
    lr_anneal: str = ""                  # "", "linear"

    # nets
    actor_hidden_sizes: tuple = (64, 64)
    critic_hidden_sizes: tuple = (64, 64)
    activation: str = "tanh"
    init_log_std: float = -0.5

    # stabilizers/toggles
    adv_norm: bool = True
    use_value_clip: bool = True
    use_tanh_squash: bool = True         # reserved (continuous uses tanh)
    reward_scale: float = 1.0            # <— NEW
    obs_norm: bool = False               # <— NEW

    # perf
    use_compile: bool = False
    float32_matmul_precision: Optional[str] = None  # e.g., "high" for TF32

    # (future-proof) allow arbitrary extra keys without crashing
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}
        # flatten basic fields (ignore helper)
        d.pop("extra", None)
        return d

# ----------------------------
# Utilities
# ----------------------------
def _device_of(cfg: Dict[str, Any]) -> torch.device:
    dev = cfg.get("device", None)
    return torch.device(dev) if dev else torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _set_tf32_guard(cfg: Dict[str, Any]):
    if torch.cuda.is_available() and cfg.get("float32_matmul_precision", "") == "high":
        try:
            torch.set_float32_matmul_precision("high")
            print("Set torch.float32 matmul precision -> high")
        except Exception as e:
            print(f"[warn] set_float32_matmul_precision failed: {e}")

def _maybe_compile(m: nn.Module, enable: bool) -> nn.Module:
    if enable and torch.cuda.is_available():
        try:
            m = torch.compile(m)
            print("[compile] torch.compile enabled")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")
    return m

class RunningNorm:
    def __init__(self, shape, eps=1e-8, device="cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(eps, device=device)
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
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
# Models
# ----------------------------
def _mlp(sizes, activation="tanh"):
    acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(acts.get(activation, nn.Tanh)())
    return nn.Sequential(*layers)

class ActorContinuous(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, activation="tanh", init_log_std=-0.5):
        super().__init__()
        self.net = _mlp([obs_dim] + list(hidden) + [act_dim], activation=activation)
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
        log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        return a, log_prob.sum(-1)

    def log_prob(self, obs, actions):
        atanh = torch.atanh(actions.clamp(-0.999999, 0.999999))
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        lp = dist.log_prob(atanh) - torch.log(1 - actions.pow(2) + 1e-6)
        return lp.sum(-1)

class ActorDiscrete(nn.Module):
    def __init__(self, obs_dim, n_act, hidden, activation="tanh"):
        super().__init__()
        self.net = _mlp([obs_dim] + list(hidden) + [n_act], activation=activation)

    def forward(self, obs):
        return self.net(obs)

    def sample(self, obs):
        logits = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a)

    def log_prob(self, obs, actions):
        logits = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden, activation="tanh"):
        super().__init__()
        self.v = _mlp([obs_dim] + list(hidden) + [1], activation=activation)

    def forward(self, obs):
        return self.v(obs).squeeze(-1)

class ActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, cfg):
        super().__init__()
        obs_dim = int(np.prod(obs_space.shape))
        self.is_cont = isinstance(act_space, Box)
        act_hidden = cfg.get("actor_hidden_sizes", (64, 64))
        crt_hidden = cfg.get("critic_hidden_sizes", (64, 64))
        activation = cfg.get("activation", "tanh")
        if self.is_cont:
            act_dim = int(np.prod(act_space.shape))
            self.actor = ActorContinuous(obs_dim, act_dim, act_hidden, activation, cfg.get("init_log_std", -0.5))
        else:
            self.actor = ActorDiscrete(obs_dim, act_space.n, act_hidden, activation)
        self.critic = Critic(obs_dim, crt_hidden, activation)

        if self.is_cont:
            self.low = torch.as_tensor(act_space.low, dtype=torch.float32)
            self.high = torch.as_tensor(act_space.high, dtype=torch.float32)
        else:
            self.low = self.high = None

    def scale_action(self, a: torch.Tensor) -> torch.Tensor:
        if not self.is_cont:
            return a
        low, high = self.low.to(a.device), self.high.to(a.device)
        return low + (0.5 * (a + 1.0)) * (high - low)

# ----------------------------
# Trainer (class API expected by train.py)
# ----------------------------
class PPO:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg.to_dict() if isinstance(cfg, PPOConfig) else dict(cfg)
        self.device = _device_of(self.cfg)
        self.net = None
        self.obs_shape = None
        self.act_dim = None
        self.n_actions = None
        self.mode = "ppo"

    def train(self, writer: Optional[SummaryWriter] = None):
        cfg = self.cfg
        _set_tf32_guard(cfg)
        use_compile = bool(cfg.get("use_compile", False)) and torch.cuda.is_available()

        # Env
        env_id = cfg.get("env_id", "Pendulum-v1")
        seed = int(cfg.get("seed", 1))
        num_envs = int(cfg.get("num_envs", 16))
        rollout_steps = int(cfg.get("rollout_steps", 256))
        total_timesteps = int(cfg.get("total_timesteps", 300_000))

        venv = gym.vector.SyncVectorEnv([_make_env(env_id, seed, i) for i in range(num_envs)])
        obs_space, act_space = venv.single_observation_space, venv.single_action_space
        self.obs_shape = (int(np.prod(obs_space.shape)),)
        if isinstance(act_space, Box):
            self.act_dim = int(np.prod(act_space.shape))
        elif isinstance(act_space, Discrete):
            self.n_actions = act_space.n

        # Model
        ac = ActorCritic(obs_space, act_space, cfg).to(self.device)
        ac.actor = _maybe_compile(ac.actor, use_compile)
        ac.critic = _maybe_compile(ac.critic, use_compile)
        self.net = ac  # for checkpointing in train.py

        # Optimizer
        lr = float(cfg.get("lr", 3e-4))
        opt = torch.optim.AdamW(ac.parameters(), lr=lr)

        # PPO knobs
        gamma = float(cfg.get("gamma", 0.99))
        lam = float(cfg.get("gae_lambda", 0.95))
        clip_coef = float(cfg.get("clip_coef", 0.2))
        vf_clip_coef = float(cfg.get("vf_clip_coef", 0.2))
        vf_coef = float(cfg.get("vf_coef", 0.5))
        ent_coef = float(cfg.get("ent_coef", 0.01))
        max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        n_epochs = int(cfg.get("n_epochs", 10))
        minibatch_size = int(cfg.get("minibatch_size", 2048))
        target_kl = float(cfg.get("target_kl", 0.015))
        lr_anneal = str(cfg.get("lr_anneal", ""))

        reward_scale = float(cfg.get("reward_scale", 1.0))
        use_obs_norm = bool(cfg.get("obs_norm", False))
        adv_norm = bool(cfg.get("adv_norm", True))
        use_value_clip = bool(cfg.get("use_value_clip", True))

        # Buffers
        if isinstance(act_space, Box):
            act_shape, act_dtype = (self.act_dim,), torch.float32
        else:
            act_shape, act_dtype = (), torch.long

        obs_buf = torch.zeros((rollout_steps, num_envs) + self.obs_shape, dtype=torch.float32, device=self.device)
        act_buf = torch.zeros((rollout_steps, num_envs) + act_shape, dtype=act_dtype, device=self.device)
        logp_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=self.device)
        rew_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=self.device)
        done_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=self.device)
        val_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=self.device)

        obs_norm = RunningNorm(self.obs_shape, device=self.device) if use_obs_norm else None

        # Reset
        next_obs, _ = venv.reset(seed=seed)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        ep_returns, ep_lengths = [], []
        ep_ret_env = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        ep_len_env = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        start = time.time()
        gstep = 0

        while gstep < total_timesteps:
            # ===== collect =====
            for t in range(rollout_steps):
                gstep += num_envs
                flat_obs = next_obs.view(num_envs, -1)
                if obs_norm is not None:
                    obs_norm.update(flat_obs)
                    flat_obs = obs_norm.normalize(flat_obs)

                with torch.no_grad():
                    v = ac.critic(flat_obs)
                    if ac.is_cont:
                        a, logp = ac.actor.sample(flat_obs)
                        env_a = ac.scale_action(a).cpu().numpy()
                    else:
                        a, logp = ac.actor.sample(flat_obs)
                        env_a = a.cpu().numpy()

                o2, r, terminated, truncated, _ = venv.step(env_a)
                done = np.logical_or(terminated, truncated)

                if reward_scale != 1.0:
                    r = np.asarray(r, dtype=np.float32) * reward_scale

                ep_ret_env += torch.as_tensor(r, device=self.device, dtype=torch.float32)
                ep_len_env += 1
                for i in range(num_envs):
                    if done[i]:
                        ep_returns.append(float(ep_ret_env[i].item()))
                        ep_lengths.append(int(ep_len_env[i].item()))
                        ep_ret_env[i] = 0.0
                        ep_len_env[i] = 0.0

                obs_buf[t] = flat_obs
                act_buf[t] = a
                logp_buf[t] = logp
                val_buf[t] = v
                rew_buf[t] = torch.as_tensor(r, device=self.device, dtype=torch.float32)
                done_buf[t] = next_done

                next_obs = torch.as_tensor(o2, dtype=torch.float32, device=self.device)
                next_done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

            # log
            fps = int(gstep / max(time.time() - start, 1e-6))
            disp_ret = (np.mean(ep_returns[-10:]) if len(ep_returns) else float("nan"))
            disp_len = (np.mean(ep_lengths[-10:]) if len(ep_lengths) else float("nan"))
            print(f"step {gstep:8d} | ep_ret {disp_ret:8.1f} | ep_len {disp_len:6.1f} | fps {fps}")


            # 2. ADD THE TENSORBOARD LOGGING LOGIC HERE
            if writer: # Only log if a writer was provided
                writer.add_scalar("charts/episodic_return", disp_ret, gstep)
                writer.add_scalar("charts/episodic_length", disp_len, gstep)
                writer.add_scalar("charts/fps", fps, gstep)
                # Also log the learning rate to visualize the annealing
                if lr_anneal:
                    writer.add_scalar("charts/learning_rate", opt.param_groups[0]["lr"], gstep)

                    
            # ===== bootstrap & GAE =====
            with torch.no_grad():
                flat_obs = next_obs.view(num_envs, -1)
                if obs_norm is not None:
                    flat_obs = obs_norm.normalize(flat_obs)
                next_v = ac.critic(flat_obs)

            adv = torch.zeros_like(rew_buf, device=self.device)
            lastgaelam = torch.zeros(num_envs, device=self.device)
            for t in reversed(range(rollout_steps)):
                nextnonterm = 1.0 - done_buf[t]
                nv = next_v if t == rollout_steps - 1 else val_buf[t + 1]
                delta = rew_buf[t] + gamma * nv * nextnonterm - val_buf[t]
                lastgaelam = delta + gamma * lam * nextnonterm * lastgaelam
                adv[t] = lastgaelam
            ret = adv + val_buf

            # flatten
            b_obs = obs_buf.reshape(-1, obs_buf.shape[-1])
            b_act = act_buf.reshape(-1, *act_buf.shape[2:])
            b_oldlog = logp_buf.reshape(-1)
            b_adv = adv.reshape(-1)
            b_ret = ret.reshape(-1)
            b_val = val_buf.reshape(-1)

            if adv_norm:
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            # LR anneal
            if lr_anneal == "linear":
                frac = 1.0 - (gstep / float(total_timesteps))
                for g in opt.param_groups:
                    g["lr"] = lr * frac

            # ===== updates =====
            idx = np.arange(b_obs.shape[0])
            for _ in range(n_epochs):
                np.random.shuffle(idx)
                stop = False
                for start_i in range(0, len(idx), minibatch_size):
                    mb = idx[start_i:start_i + minibatch_size]
                    mb_obs, mb_act = b_obs[mb], b_act[mb]
                    mb_old = b_oldlog[mb]
                    mb_adv, mb_ret, mb_val = b_adv[mb], b_ret[mb], b_val[mb]

                    if ac.is_cont:
                        newlog = ac.actor.log_prob(mb_obs, mb_act)
                    else:
                        newlog = ac.actor.log_prob(mb_obs, mb_act)

                    ratio = torch.exp(newlog - mb_old)
                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    newv = ac.critic(mb_obs)
                    if use_value_clip:
                        v_clip = mb_val + (newv - mb_val).clamp(-vf_clip_coef, vf_clip_coef)
                        v_uncl = (newv - mb_ret).pow(2)
                        v_clipd = (v_clip - mb_ret).pow(2)
                        v_loss = 0.5 * torch.max(v_uncl, v_clipd).mean()
                    else:
                        v_loss = 0.5 * (newv - mb_ret).pow(2).mean()

                    if ac.is_cont:
                        _, std = ac.actor(mb_obs)
                        entropy = (0.5 * (1.0 + math.log(2 * math.pi)) + torch.log(std)).sum(-1).mean()
                    else:
                        logits = ac.actor(mb_obs)
                        dist = torch.distributions.Categorical(logits=logits)
                        entropy = dist.entropy().mean()

                    loss = pg_loss + v_loss * vf_coef - ent_coef * entropy
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                    opt.step()

                    with torch.no_grad():
                        # approx_kl = E[logpi_new - logpi_old] ≈ (log_ratio - (ratio-1))
                        logratio = newlog - mb_old
                        approx_kl = (logratio.exp() - 1 - logratio).mean().abs()
                    if approx_kl > target_kl:
                        stop = True
                        break
                if stop:
                    break

        mean_return = float(np.mean(ep_returns[-100:])) if ep_returns else float("nan")

        # checkpoint (also done by train.py, but fine to save here too)
        os.makedirs("checkpoints", exist_ok=True)
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
            "checkpoints/ppo_final.pt",
        )
        print(f"Training finished: {{'steps': {gstep}, 'mean_return': {mean_return}}}")
        print("Saved checkpoint to checkpoints/ppo_final.pt")
        return {"steps": gstep, "mean_return": mean_return}

# ----------------------------
# Helpers
# ----------------------------
def _make_env(env_id: str, seed: int, idx: int):
    def thunk():
        env = gym.make(env_id)
        env.reset(seed=seed + idx)
        return env
    return thunk
