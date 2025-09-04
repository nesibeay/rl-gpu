# File: rl/ppo/ppo.py (unified PPO)
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler


from rl.policy.networks import ActorCritic

@dataclass
class PPOConfig:
    env_id: str = "Pendulum-v1"
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
    device: str = "auto" # auto|cpu|cuda|mps
    vector_env: str = "sync" # sync|async
    use_amp: bool = False
    use_compile: bool | str = False 
    float32_matmul_precision: str = "high"
    #add checkpoints + periodic evaluation
    save_every_updates: int = 0
    resume_from: Optional[str] = None
    eval_every_updates: int = 20
    eval_episodes: int = 5
    deterministic_eval: bool = True


class PPOAgent:
    def __init__(self, envs, config: PPOConfig):
        import gymnasium as gym
        self.envs = envs
        self.cfg = config
        # Device
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device(self.cfg.device)
            # Precision hint for matmul kernels (PyTorch 2.x)
            torch.set_float32_matmul_precision(self.cfg.float32_matmul_precision)

            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.cfg.device)


        # Policy
        obs_space = envs.single_observation_space
        action_space = envs.single_action_space
        self.policy = ActorCritic(obs_space, action_space, hidden=(64, 64)).to(self.device)
        # Allow use_compile to be True/False or a mode string like "max-autotune"
        cmode = None
        if isinstance(self.cfg.use_compile, str) and self.cfg.use_compile:
            cmode = self.cfg.use_compile
        elif isinstance(self.cfg.use_compile, bool) and self.cfg.use_compile:
            cmode = "max-autotune"
        if cmode and hasattr(torch, "compile"):
            self.policy = torch.compile(self.policy, mode=cmode)  # type: ignore[attr-defined]

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr, eps=1e-5)
        from torch.amp import GradScaler as AmpGradScaler
        self.scaler = AmpGradScaler("cuda", enabled=(self.device.type == "cuda" and self.cfg.use_amp))


        # Storage
        self.rollout_steps = self.cfg.rollout_steps
        self.num_envs = self.cfg.num_envs
        self.obs = None


        # For episodic return tracking
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_returns = []
        self.completed_lengths = []

    def _obs_to_tensor(self, obs):
        import gymnasium as gym
        if isinstance(obs, dict):
            # not handled in this minimal version
            raise NotImplementedError("Dict observations not supported in this patch.")
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # If image HWC, permute to NCHW
        if x.ndim == 4 and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2)
        return x


    
    def collect_rollout(self):
        import numpy as np
        import torch

        obs_shape = self.envs.single_observation_space.shape
        obs_buf = torch.zeros((self.rollout_steps, self.num_envs) + obs_shape, dtype=torch.float32)

        if self.policy.spec.action_discrete:
            actions_buf = torch.zeros(self.rollout_steps, self.num_envs, dtype=torch.long)
        else:
            actions_buf = torch.zeros(self.rollout_steps, self.num_envs, self.policy.spec.action_dim, dtype=torch.float32)

        logprobs_buf = torch.zeros(self.rollout_steps, self.num_envs, dtype=torch.float32)
        rewards_buf  = torch.zeros(self.rollout_steps, self.num_envs, dtype=torch.float32)
        dones_buf    = torch.zeros(self.rollout_steps, self.num_envs, dtype=torch.float32)
        values_buf   = torch.zeros(self.rollout_steps, self.num_envs, dtype=torch.float32)

        if self.obs is None:
            obs, _ = self.envs.reset(seed=self.cfg.seed)
            self.obs = obs

        use_amp = (self.device.type == "cuda" and self.cfg.use_amp)
        from torch.amp import autocast as amp_autocast  # modern API

        for t in range(self.rollout_steps):
            obs_buf[t] = torch.from_numpy(self.obs)
            obs_t = self._obs_to_tensor(self.obs)

            with torch.no_grad(), amp_autocast("cuda", enabled=use_amp):
                feats   = self.policy.features(obs_t)
                values  = self.policy.value(feats).squeeze(-1)
                action, logprob, _ = self.policy.act(feats, deterministic=False)

            values_buf[t]  = values.detach().cpu()
            logprobs_buf[t] = logprob.detach().cpu()

            if self.policy.spec.action_discrete:
                act_np = action.detach().cpu().numpy()
                actions_buf[t] = action.detach().cpu()
            else:
                low  = self.envs.single_action_space.low
                high = self.envs.single_action_space.high
                act_np = action.detach().cpu().numpy()
                act_np = np.clip(act_np, low, high)
                actions_buf[t] = torch.from_numpy(act_np)

            next_obs, reward, terminated, truncated, info = self.envs.step(act_np)
            done = np.logical_or(terminated, truncated)

            rewards_buf[t] = torch.tensor(reward, dtype=torch.float32)
            dones_buf[t]   = torch.tensor(done,   dtype=torch.float32)

            # episode tracking
            self.episode_returns += reward
            self.episode_lengths += 1
            for i, d in enumerate(done):
                if d:
                    self.completed_returns.append(float(self.episode_returns[i]))
                    self.completed_lengths.append(int(self.episode_lengths[i]))
                    self.episode_returns[i] = 0.0
                    self.episode_lengths[i] = 0

            self.obs = next_obs

        with torch.no_grad():
            obs_t = self._obs_to_tensor(self.obs)
            feats = self.policy.features(obs_t)
            next_value = self.policy.value(feats).squeeze(-1).detach().cpu()

        return (obs_buf, actions_buf, logprobs_buf, rewards_buf, dones_buf, values_buf, next_value)


    def compute_gae(self, rewards, dones, values, next_value):
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(self.num_envs)
        for t in reversed(range(self.rollout_steps)):
            next_nonterminal = 1.0 - dones[t]
            next_values = next_value if t == self.rollout_steps - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * next_values * next_nonterminal - values[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns



    def update(self, obs_buf, actions, old_logprobs, advantages, returns, values):
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.amp import autocast as amp_autocast

        # -------- Flatten rollout tensors into a single batch --------
        b_obs = obs_buf.view(self.rollout_steps * self.num_envs, *self.envs.single_observation_space.shape)
        b_obs = self._obs_to_tensor(b_obs.cpu().numpy())  # to device, float32

        if self.policy.spec.action_discrete:
            b_actions = actions.view(-1)
        else:
            b_actions = actions.view(self.rollout_steps * self.num_envs, -1)

        b_logprobs = old_logprobs.view(-1)
        b_adv      = advantages.view(-1)
        b_returns  = returns.view(-1)
        b_values   = values.view(-1)

        # -------- Normalize advantages --------
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # -------- Optional LR anneal --------
        if self.cfg.anneal_lr:
            frac = 1.0 - (self.global_step / float(self.cfg.total_timesteps))
            lrnow = self.cfg.lr * frac
            for pg in self.optimizer.param_groups:
                pg["lr"] = lrnow

        batch_size = b_obs.shape[0]
        minibatch_size = self.cfg.minibatch_size
        inds = np.arange(batch_size)

        use_amp = (self.device.type == "cuda" and self.cfg.use_amp)

        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                with amp_autocast("cuda", enabled=use_amp):
                    # --- forward ---
                    feats = self.policy.features(b_obs[mb_inds])
                    new_logprob = self.policy.log_prob(feats, b_actions[mb_inds])
                    new_value = self.policy.value(feats).squeeze(-1)

                    # --- entropy (exploration) ---
                    if self.policy.spec.action_discrete:
                        logits = self.policy.actor(feats)
                        dist = torch.distributions.Categorical(logits=logits)
                        entropy = dist.entropy().mean()
                    else:
                        mean = self.policy.actor_mean(feats)
                        std  = self.policy.log_std.exp().expand_as(mean)
                        dist = torch.distributions.Normal(mean, std)
                        entropy = dist.entropy().sum(-1).mean()

                    # --- PPO clipped policy loss ---
                    logratio = new_logprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    adv = b_adv[mb_inds]
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # --- Value loss (clipped) ---
                    v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_value - b_values[mb_inds],
                        -self.cfg.clip_coef, self.cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # --- Final loss: policy + value - entropy ---
                    loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()


    def train(self):
        import time, numpy as np, os
        from rl.utils.logger import CSVLogger

        # optional resume
        run_name = f"{self.cfg.env_id.lower()}_{int(time.time())}"
        if self.cfg.resume_from:
            self.load_checkpoint(self.cfg.resume_from)
            run_name = os.path.splitext(os.path.basename(self.cfg.resume_from))[0] + "_resumed"

        obs, _ = self.envs.reset(seed=self.cfg.seed)
        self.obs = obs

        logger = CSVLogger(path=f"runs/{run_name}.csv")

        self.global_step = getattr(self, "global_step", 0)
        steps_per_rollout = self.cfg.rollout_steps * self.cfg.num_envs
        # if resuming, continue until total_timesteps regardless of previous steps
        remaining_steps = max(self.cfg.total_timesteps - self.global_step, 0)
        num_updates = remaining_steps // steps_per_rollout

        try:
            last_t = time.perf_counter()
            for u in range(num_updates):
                (obs_buf, actions_buf, logprobs_buf,
                 rewards_buf, dones_buf, values_buf, next_value) = self.collect_rollout()

                advantages, returns = self.compute_gae(rewards_buf, dones_buf, values_buf, next_value)
                self.update(obs_buf, actions_buf, logprobs_buf, advantages, returns, values_buf)
                self.global_step += steps_per_rollout

                now = time.perf_counter()
                fps = steps_per_rollout / max(now - last_t, 1e-8)
                last_t = now

                if len(self.completed_returns) > 0:
                    mean_ret = float(np.mean(self.completed_returns[-10:]))
                    mean_len = float(np.mean(self.completed_lengths[-10:]))
                else:
                    mean_ret = 0.0
                    mean_len = 0.0

                eval_ret = None
                if self.cfg.eval_every_updates and ((u + 1) % self.cfg.eval_every_updates == 0):
                    eval_ret = self.evaluate(self.cfg.eval_episodes)

                print(f"update {u+1}/{num_updates} | step {self.global_step} | "
                      f"return {mean_ret:.1f} | ep_len {mean_len:.0f} | fps {fps:.0f}"
                      + (f" | eval {eval_ret:.1f}" if eval_ret is not None else ""))
                logger.write(u+1, self.global_step, mean_ret, mean_len, fps, eval_ret)

                if self.cfg.save_every_updates and ((u + 1) % self.cfg.save_every_updates == 0):
                    ckpt_path = f"checkpoints/{run_name}_u{u+1}.pt"
                    self.save_checkpoint(ckpt_path)
        finally:
            logger.close()


    def save_checkpoint(self, path: str):
        import os, torch
        from dataclasses import asdict
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "completed_returns": self.completed_returns,
            "completed_lengths": self.completed_lengths,
            "cfg": asdict(self.cfg),
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        import torch
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = int(ckpt.get("global_step", 0))
        if ckpt.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt["scaler"])
        self.completed_returns = list(ckpt.get("completed_returns", []))
        self.completed_lengths = list(ckpt.get("completed_lengths", []))

    def evaluate(self, episodes: int) -> float:
        import numpy as np, gymnasium as gym, torch
        env = gym.make(self.cfg.env_id)
        total = 0.0
        for ep in range(episodes):
            obs, _ = env.reset(seed=self.cfg.seed + ep)
            done = False
            ep_ret = 0.0
            while not done:
                x = self._obs_to_tensor(obs[None, ...])
                with torch.no_grad():
                    feats = self.policy.features(x)
                    act, _, _ = self.policy.act(feats, deterministic=self.cfg.deterministic_eval)
                if self.policy.spec.action_discrete:
                    act_np = int(act.item())
                else:
                    low, high = env.action_space.low, env.action_space.high
                    act_np = act.detach().cpu().numpy()[0]
                    act_np = np.clip(act_np, low, high)
                obs, r, terminated, truncated, _ = env.step(act_np)
                done = terminated or truncated
                ep_ret += float(r)
            total += ep_ret
        env.close()
        return total / float(episodes)
