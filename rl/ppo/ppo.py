# File: rl/ppo/ppo.py (unified and corrected)
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, Optional


import numpy as np
import torch
import torch.nn as nn
import time
import os

# Assuming your ActorCritic is in this path
from rl.policy.networks import ActorCritic
from rl.utils.logger import CSVLogger


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
        
        # --- Device Selection Logic (FIXED) ---
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # This precision setting is specific to CUDA, so it belongs inside this block
                torch.set_float32_matmul_precision(self.cfg.float32_matmul_precision)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            # User has specified a device manually
            self.device = torch.device(self.cfg.device)

        print(f"INFO: Using device: {self.device}")

        # Policy
        obs_space = envs.single_observation_space
        action_space = envs.single_action_space
        self.policy = ActorCritic(obs_space, action_space).to(self.device)
        
        # Allow use_compile to be True/False or a mode string like "max-autotune"
        cmode = None
        if isinstance(self.cfg.use_compile, str) and self.cfg.use_compile:
            cmode = self.cfg.use_compile
        elif isinstance(self.cfg.use_compile, bool) and self.cfg.use_compile:
            cmode = "max-autotune"
        if cmode and hasattr(torch, "compile"):
            print("INFO: Compiling the policy...")
            self.policy = torch.compile(self.policy, mode=cmode)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr, eps=1e-5)
        from torch.amp import GradScaler as AmpGradScaler
        self.scaler = AmpGradScaler("cuda", enabled=(self.device.type == "cuda" and self.cfg.use_amp))

        # Storage
        self.rollout_steps = self.cfg.rollout_steps
        self.num_envs = self.cfg.num_envs
        self.obs = None
        self.global_step = 0

        # For episodic return tracking
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_returns = []
        self.completed_lengths = []

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            raise NotImplementedError("Dict observations not supported.")
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.ndim == 4 and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2)
        return x

    def collect_rollout(self) -> Tuple[torch.Tensor, ...]:
        obs_shape = self.envs.single_observation_space.shape
        action_shape = self.envs.single_action_space.shape

        obs_buf = torch.zeros((self.rollout_steps, self.num_envs) + obs_shape, dtype=torch.float32)
        actions_buf = torch.zeros((self.rollout_steps, self.num_envs) + action_shape, dtype=torch.float32)
        logprobs_buf = torch.zeros(self.rollout_steps, self.num_envs, dtype=torch.float32)
        rewards_buf  = torch.zeros(self.rollout_steps, self.num_envs, dtype=torch.float32)
        dones_buf    = torch.zeros(self.rollout_steps, self.num_envs, dtype=torch.float32)
        values_buf   = torch.zeros(self.rollout_steps, self.num_envs, dtype=torch.float32)

        if self.obs is None:
            obs, _ = self.envs.reset(seed=self.cfg.seed)
            self.obs = obs

        use_amp = (self.device.type == "cuda" and self.cfg.use_amp)
        from torch.amp import autocast as amp_autocast

        for t in range(self.rollout_steps):
            obs_buf[t] = torch.from_numpy(self.obs)
            obs_t = self._obs_to_tensor(self.obs)

            with torch.no_grad(), amp_autocast("cuda", enabled=use_amp):
                action, logprob, _, value = self.policy.get_action_and_value(obs_t)

            values_buf[t]  = value.squeeze(-1).detach().cpu()
            logprobs_buf[t] = logprob.detach().cpu()
            actions_buf[t] = action.detach().cpu()
            
            act_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.envs.step(act_np)
            done = np.logical_or(terminated, truncated)

            rewards_buf[t] = torch.tensor(reward, dtype=torch.float32)
            dones_buf[t]   = torch.tensor(done,   dtype=torch.float32)

            # Episode tracking
            self.episode_returns += reward
            self.episode_lengths += 1
            for i, d in enumerate(done):
                if d:
                    self.completed_returns.append(float(self.episode_returns[i]))
                    self.completed_lengths.append(int(self.episode_lengths[i]))
                    self.episode_returns[i] = 0.0
                    self.episode_lengths[i] = 0

            self.obs = next_obs

        with torch.no_grad(), amp_autocast("cuda", enabled=use_amp):
            next_obs_t = self._obs_to_tensor(self.obs)
            next_value = self.policy.get_value(next_obs_t).squeeze(-1).detach().cpu()

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
        from torch.amp import autocast as amp_autocast

        b_obs = obs_buf.view((self.rollout_steps * self.num_envs,) + self.envs.single_observation_space.shape)
        b_actions = actions.view((self.rollout_steps * self.num_envs,) + self.envs.single_action_space.shape)
        b_logprobs = old_logprobs.view(-1)
        b_adv = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = values.view(-1)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        batch_size = b_obs.shape[0]
        minibatch_size = self.cfg.minibatch_size or batch_size
        
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=False)
        use_amp = (self.device.type == "cuda" and self.cfg.use_amp)

        for _ in range(self.cfg.n_epochs):
            for mb_inds in sampler:
                mb_obs = self._obs_to_tensor(b_obs[mb_inds].numpy())
                mb_actions = b_actions[mb_inds].to(self.device)

                with amp_autocast("cuda", enabled=use_amp):
                    _, new_logprob, entropy, new_value = self.policy.get_action_and_value(mb_obs, mb_actions)
                    
                    logratio = new_logprob - b_logprobs[mb_inds].to(self.device)
                    ratio = logratio.exp()
                    
                    adv = b_adv[mb_inds].to(self.device)
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((new_value.view(-1) - b_returns[mb_inds].to(self.device)) ** 2).mean()

                    loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * entropy.mean()

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
        run_name = f"{self.cfg.env_id.lower()}_{int(time.time())}"
        if self.cfg.resume_from:
            self.load_checkpoint(self.cfg.resume_from)
            run_name = os.path.splitext(os.path.basename(self.cfg.resume_from))[0] + "_resumed"

        logger = CSVLogger(path=f"runs/{run_name}.csv")

        steps_per_rollout = self.cfg.rollout_steps * self.cfg.num_envs
        num_updates = self.cfg.total_timesteps // steps_per_rollout

        try:
            last_t = time.perf_counter()
            for u in range(1, num_updates + 1):
                if self.cfg.anneal_lr:
                    frac = 1.0 - ((u - 1) / num_updates)
                    lrnow = self.cfg.lr * frac
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lrnow

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
                else: mean_ret, mean_len = 0.0, 0.0

                eval_ret = None
                if self.cfg.eval_every_updates and (u % self.cfg.eval_every_updates == 0):
                    eval_ret = self.evaluate(self.cfg.eval_episodes)

                print(f"update {u}/{num_updates} | step {self.global_step} | "
                      f"return {mean_ret:.1f} | ep_len {mean_len:.0f} | fps {fps:.0f}"
                      + (f" | eval {eval_ret:.1f}" if eval_ret is not None else ""))
                logger.write(u, self.global_step, mean_ret, mean_len, fps, eval_ret)

                if self.cfg.save_every_updates and (u % self.cfg.save_every_updates == 0):
                    ckpt_path = f"checkpoints/{run_name}_u{u}.pt"
                    self.save_checkpoint(ckpt_path)
        finally:
            logger.close()

    def save_checkpoint(self, path: str):
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
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = int(ckpt.get("global_step", 0))
        if ckpt.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt["scaler"])
        self.completed_returns = list(ckpt.get("completed_returns", []))
        self.completed_lengths = list(ckpt.get("completed_lengths", []))

    def evaluate(self, episodes: int) -> float:
        import gymnasium as gym
        env = gym.make(self.cfg.env_id)
        total = 0.0
        for ep in range(episodes):
            obs, _ = env.reset(seed=self.cfg.seed + ep)
            done = False
            ep_ret = 0.0
            while not done:
                x = self._obs_to_tensor(obs[None, ...])
                with torch.no_grad():
                    act, _, _, _ = self.policy.get_action_and_value(x, deterministic=self.cfg.deterministic_eval)
                
                low, high = env.action_space.low, env.action_space.high
                act_np = act.detach().cpu().numpy()[0]
                act_np = np.clip(act_np, low, high)
                
                obs, r, terminated, truncated, _ = env.step(act_np)
                done = terminated or truncated
                ep_ret += float(r)
            total += ep_ret
        env.close()
        return total / float(episodes)
        