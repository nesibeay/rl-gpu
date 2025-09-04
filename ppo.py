# File: rl/ppo/ppo.py (unified PPO)
new_value = self.policy.value(feats).squeeze(-1)


logratio = new_logprob - b_logprobs[mb_inds]
ratio = logratio.exp()


# Policy loss
adv = b_adv[mb_inds]
pg_loss1 = -adv * ratio
pg_loss2 = -adv * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()


# Value loss (clipped)
v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
v_clipped = b_values[mb_inds] + torch.clamp(new_value - b_values[mb_inds], -self.cfg.clip_coef, self.cfg.clip_coef)
v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()


loss = pg_loss + self.cfg.vf_coef * v_loss
# (optional) add entropy bonus: requires computing dist.entropy in act/log_prob API


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
import gymnasium as gym
# Init obs
obs, _ = self.envs.reset(seed=self.cfg.seed)
self.obs = obs


self.global_step = 0
steps_per_rollout = self.cfg.rollout_steps * self.cfg.num_envs
num_updates = self.cfg.total_timesteps // steps_per_rollout


for update in range(num_updates):
obs_buf, actions_buf, logprobs_buf, rewards_buf, dones_buf, values_buf, next_value = self.collect_rollout()


# Build actions tensor now that we know discrete/continuous
if self.policy.spec.action_discrete:
# we didn't store actions; we can recompute by sampling again would be wrong.
# Instead, adjust: store actions during rollout.
pass
# To fix: We will refactor collect_rollout to store actions. See below patch.
raise NotImplementedError("collect_rollout must store actions; update the function per below.")

