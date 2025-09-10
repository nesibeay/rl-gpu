# eval.py — deterministic evaluation using saved checkpoint
import argparse, torch, numpy as np, gymnasium as gym
from rl.ppo.ppo import select_device, ActorCritic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_final.pt")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    env = gym.make(args.env_id, render_mode=None)

    obs_space, act_space = env.observation_space, env.action_space
    assert len(obs_space.shape) == 1, "This eval script assumes vector observations."

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})  # contains actor_hidden_sizes / critic_hidden_sizes, etc.

    # Build the same network class used during training
    net = ActorCritic(obs_space, act_space, cfg).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # Helpers
    def obs_to_tensor(o):
        x = torch.as_tensor(o, dtype=torch.float32, device=device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        steps = 0
        while not done:
            with torch.no_grad():
                x = obs_to_tensor(obs)
                if net.is_cont:
                    mean, _ = net.actor(x)
                    a_tanh = torch.tanh(mean)            # deterministic: use mean -> tanh
                    a_env = net.scale_action(a_tanh)
                    act = a_env.squeeze(0).cpu().numpy()
                else:
                    logits = net.actor(x)
                    act = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(act)
            done = bool(terminated or truncated)
            ret += float(reward)
            steps += 1
        returns.append(ret)
        print(f"Episode {ep+1}: return={ret:.2f}, steps={steps}")

    print(f"Average return over {args.episodes} episodes: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")

if __name__ == "__main__":
    main()
