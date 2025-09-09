# eval.py — deterministic evaluation using saved checkpoint (continuous + discrete)
import argparse, torch, numpy as np, gymnasium as gym
from gymnasium.spaces import Box, Discrete
from rl.ppo.ppo import select_device
from rl.policy.networks import ActorCriticContinuous, ActorCriticDiscrete


def scale_action(a_tanh, low, high):
    return low + (a_tanh + 1.0) * 0.5 * (high - low)


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

    obs_dim = int(np.prod(obs_space.shape))

    ckpt = torch.load(args.checkpoint, map_location=device)
    hidden = tuple(ckpt.get('config', {}).get('hidden_sizes', (64, 64)))

    if isinstance(act_space, Box):
        act_dim = int(np.prod(act_space.shape))
        net = ActorCriticContinuous(obs_dim, act_dim, hidden=hidden).to(device)
    elif isinstance(act_space, Discrete):
        n_actions = int(act_space.n)
        net = ActorCriticDiscrete(obs_dim, n_actions, hidden=hidden).to(device)
    else:
        raise ValueError("Unsupported action space for eval.")

    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()

    low = torch.as_tensor(getattr(act_space, 'low', None), dtype=torch.float32, device=device) if isinstance(act_space, Box) else None
    high = torch.as_tensor(getattr(act_space, 'high', None), dtype=torch.float32, device=device) if isinstance(act_space, Box) else None

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
                if isinstance(act_space, Box):
                    mean, _ = net.dist_params(x)
                    a_tanh = torch.tanh(mean)
                    a_env = scale_action(a_tanh, low, high)
                    act = a_env.squeeze(0).cpu().numpy()
                else:
                    logits, _ = net.forward(x)
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
