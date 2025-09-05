# train.py — unified entrypoint for PPO (supports continuous + discrete)
import argparse, yaml, torch, numpy as np, random

from rl.ppo.ppo import PPO, PPOConfig
import gymnasium as gym


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo_pendulum.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = PPOConfig(**cfg_dict)
    # Optional: honor float32_matmul_precision from YAML (useful on NVIDIA GPUs)
    prec = getattr(cfg, "float32_matmul_precision", None)
    if prec:
        try:
            import torch
            torch.set_float32_matmul_precision(str(prec))
            print("Set torch.float32 matmul precision ->", prec)
        except Exception as e:
            print("Could not set float32 matmul precision:", e)

    set_seed(cfg.seed)

    # We now support both continuous and discrete envs; no early exit.
    # (train will auto-detect the action space and build the right head.)
    agent = PPO(cfg)
    stats = agent.train()

    # Save checkpoint
    import os
    os.makedirs('checkpoints', exist_ok=True)
    ckpt = {
        'model_state_dict': agent.net.state_dict(),
        'config': cfg.__dict__,
        'env_id': cfg.env_id,
        'mode': getattr(agent, 'mode', 'unknown'),
        'obs_shape': agent.obs_shape,
        'act_dim': getattr(agent, 'act_dim', None),
        'n_actions': getattr(agent, 'n_actions', None),
    }
    torch.save(ckpt, 'checkpoints/ppo_final.pt')
    print("Training finished:", stats)
    print("Saved checkpoint to checkpoints/ppo_final.pt")


if __name__ == "__main__":
    main()
