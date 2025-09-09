# train.py — unified entrypoint for PPO (supports continuous + discrete)
import argparse, yaml, torch, numpy as np, random, dataclasses

from rl.ppo.ppo import PPO, PPOConfig


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

    # Filter out unknown YAML keys so PPOConfig(**...) never crashes
    valid_keys = {f.name for f in dataclasses.fields(PPOConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    extras = {k: v for k, v in cfg_dict.items() if k not in valid_keys}
    if extras:
        print("[train] Ignoring extra config keys:", sorted(extras.keys()))

    cfg = PPOConfig(**filtered)

    # Optional: honor float32_matmul_precision from YAML (useful on NVIDIA GPUs)
    prec = getattr(cfg, "float32_matmul_precision", None)
    if prec:
        try:
            torch.set_float32_matmul_precision(str(prec))
            print("Set torch.float32 matmul precision ->", prec)
        except Exception as e:
            print("Could not set float32 matmul precision:", e)

    set_seed(cfg.seed)

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
