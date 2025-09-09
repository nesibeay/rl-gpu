# train.py — unified entrypoint for PPO (supports continuous + discrete)
import argparse, yaml, torch, numpy as np, random, dataclasses, os

from rl.ppo.ppo import PPO, PPOConfig  # keeps your original import style

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

    # Read YAML as UTF-8 (Windows-safe)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    # Expand the whitelist by using PPOConfig dataclass fields
    valid_keys = {f.name for f in dataclasses.fields(PPOConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    extras = {k: v for k, v in cfg_dict.items() if k not in valid_keys}
    if extras:
        # keep, but they won’t block training; we store them in PPOConfig.extra
        print("[train] Ignoring extra config keys:", sorted(extras.keys()))
        filtered["extra"] = extras

    cfg = PPOConfig(**filtered)

    # Honor float32 matmul precision if given (CUDA only)
    if cfg.float32_matmul_precision:
        try:
            torch.set_float32_matmul_precision(str(cfg.float32_matmul_precision))
            print("Set torch.float32 matmul precision ->", cfg.float32_matmul_precision)
        except Exception as e:
            print("Could not set float32 matmul precision:", e)

    set_seed(cfg.seed)

    agent = PPO(cfg)
    stats = agent.train()

    # Save checkpoint again in the format you had
    os.makedirs('checkpoints', exist_ok=True)
    ckpt = {
        'model_state_dict': agent.net.state_dict() if agent.net is not None else None,
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
