# train.py  — unified entrypoint for discrete & continuous PPO
import argparse, yaml, torch
from rl.utils.envs import make_vec_env
from rl.ppo.ppo import PPOConfig, PPOAgent

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    # ---- legacy → new field names (so your old configs still run) ----
    if "update_epochs" in d and "n_epochs" not in d:
        d["n_epochs"] = int(d.pop("update_epochs"))
    if "learning_rate" in d and "lr" not in d:
        d["lr"] = float(d.pop("learning_rate"))
    # sensible defaults if missing (for old configs)
    d.setdefault("vector_env", "sync")
    d.setdefault("use_amp", False)
    d.setdefault("use_compile", False)
    return d

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "--cfg", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg_dict = load_cfg(args.config)
    cfg = PPOConfig(**cfg_dict)

    # (Optional) slightly faster matmul on CUDA
    if cfg.device in ("auto", "cuda") and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    envs = make_vec_env(cfg.env_id, cfg.num_envs, cfg.seed, vector_type=cfg.vector_env)
    agent = PPOAgent(envs, cfg)
    agent.train()
