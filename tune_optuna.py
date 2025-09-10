
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import optuna
import yaml

# --- Utilities ---

MEAN_RET_RE = re.compile(r"Training finished: \{'steps':\s*(\d+),\s*'mean_return':\s*([-\d\.naninf]+)\}")

def run_train_with_config(config_path: Path) -> float:
    """
    Runs `python train.py --config <config_path>` and returns the parsed mean_return.
    Assumes train.py prints: Training finished: {'steps': X, 'mean_return': Y}
    """
    cmd = [sys.executable, "train.py", "--config", str(config_path)]
    print(f"[tune] Launch: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

    mean_return = None
    # stream output live (useful for debugging and TB paths)
    for line in proc.stdout:
        print(line, end="")
        m = MEAN_RET_RE.search(line)
        if m:
            # steps = int(m.group(1))
            try:
                mean_return = float(m.group(2))
            except:
                mean_return = None

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Training process exited with code {proc.returncode}")

    if mean_return is None:
        raise RuntimeError("Could not parse mean_return from train.py output. "
                           "Ensure train.py prints a line like: "
                           "Training finished: {'steps': 123456, 'mean_return': -250.3}")
    return mean_return

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

# --- Search Space ---

def sample_params(trial: optuna.Trial) -> dict:
    """
    Defines the hyperparameter search space.
    Keep it modest first; you can expand once flow is proven.
    """
    params = {
        # core PPO
        "lr": trial.suggest_loguniform("lr", 1e-5, 5e-3),
        "clip_coef": trial.suggest_float("clip_coef", 0.1, 0.3),
        "ent_coef": trial.suggest_loguniform("ent_coef", 1e-4, 2e-2),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.9),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 1.5),

        # credit assignment & discount
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.98),

        # rollout / optimization
        "rollout_steps": trial.suggest_categorical("rollout_steps", [512, 768, 1024]),
        "n_epochs": trial.suggest_int("n_epochs", 10, 25),
        "minibatch_size": trial.suggest_categorical("minibatch_size", [2048, 3072, 4096]),

        # policy noise
        "init_log_std": trial.suggest_float("init_log_std", -1.5, -0.3),

        # stabilizers
        "adv_norm": True,
        "use_value_clip": True,
        "obs_norm": True,
        "lr_anneal": "linear",
        "target_kl": trial.suggest_float("target_kl", 0.01, 0.04),
    }
    return params

# --- Objective ---

def make_objective(base_cfg_path: Path, budget_timesteps: int, constant_overrides: dict):
    base_cfg = load_yaml(base_cfg_path)

    def objective(trial: optuna.Trial):
        # Copy base config and apply sampled params + constant overrides
        cfg = dict(base_cfg)
        # Ensure required keys exist
        cfg.setdefault("env_id", "Pendulum-v1")
        cfg.setdefault("num_envs", 16)
        cfg.setdefault("actor_hidden_sizes", [64, 64])
        cfg.setdefault("critic_hidden_sizes", [64, 64])
        cfg.setdefault("activation", "tanh")
        cfg.setdefault("float32_matmul_precision", "high")

        sampled = sample_params(trial)
        cfg.update(sampled)
        cfg.update(constant_overrides or {})

        # Reduce training budget per trial to speed search
        cfg["total_timesteps"] = int(budget_timesteps)

        # Create a temp YAML for this trial
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            tmp_yaml = td / "trial.yaml"
            dump_yaml(cfg, tmp_yaml)

            # Each trial should write to its own TB run dir; train.py already does this
            score = run_train_with_config(tmp_yaml)

        # We want to MAXIMIZE mean_return (closer to 0 is better for Pendulum-v1),
        # so we return it directly and ask Optuna to maximize the objective.
        return score

    return objective

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, required=True, help="Path to base YAML (e.g., configs/ppo_pendulum.yaml)")
    ap.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    ap.add_argument("--timesteps_per_trial", type=int, default=150000, help="Training budget per trial")
    ap.add_argument("--study_name", type=str, default="ppo_pendulum_optuna")
    ap.add_argument("--storage", type=str, default="", help="Optuna storage URL (e.g., sqlite:///ppo.db). If empty, uses in-memory.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    base_cfg_path = Path(args.base_config)
    if not base_cfg_path.exists():
        print(f"Base config not found: {base_cfg_path}", file=sys.stderr)
        sys.exit(1)

    # Constant overrides that you prefer to keep fixed during search
    constant_overrides = {
        "env_id": "Pendulum-v1",
        "num_envs": 16,
        "use_compile": False,  # keep off during search unless compile is stable
        "device": None,        # let code auto-pick CUDA if available
    }

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=min(5, args.trials//2 or 1))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(3, args.trials), n_warmup_steps=1)

    objective = make_objective(base_cfg_path, args.timesteps_per_trial, constant_overrides)

    if args.storage:
        study = optuna.create_study(direction="maximize", study_name=args.study_name, storage=args.storage, load_if_exists=True, sampler=sampler)
    else:
        study = optuna.create_study(direction="maximize", study_name=args.study_name, sampler=sampler)

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True, callbacks=[])

    print("\n=== Best Trial ===")
    best = study.best_trial
    print(f"Value (mean_return): {best.value}")
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Optionally save best params to a YAML you can reuse
    out_cfg = Path("configs") / f"ppo_pendulum_best_{args.study_name}.yaml"
    base_cfg = load_yaml(base_cfg_path)
    base_cfg.update(best.params)
    base_cfg["total_timesteps"] = 300000  # restore full budget for final training
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(base_cfg, out_cfg)
    print(f"\nSaved best-config YAML to: {out_cfg}")

if __name__ == "__main__":
    main()
