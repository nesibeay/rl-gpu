
import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path
import subprocess
from typing import Optional, List

import optuna
import yaml

MEAN_RET_RAW_RE = re.compile(r"Training finished: \{'steps':\s*(\d+),\s*'mean_return':\s*([-\d\.naninf]+),\s*'mean_return_raw':\s*([-\d\.naninf]+)\}")
MEAN_RET_RE = re.compile(r"Training finished: \{'steps':\s*(\d+),\s*'mean_return':\s*([-\d\.naninf]+)\}")

def run_train_with_config(config_path: Path, timeout_s: Optional[int] = None) -> float:
    cmd = [sys.executable, "train.py", "--config", str(config_path)]
    print(f"[tune] Launch: {' '.join(cmd)}")
    start = time.time()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True) as proc:
        ret_raw = None
        ret_scaled = None
        for line in proc.stdout:
            print(line, end="")
            m_raw = MEAN_RET_RAW_RE.search(line)
            if m_raw:
                try:
                    ret_scaled = float(m_raw.group(2))
                    ret_raw = float(m_raw.group(3))
                except Exception:
                    ret_scaled, ret_raw = None, None
            else:
                m = MEAN_RET_RE.search(line)
                if m:
                    try:
                        ret_scaled = float(m.group(2))
                    except Exception:
                        ret_scaled = None
            if timeout_s is not None and (time.time() - start) > timeout_s:
                print("[tune] Trial timeout reached, terminating training process...")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break

        retcode = proc.wait()
        if retcode not in (0, None):
            print(f"[warn] train.py exited with code {retcode}")

    if ret_raw is not None:
        return ret_raw
    if ret_scaled is not None:
        return ret_scaled
    raise RuntimeError("Could not parse mean return from train.py output.")

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def sample_params(trial: optuna.Trial) -> dict:
    net_choices = [(64,64), (128,128), (256,256)]
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "clip_coef": trial.suggest_float("clip_coef", 0.08, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 2e-4, 2e-2, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.9),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 1.5),
        "gamma": trial.suggest_float("gamma", 0.97, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.98),
        "rollout_steps": trial.suggest_categorical("rollout_steps", [512, 768, 1024, 1536, 2048]),
        "n_epochs": trial.suggest_int("n_epochs", 10, 28),
        "minibatch_size": trial.suggest_categorical("minibatch_size", [2048, 3072, 4096, 6144, 8192]),
        "init_log_std": trial.suggest_float("init_log_std", -2.0, -0.5),
        "target_kl": trial.suggest_float("target_kl", 0.01, 0.05),
        "actor_hidden_sizes": trial.suggest_categorical("actor_hidden_sizes", net_choices),
        "critic_hidden_sizes": trial.suggest_categorical("critic_hidden_sizes", net_choices),
        "activation": trial.suggest_categorical("activation", ["tanh", "relu"]),
        "obs_norm": trial.suggest_categorical("obs_norm", [True, False]),
        "use_value_clip": trial.suggest_categorical("use_value_clip", [True, False]),
        "adv_norm": True,
        "lr_anneal": "linear",
        "use_compile": False,
        "float32_matmul_precision": "high",
    }

def make_objective(base_cfg_path: Path, budget_timesteps: int, constant_overrides: dict, timeout_s: Optional[int], seeds: List[int]):
    base_cfg = load_yaml(base_cfg_path)

    def objective(trial: optuna.Trial):
        sampled = sample_params(trial)
        scores = []
        for seed in seeds:
            cfg = dict(base_cfg)
            cfg.update(sampled)
            cfg.update(constant_overrides or {})
            cfg["total_timesteps"] = int(budget_timesteps)
            cfg["seed"] = int(seed)

            with tempfile.TemporaryDirectory() as td:
                tmp_yaml = Path(td) / f"trial_seed{seed}.yaml"
                dump_yaml(cfg, tmp_yaml)
                score = run_train_with_config(tmp_yaml, timeout_s=timeout_s)
                scores.append(score)

        avg_score = sum(scores) / max(1, len(scores))
        print(f"[tune] Trial {trial.number} seeds {seeds} -> avg mean_return(raw) {avg_score:.3f}")
        return avg_score

    return objective

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timesteps_per_trial", type=int, default=120000)
    ap.add_argument("--study_name", type=str, default="pendulum_tune")
    ap.add_argument("--storage", type=str, default="sqlite:///ppo.db")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout_minutes", type=int, default=0)
    ap.add_argument("--seeds", type=str, default="1,2")
    return ap.parse_args()

def main():
    args = parse_args()
    base_cfg_path = Path(args.base_config)
    if not base_cfg_path.exists():
        print(f"Base config not found: {base_cfg_path}", file=sys.stderr)
        sys.exit(1)

    constant_overrides = {
        "env_id": "Pendulum-v1",
        "num_envs": 16,
        "device": None,
    }

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] or [1]
    timeout_s = (args.timeout_minutes * 60) if args.timeout_minutes > 0 else None

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10)
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=3, min_early_stopping_rate=0)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    print(f"[tune] Using storage: {args.storage}")
    print(f"[tune] Study name: {args.study_name}")
    if args.storage.startswith("sqlite:///"):
        db_path = args.storage[len("sqlite:///"):]
        print(f"[tune] Visualize with: optuna-dashboard sqlite:///{db_path} --study {args.study_name}")

    objective = make_objective(base_cfg_path, args.timesteps_per_trial, constant_overrides, timeout_s, seeds)

    try:
        study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user. Printing best-so-far...")

    if len(study.trials) == 0:
        print("No trials completed.")
        return

    best = study.best_trial
    print("\n=== Best Trial ===")
    print(f"Value (mean_return_raw): {best.value}")
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    out_cfg = Path("configs") / f"ppo_pendulum_best_{args.study_name}.yaml"
    base_cfg = load_yaml(base_cfg_path)
    base_cfg.update(best.params)
    base_cfg["total_timesteps"] = 300000
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(base_cfg, out_cfg)
    print(f"\nSaved best-config YAML to: {out_cfg}")

if __name__ == "__main__":
    main()
