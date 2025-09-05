# rl/utils/envs.py
from __future__ import annotations
from typing import Callable
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv


def make_env(env_id: str, seed: int, idx: int = 0, capture_video: bool = False) -> Callable:
    def _thunk():
        env = gym.make(env_id)
        env.reset(seed=seed + idx)
        # Add wrappers here if needed (NormalizeObservation, etc.)
        return env
    return _thunk


def make_vec_env(env_id: str, num_envs: int, seed: int, vector_type: str = "sync"):
    thunks = [make_env(env_id, seed, i) for i in range(num_envs)]
    if vector_type.lower().startswith("async"):
        return AsyncVectorEnv(thunks)
    return SyncVectorEnv(thunks)
