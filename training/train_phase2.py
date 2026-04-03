"""
Phase 2: fine-tune Phase 1 PPO for margin targeting (spec).

Loads models/pong_competent.zip (or --load-path) and trains with MarginTargetingWrapper.

Requires a Phase 1 checkpoint:
    python training/train_phase1.py

Then:
    python training/train_phase2.py
    python training/train_phase2.py --timesteps 8192 --no-tensorboard --eval-games 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg
import numpy as np
from envs.margin_env import MarginTargetingWrapper
from envs.pong_env import PongEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def _make_margin_env() -> MarginTargetingWrapper:
    base = PongEnv(opponent="rule_based")
    return MarginTargetingWrapper(base, randomize_opponent_noise=True)


def evaluate_margin_error(model: Any, n_episodes: int, seed: int) -> tuple[float, float]:
    """Mean and std of |final_margin - target| over episodes."""
    errors: list[float] = []
    for ep in range(n_episodes):
        env = MarginTargetingWrapper(PongEnv(opponent="rule_based"), randomize_opponent_noise=True)
        obs, _ = env.reset(seed=seed + ep)
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, _, _ = env.step(int(action))
        target = int(round(float(obs[8])))
        margin = int(round(float(obs[9])))
        errors.append(abs(margin - target))
    arr = np.array(errors, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 PPO fine-tune (margin targeting).")
    parser.add_argument("--timesteps", type=int, default=cfg.PHASE2_TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=cfg.PHASE2_N_ENVS)
    parser.add_argument("--learning-rate", type=float, default=cfg.PHASE2_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument(
        "--load-path",
        type=str,
        default=cfg.PHASE2_LOAD_PATH,
        help="Phase 1 checkpoint path (with or without .zip).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=cfg.PHASE2_MODEL_PATH,
        help="Output path without extension; saved as .zip",
    )
    parser.add_argument("--tensorboard", type=str, default="training/logs/phase2_ppo")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="If > 0, report mean |margin - target| over this many games after training.",
    )
    args = parser.parse_args()

    os.chdir(_ROOT)

    load_path = Path(args.load_path)
    if not load_path.suffix:
        load_path = load_path.with_suffix(".zip")
    if not load_path.is_file():
        raise FileNotFoundError(
            f"Phase 1 model not found at {load_path}. Train Phase 1 first: "
            f"python training/train_phase1.py"
        )

    vec_env = make_vec_env(
        _make_margin_env,
        n_envs=args.n_envs,
        seed=args.seed,
    )

    tensorboard_log = None if args.no_tensorboard else args.tensorboard
    model = PPO.load(
        str(load_path),
        env=vec_env,
        learning_rate=args.learning_rate,
        seed=args.seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_path)
    print(f"Saved model to {args.model_path}.zip")

    if args.eval_games > 0:
        mean_e, std_e = evaluate_margin_error(
            model, args.eval_games, seed=args.seed + 20_000
        )
        print(
            f"Eval margin error |final - target|: mean={mean_e:.2f}, std={std_e:.2f} "
            f"over {args.eval_games} games"
        )

    vec_env.close()


if __name__ == "__main__":
    main()
