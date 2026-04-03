"""
Phase 1: train a competent PPO agent against the rule-based opponent (spec).

Run from the project root, e.g.:
    python training/train_phase1.py
    python training/train_phase1.py --quick
    python training/train_phase1.py --timesteps 10000 --eval-games 5

Monitoring UI: py -m pip install tensorboard
    py -m tensorboard.main --logdir training/logs/phase1_ppo
    (or: Scripts\\tensorboard.exe under your Python install if PATH is set)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# Project root on sys.path when launched as python training/train_phase1.py
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg
from envs.pong_env import PongEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def _tensorboard_log(path: str | None, no_tensorboard: bool) -> str | None:
    if no_tensorboard or not path:
        return None
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print(
            "[warn] tensorboard not installed; training without TB logs. "
            "Install: py -m pip install tensorboard"
        )
        return None
    return path


def _make_pong() -> PongEnv:
    return PongEnv(opponent="rule_based")


def evaluate_win_rate(model: Any, n_episodes: int, seed: int) -> float:
    """Fraction of episodes won by the agent (first to MAX_SCORE)."""
    wins = 0
    for ep in range(n_episodes):
        env = PongEnv(opponent="rule_based")
        obs, _ = env.reset(seed=seed + ep)
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, _, _ = env.step(int(action))
        if obs[6] > obs[7]:
            wins += 1
    return wins / max(1, n_episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 PPO training (competent Pong agent).")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=cfg.PHASE1_TOTAL_TIMESTEPS,
        help="Total environment steps across all parallel envs.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Smoke run: {cfg.PHASE1_QUICK_TIMESTEPS} steps (~10 min on many CPUs; YMMV). Overrides --timesteps.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=cfg.PHASE1_N_ENVS,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=cfg.PHASE1_LEARNING_RATE,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.SEED,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=cfg.PHASE1_MODEL_PATH,
        help="Path without extension; SB3 saves as .zip",
    )
    parser.add_argument(
        "--tensorboard",
        type=str,
        default="training/logs/phase1_ppo",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging.",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="If > 0, run this many eval games after training (deterministic policy).",
    )
    args = parser.parse_args()

    if args.quick:
        args.timesteps = cfg.PHASE1_QUICK_TIMESTEPS
        print(
            f"[quick] {args.timesteps} timesteps "
            f"(~10 min wall time on a typical desktop CPU; open TensorBoard to watch curves)."
        )

    os.chdir(_ROOT)

    vec_env = make_vec_env(
        _make_pong,
        n_envs=args.n_envs,
        seed=args.seed,
    )

    tensorboard_log = _tensorboard_log(args.tensorboard, args.no_tensorboard)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        seed=args.seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_path)
    print(f"Saved model to {args.model_path}.zip")

    if args.eval_games > 0:
        rate = evaluate_win_rate(model, args.eval_games, seed=args.seed + 10_000)
        print(f"Eval win rate over {args.eval_games} games: {rate:.1%}")

    vec_env.close()


if __name__ == "__main__":
    main()
