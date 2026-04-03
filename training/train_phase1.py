"""
Phase 1: train a competent PPO agent against the rule-based opponent (spec).

Run from the project root, e.g.:
    python training/train_phase1.py
        (without --model-path you are prompted for a save name; --quick skips the prompt and uses config default)
    python training/train_phase1.py --quick
    python training/train_phase1.py --timesteps 10000 --eval-games 5 --model-path models/my_run

TensorBoard (separate terminal from training — it does not open automatically):
    py -m pip install -r requirements.txt
    py -m tensorboard.main --logdir training/logs/phase1_ppo --port 6006
    Browser: http://127.0.0.1:6006  (if that fails, try http://localhost:6006)
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
from training.output_model_path import resolve_output_model_path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def _tensorboard_log(path: str | None, no_tensorboard: bool) -> str | None:
    """Return log dir only if PyTorch can use TensorBoard (same check SB3 uses internally)."""
    if no_tensorboard or not path:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None  # type: ignore[misc, assignment]
    if SummaryWriter is None:
        print(
            "[warn] TensorBoard not available for PyTorch; training without TB logs. "
            "Install: py -m pip install -r requirements.txt"
        )
        return None
    return path


def _make_pong(
    *,
    paddle_speed_scale: float = 1.0,
    ball_speed_scale: float = 1.0,
) -> PongEnv:
    return PongEnv(
        opponent="rule_based",
        paddle_speed_scale=paddle_speed_scale,
        ball_speed_scale=ball_speed_scale,
    )


def evaluate_win_rate(
    model: Any,
    n_episodes: int,
    seed: int,
    *,
    paddle_speed_scale: float,
    ball_speed_scale: float,
) -> float:
    """Fraction of episodes won by the agent (first to MAX_SCORE)."""
    wins = 0
    for ep in range(n_episodes):
        env = PongEnv(
            opponent="rule_based",
            paddle_speed_scale=paddle_speed_scale,
            ball_speed_scale=ball_speed_scale,
        )
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
        "--paddle-speed-scale",
        type=float,
        default=1.0,
        help="Scale paddle speed for training/eval envs (default 1.0).",
    )
    parser.add_argument(
        "--ball-speed-scale",
        type=float,
        default=1.0,
        help="Scale ball speed for training/eval envs (default 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.SEED,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=argparse.SUPPRESS,
        help=(
            "Output path without extension; SB3 saves as .zip. "
            "If omitted (when not using --quick), you are prompted for a name in the terminal."
        ),
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
        "--no-progress-bar",
        action="store_true",
        help="Disable the tqdm/rich bar (includes estimated time remaining).",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="If > 0, run this many eval games after training (deterministic policy).",
    )
    args = parser.parse_args()
    model_cli_provided = hasattr(args, "model_path")
    model_cli_value = getattr(args, "model_path", None)

    if args.quick:
        args.timesteps = cfg.PHASE1_QUICK_TIMESTEPS
        print(
            f"[quick] {args.timesteps} timesteps "
            f"(~10 min wall time on a typical desktop CPU; open TensorBoard to watch curves)."
        )

    os.chdir(_ROOT)

    model_save_path = resolve_output_model_path(
        quick=args.quick,
        cli_provided=model_cli_provided,
        cli_value=model_cli_value,
        quick_default=cfg.PHASE1_MODEL_PATH,
    )
    print(f"[save] Checkpoint will be written to {model_save_path}.zip")

    tensorboard_log = _tensorboard_log(args.tensorboard, args.no_tensorboard)
    if tensorboard_log:
        log_abs = os.path.abspath(tensorboard_log)
        print(
            "\n[TensorBoard] Start in a **separate** terminal and leave it open "
            "(nothing listens on :6006 until you run this — ERR_CONNECTION_REFUSED means it is not running):\n"
            f'  py -m tensorboard.main --logdir "{log_abs}" --host 127.0.0.1 --port 6006\n'
            "  Then http://127.0.0.1:6006/ — if Cursor’s Simple Browser fails, use Chrome or Edge.\n"
        )

    vec_env = make_vec_env(
        lambda: _make_pong(
            paddle_speed_scale=args.paddle_speed_scale,
            ball_speed_scale=args.ball_speed_scale,
        ),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        seed=args.seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=not args.no_progress_bar,
    )
    model.save(model_save_path)
    print(f"Saved model to {model_save_path}.zip")

    if args.eval_games > 0:
        rate = evaluate_win_rate(
            model,
            args.eval_games,
            seed=args.seed + 10_000,
            paddle_speed_scale=args.paddle_speed_scale,
            ball_speed_scale=args.ball_speed_scale,
        )
        print(f"Eval win rate over {args.eval_games} games: {rate:.1%}")

    vec_env.close()


if __name__ == "__main__":
    main()
