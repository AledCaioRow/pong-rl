"""
Headless evaluation: load a PPO .zip, run N full games, optional JSON output.

Run from repo root:
    py evaluation/eval_agent.py --model models/pong_competent.zip --phase 1 --games 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from stable_baselines3 import PPO

import config as cfg
from envs.margin_env import MarginTargetingWrapper
from envs.pong_env import PongEnv
from envs.smooth_action_wrapper import SmoothActionWrapper

Opponent = Literal["rule_based", "idle"]


def build_env(phase: int, opponent: Opponent, hold_steps: int) -> Any:
    base = PongEnv(opponent=opponent, render_mode=None)
    env: Any = MarginTargetingWrapper(base) if phase == 2 else base
    if hold_steps > 1:
        env = SmoothActionWrapper(env, hold_steps=hold_steps)
    return env


def _motor_action(env: Any, raw_int: int) -> int:
    """Action actually applied to the inner MDP this step (post smoothing)."""
    if isinstance(env, SmoothActionWrapper):
        ex = env.last_executed_action
        assert ex is not None
        return int(ex)
    return int(raw_int)


def run_one_game(
    model: PPO,
    env: Any,
    game_idx: int,
    phase: int,
    seed: Optional[int],
    deterministic: bool,
) -> dict[str, Any]:
    if seed is not None:
        obs, _info = env.reset(seed=int(seed) + game_idx)
    else:
        obs, _info = env.reset()

    total_steps = 0
    rallies: list[int] = []
    action_changes = 0
    prev_motor: Optional[int] = None
    terminated = False

    while not terminated:
        raw_action, _ = model.predict(obs, deterministic=deterministic)
        raw_int = int(np.asarray(raw_action).item())
        obs, _reward, terminated, _trunc, info = env.step(raw_int)
        total_steps += 1

        motor = _motor_action(env, raw_int)
        if prev_motor is not None and motor != prev_motor:
            action_changes += 1
        prev_motor = motor

        if isinstance(info, dict) and "rally_hits" in info:
            rallies.append(int(info["rally_hits"]))

    agent_score = int(obs[6])
    opponent_score = int(obs[7])
    margin = agent_score - opponent_score
    total_points = agent_score + opponent_score

    target_margin: Optional[int] = None
    margin_error: Optional[int] = None
    if phase == 2:
        target_margin = int(round(float(obs[8])))
        margin_error = abs(margin - target_margin)

    mean_rally = float(np.mean(rallies)) if rallies else 0.0
    max_rally = int(max(rallies)) if rallies else 0
    acr = action_changes / max(1, total_steps)

    return {
        "game": game_idx + 1,
        "agent_score": agent_score,
        "opponent_score": opponent_score,
        "margin": margin,
        "target_margin": target_margin,
        "margin_error": margin_error,
        "total_points": total_points,
        "total_steps": total_steps,
        "rallies": rallies,
        "mean_rally_length": mean_rally,
        "max_rally_length": max_rally,
        "action_changes": action_changes,
        "action_change_rate": acr,
        "episode_duration_steps": total_steps,
        "win": agent_score > opponent_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint headlessly.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--games", type=int, default=cfg.EVAL_GAMES)
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=1,
        help="SmoothActionWrapper hold (1=off).",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=("rule_based", "idle"),
        default="rule_based",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=cfg.EVAL_DETERMINISTIC,
    )
    args = parser.parse_args()

    os.chdir(_ROOT)

    raw_mp = Path(args.model)
    model_path = raw_mp
    if not model_path.is_file():
        alt = raw_mp.with_suffix(".zip")
        if alt.is_file():
            model_path = alt
        else:
            raise FileNotFoundError(f"Model not found: {args.model}")

    model = PPO.load(str(model_path))

    hold_steps = max(1, args.hold_steps)
    games_out: list[dict[str, Any]] = []
    for g in range(args.games):
        env = build_env(args.phase, args.opponent, hold_steps)
        try:
            games_out.append(
                run_one_game(model, env, g, args.phase, args.seed, args.deterministic)
            )
        finally:
            env.close()

    payload: dict[str, Any] = {
        "meta": {
            "model": str(model_path.resolve()),
            "phase": args.phase,
            "games": args.games,
            "seed": args.seed,
            "hold_steps": hold_steps,
            "opponent": args.opponent,
            "deterministic": args.deterministic,
        },
        "games": games_out,
    }

    text = json.dumps(payload, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path} ({args.games} games).")
    else:
        print(text)


if __name__ == "__main__":
    main()
