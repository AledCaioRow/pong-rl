"""
Margin-targeting reward wrapper (Phase 2, spec).

Wraps PongEnv: each episode samples target_margin in [TARGET_MARGIN_MIN, TARGET_MARGIN_MAX],
replaces sparse score rewards with margin-aware shaping, rally bonus on points, and terminal
proximity bonus. Optionally randomises opponent track noise per episode for a harder mix.
"""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np

import config as cfg
from envs.pong_env import PongEnv


def _margin_point_reward(scorer: str, margin: int, target: int) -> float:
    """Per-point reward from spec (after the point is scored)."""
    if scorer == "agent":
        if margin <= target:
            return 1.0
        return -0.5
    # human scored
    if margin > target:
        return 0.3
    return -1.0


class MarginTargetingWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        randomize_opponent_noise: bool = True,
    ) -> None:
        super().__init__(env)
        self.randomize_opponent_noise = randomize_opponent_noise

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        opts = dict(options) if options else {}
        if "target_margin" not in opts:
            gen = np.random.default_rng(seed)
            opts["target_margin"] = int(
                gen.integers(cfg.TARGET_MARGIN_MIN, cfg.TARGET_MARGIN_MAX + 1)
            )

        obs, info = self.env.reset(seed=seed, options=opts)

        base = self.env.unwrapped
        if self.randomize_opponent_noise and isinstance(base, PongEnv):
            levels = cfg.PHASE2_OPPONENT_NOISE_LEVELS
            if levels:
                idx = int(base.np_random.integers(0, len(levels)))
                base.opponent_track_noise = float(levels[idx])

        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, _base_rew, terminated, truncated, info = self.env.step(action)
        rew = self._compute_reward(obs, info, terminated)
        return obs, rew, terminated, truncated, info

    def _compute_reward(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        terminated: bool,
    ) -> float:
        target = int(round(float(obs[8])))
        margin = int(round(float(obs[9])))
        total = 0.0

        if "point_scored_by" in info:
            scorer = str(info["point_scored_by"])
            total += _margin_point_reward(scorer, margin, target)
            rh = int(info.get("rally_hits", 0))
            total += min(rh / 50.0, 0.5)

        if terminated:
            total += max(0.0, 5.0 - abs(margin - target))

        return float(total)
