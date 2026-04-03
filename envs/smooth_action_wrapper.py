"""
Post-hoc action smoothing (Option D — inference-only).

Holds the last executed discrete action for `hold_steps` frames before accepting
a different one. Training does not use this; checkpoints stay valid.

See docs/action-smoothing-options.md
"""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np

import config as cfg


class SmoothActionWrapper(gym.Wrapper):
    """Filters raw actions so direction changes require a minimum commit time."""

    def __init__(self, env: gym.Env, hold_steps: Optional[int] = None) -> None:
        super().__init__(env)
        self.hold_steps = max(
            1, int(cfg.ACTION_HOLD_STEPS if hold_steps is None else hold_steps)
        )
        self._current_action: Optional[int] = None
        self._steps_held: int = 0

    @property
    def last_executed_action(self) -> Optional[int]:
        """Action last applied to the inner env (for eval / motor stats)."""
        return self._current_action

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        self._current_action = None
        self._steps_held = 0
        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        action: int | np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        a = int(action) if not isinstance(action, int) else action
        if self._current_action is None:
            self._current_action = a
            self._steps_held = 1
        elif a == self._current_action:
            self._steps_held += 1
        elif self._steps_held >= self.hold_steps:
            self._current_action = a
            self._steps_held = 1
        else:
            self._steps_held += 1
        assert self._current_action is not None
        return self.env.step(self._current_action, **kwargs)


class ActionSmoother:
    """Same hold logic as `SmoothActionWrapper`, for use outside Gym (e.g. manual predict→step)."""

    def __init__(self, hold_steps: Optional[int] = None) -> None:
        self.hold_steps = max(
            1, int(cfg.ACTION_HOLD_STEPS if hold_steps is None else hold_steps)
        )
        self._current_action: Optional[int] = None
        self._steps_held: int = 0

    def reset(self) -> None:
        self._current_action = None
        self._steps_held = 0

    def __call__(self, action: int | np.ndarray) -> int:
        a = int(action) if not isinstance(action, int) else action
        if self._current_action is None:
            self._current_action = a
            self._steps_held = 1
        elif a == self._current_action:
            self._steps_held += 1
        elif self._steps_held >= self.hold_steps:
            self._current_action = a
            self._steps_held = 1
        else:
            self._steps_held += 1
        assert self._current_action is not None
        return self._current_action
