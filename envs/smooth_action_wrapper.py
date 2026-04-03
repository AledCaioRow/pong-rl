"""
Post-hoc action smoothing (Option D — inference-only).

Constrains the agent's motor output so it looks human:

* **Base hold** — every action must be held for at least `hold_steps` frames
  before a different one is accepted.
* **Reversal penalty** — switching between up (1) and down (2) requires
  `reversal_hold_steps` frames (≥ `hold_steps`).  Humans physically cannot
  reverse paddle direction as fast as they can stop or start.
* **Random jitter** — each hold target is varied by up to ±`jitter` frames so
  direction changes don't land on a metronomic cadence.

Training does not use this; checkpoints stay valid.
"""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np

import config as cfg

_REVERSAL_PAIRS = frozenset({(1, 2), (2, 1)})


def _is_reversal(old: int, new: int) -> bool:
    return (old, new) in _REVERSAL_PAIRS


class SmoothActionWrapper(gym.Wrapper):
    """Gym wrapper that enforces human-like motor constraints on discrete actions."""

    def __init__(
        self,
        env: gym.Env,
        hold_steps: Optional[int] = None,
        reversal_hold_steps: Optional[int] = None,
        jitter: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self.hold_steps = max(
            1, int(cfg.ACTION_HOLD_STEPS if hold_steps is None else hold_steps)
        )
        self.reversal_hold_steps = max(
            self.hold_steps,
            int(
                cfg.ACTION_REVERSAL_HOLD_STEPS
                if reversal_hold_steps is None
                else reversal_hold_steps
            ),
        )
        self.jitter = max(
            0, int(cfg.ACTION_HOLD_JITTER if jitter is None else jitter)
        )

        self._current_action: Optional[int] = None
        self._steps_held: int = 0
        self._hold_target: int = self.hold_steps
        self._rng = np.random.default_rng()

    @property
    def last_executed_action(self) -> Optional[int]:
        """Action last applied to the inner env (for eval / motor stats)."""
        return self._current_action

    def _jittered_target(self, base: int) -> int:
        if self.jitter <= 0:
            return base
        return base + int(self._rng.integers(0, self.jitter + 1))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        self._current_action = None
        self._steps_held = 0
        self._hold_target = self.hold_steps
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
            self._hold_target = self._jittered_target(self.hold_steps)
        elif a == self._current_action:
            self._steps_held += 1
        elif self._steps_held >= self._hold_target:
            reversal = _is_reversal(self._current_action, a)
            self._current_action = a
            self._steps_held = 1
            base = self.reversal_hold_steps if reversal else self.hold_steps
            self._hold_target = self._jittered_target(base)
        else:
            self._steps_held += 1

        assert self._current_action is not None
        return self.env.step(self._current_action, **kwargs)


class ActionSmoother:
    """Same hold logic as `SmoothActionWrapper`, for use outside Gym (e.g. manual predict->step)."""

    def __init__(
        self,
        hold_steps: Optional[int] = None,
        reversal_hold_steps: Optional[int] = None,
        jitter: Optional[int] = None,
    ) -> None:
        self.hold_steps = max(
            1, int(cfg.ACTION_HOLD_STEPS if hold_steps is None else hold_steps)
        )
        self.reversal_hold_steps = max(
            self.hold_steps,
            int(
                cfg.ACTION_REVERSAL_HOLD_STEPS
                if reversal_hold_steps is None
                else reversal_hold_steps
            ),
        )
        self.jitter = max(
            0, int(cfg.ACTION_HOLD_JITTER if jitter is None else jitter)
        )

        self._current_action: Optional[int] = None
        self._steps_held: int = 0
        self._hold_target: int = self.hold_steps
        self._rng = np.random.default_rng()

    def _jittered_target(self, base: int) -> int:
        if self.jitter <= 0:
            return base
        return base + int(self._rng.integers(0, self.jitter + 1))

    def reset(self) -> None:
        self._current_action = None
        self._steps_held = 0
        self._hold_target = self.hold_steps

    def __call__(self, action: int | np.ndarray) -> int:
        a = int(action) if not isinstance(action, int) else action

        if self._current_action is None:
            self._current_action = a
            self._steps_held = 1
            self._hold_target = self._jittered_target(self.hold_steps)
        elif a == self._current_action:
            self._steps_held += 1
        elif self._steps_held >= self._hold_target:
            reversal = _is_reversal(self._current_action, a)
            self._current_action = a
            self._steps_held = 1
            base = self.reversal_hold_steps if reversal else self.hold_steps
            self._hold_target = self._jittered_target(base)
        else:
            self._steps_held += 1

        assert self._current_action is not None
        return self._current_action
