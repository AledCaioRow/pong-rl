"""
Classic Pong in normalised court coordinates with vector observations.

Agent paddle is on the right; human/opponent on the left.
Episode ends when either side reaches MAX_SCORE (see config).
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as cfg

# "keyboard" = left paddle from external discrete actions (human vs PPO).
OpponentType = Literal["rule_based", "idle", "keyboard"]


def _clamp(y: float, lo: float, hi: float) -> float:
    return float(np.clip(y, lo, hi))


def _norm_v(vx: float, vy: float, speed_cap: float | None = None) -> Tuple[float, float]:
    """Map velocities into [-1, 1] using `speed_cap` (defaults to cfg.BALL_MAX_SPEED)."""
    cap = float(cfg.BALL_MAX_SPEED if speed_cap is None else speed_cap)
    if cap <= 0:
        cap = 1e-6
    return float(np.clip(vx / cap, -1.0, 1.0)), float(np.clip(vy / cap, -1.0, 1.0))


class PongEnv(gym.Env):
    """
    Observation (float32 vector length 10):
        ball_x, ball_y, ball_vx_norm, ball_vy_norm,
        agent_paddle_y, human_paddle_y,
        score_agent, score_human, target_margin, current_margin
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": cfg.FPS}

    # Internal geometry: left = human, right = agent
    _MARGIN_X: float = 0.04

    def __init__(
        self,
        *,
        opponent: OpponentType = "rule_based",
        target_margin: int = 0,
        render_mode: Optional[str] = None,
        # Rule-based opponent: Gaussian noise (court height units) on tracked y.
        # None -> use config.RULE_BASED_OPPONENT_TRACK_NOISE.
        opponent_track_noise: Optional[float] = None,
        # Scales cfg.PADDLE_SPEED (e.g. 0.6 for slower human playtests; keep 1.0 for training).
        paddle_speed_scale: float = 1.0,
        # Scales cfg.BALL_INITIAL_SPEED / BALL_MAX_SPEED (human playtests; keep 1.0 for training).
        ball_speed_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.opponent = opponent
        self._target_margin = int(np.clip(target_margin, 0, cfg.MAX_SCORE))
        self.render_mode = render_mode
        self.opponent_track_noise = float(
            cfg.RULE_BASED_OPPONENT_TRACK_NOISE
            if opponent_track_noise is None
            else opponent_track_noise
        )
        self._paddle_speed = float(cfg.PADDLE_SPEED * max(0.01, paddle_speed_scale))
        _bs = max(0.05, float(ball_speed_scale))
        self._ball_initial_speed = float(cfg.BALL_INITIAL_SPEED * _bs)
        self._ball_max_speed = float(cfg.BALL_MAX_SPEED * _bs)

        low = np.array(
            [
                0.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                float(-cfg.MAX_SCORE),
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                float(cfg.MAX_SCORE),
                float(cfg.MAX_SCORE),
                float(cfg.MAX_SCORE),
                float(cfg.MAX_SCORE),
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.ball_x = 0.5
        self.ball_y = 0.5
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.agent_y = 0.5
        self.human_y = 0.5
        self.agent_score = 0
        self.human_score = 0
        # Paddle contacts in the current rally (reset on point scored).
        self._rally_hits: int = 0

        self._human_x = self._MARGIN_X + cfg.PADDLE_HALF_WIDTH
        self._agent_x = cfg.COURT_WIDTH - self._MARGIN_X - cfg.PADDLE_HALF_WIDTH

        self._paddle_min_y = cfg.PADDLE_HALF_HEIGHT
        self._paddle_max_y = cfg.COURT_HEIGHT - cfg.PADDLE_HALF_HEIGHT

        # Rendering cache
        self._render_size = (400, 400)

    @property
    def target_margin(self) -> int:
        return self._target_margin

    @target_margin.setter
    def target_margin(self, value: int) -> None:
        self._target_margin = int(np.clip(value, 0, cfg.MAX_SCORE))

    @property
    def rally_hits(self) -> int:
        """Paddle hits in the current rally (for shaping / margin wrapper)."""
        return self._rally_hits

    def _get_obs(self) -> np.ndarray:
        # Match physics cap: ball_speed_scale uses self._ball_max_speed, so normalize with the same.
        nvx, nvy = _norm_v(self.ball_vx, self.ball_vy, self._ball_max_speed)
        margin = self.agent_score - self.human_score
        return np.array(
            [
                float(self.ball_x),
                float(self.ball_y),
                nvx,
                nvy,
                float(self.agent_y),
                float(self.human_y),
                float(self.agent_score),
                float(self.human_score),
                float(self._target_margin),
                float(margin),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.agent_score = 0
        self.human_score = 0
        self.agent_y = 0.5
        self.human_y = 0.5

        if options and "target_margin" in options:
            self._target_margin = int(
                np.clip(int(options["target_margin"]), 0, cfg.MAX_SCORE)
            )

        rng = self.np_random
        self.ball_x = 0.5
        self.ball_y = float(rng.uniform(0.25, 0.75))

        direction = 1.0 if rng.random() < 0.5 else -1.0
        angle = float(rng.uniform(-math.pi / 5, math.pi / 5))
        speed = self._ball_initial_speed
        self.ball_vx = direction * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)
        self._rally_hits = 0

        return self._get_obs(), {}

    def step(
        self,
        action: int | np.ndarray,
        *,
        human_action: int | np.ndarray | None = None,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        a = int(action) if not isinstance(action, int) else action
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        # --- Agent paddle (right) ---
        self._move_agent(a)

        # --- Left paddle ---
        if self.opponent == "idle":
            pass
        elif self.opponent == "keyboard":
            if human_action is None:
                raise ValueError("opponent='keyboard' requires human_action= each step.")
            ha = int(human_action) if not isinstance(human_action, int) else human_action
            self._move_human_paddle(ha)
        else:
            rng = self.np_random
            noise = (
                0.0
                if self.opponent_track_noise <= 0
                else float(rng.normal(0.0, self.opponent_track_noise))
            )
            target_y = _clamp(self.ball_y + noise, self._paddle_min_y, self._paddle_max_y)
            self._move_paddle_towards(self.human_y, target_y, attr="human")

        # --- Ball integration ---
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Wall bounces (top / bottom)
        if self.ball_y - cfg.BALL_RADIUS < 0.0:
            self.ball_y = cfg.BALL_RADIUS
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y + cfg.BALL_RADIUS > cfg.COURT_HEIGHT:
            self.ball_y = cfg.COURT_HEIGHT - cfg.BALL_RADIUS
            self.ball_vy = -abs(self.ball_vy)

        # Paddle collisions (resolve after position update)
        self._maybe_bounce_off_paddle(is_agent=False)
        self._maybe_bounce_off_paddle(is_agent=True)

        # Scoring: ball past left -> agent scores; past right -> human scores
        if self.ball_x - cfg.BALL_RADIUS < 0.0:
            info["rally_hits"] = self._rally_hits
            self.agent_score += 1
            reward = 1.0
            info["point_scored_by"] = "agent"
            if self.agent_score >= cfg.MAX_SCORE:
                terminated = True
            self._respawn_ball(favor_side="human")

        elif self.ball_x + cfg.BALL_RADIUS > cfg.COURT_WIDTH:
            info["rally_hits"] = self._rally_hits
            self.human_score += 1
            reward = -1.0
            info["point_scored_by"] = "human"
            if self.human_score >= cfg.MAX_SCORE:
                terminated = True
            self._respawn_ball(favor_side="agent")

        return self._get_obs(), reward, terminated, False, info

    def _move_agent(self, action: int) -> None:
        """Actions: 0 stay, 1 up (+y), 2 down (-y) in normalised court space."""
        y = self.agent_y
        if action == 1:
            y += self._paddle_speed
        elif action == 2:
            y -= self._paddle_speed
        self.agent_y = _clamp(y, self._paddle_min_y, self._paddle_max_y)

    def _move_human_paddle(self, action: int) -> None:
        """Same action semantics as `_move_agent`, for the left paddle."""
        y = self.human_y
        if action == 1:
            y += self._paddle_speed
        elif action == 2:
            y -= self._paddle_speed
        self.human_y = _clamp(y, self._paddle_min_y, self._paddle_max_y)

    def _move_paddle_towards(self, _: float, target_y: float, *, attr: str) -> None:
        y = self.human_y if attr == "human" else self.agent_y
        if target_y > y + 1e-6:
            y = min(y + self._paddle_speed, target_y)
        elif target_y < y - 1e-6:
            y = max(y - self._paddle_speed, target_y)
        self.human_y = _clamp(y, self._paddle_min_y, self._paddle_max_y)

    def _maybe_bounce_off_paddle(self, *, is_agent: bool) -> None:
        px = self._agent_x if is_agent else self._human_x
        py = self.agent_y if is_agent else self.human_y

        # Approaching paddle check
        if is_agent and self.ball_vx <= 0:
            return
        if not is_agent and self.ball_vx >= 0:
            return

        dx = abs(self.ball_x - px)
        dy = abs(self.ball_y - py)
        reach_x = cfg.PADDLE_HALF_WIDTH + cfg.BALL_RADIUS
        reach_y = cfg.PADDLE_HALF_HEIGHT + cfg.BALL_RADIUS
        if dx > reach_x or dy > reach_y:
            return

        # Bounce
        self.ball_x = px + (-1 if is_agent else 1) * (reach_x + 1e-4)
        self.ball_vx = -self.ball_vx

        offset = (self.ball_y - py) / max(cfg.PADDLE_HALF_HEIGHT, 1e-6)
        offset = float(np.clip(offset, -1.0, 1.0))
        self.ball_vy += offset * self._ball_initial_speed * 1.8

        speed = math.hypot(self.ball_vx, self.ball_vy)
        cap = self._ball_max_speed
        if speed > cap and speed > 1e-8:
            s = cap / speed
            self.ball_vx *= s
            self.ball_vy *= s
        if speed < self._ball_initial_speed * 0.35 and speed > 0:
            s = (self._ball_initial_speed * 0.5) / speed
            self.ball_vx *= s
            self.ball_vy *= s

        self._rally_hits += 1

    def _respawn_ball(self, *, favor_side: Literal["agent", "human"]) -> None:
        rng = self.np_random
        self.ball_x = 0.5
        self.ball_y = float(rng.uniform(0.25, 0.75))
        if favor_side == "agent":
            direction = 1.0
        else:
            direction = -1.0
        angle = float(rng.uniform(-math.pi / 5, math.pi / 5))
        speed = self._ball_initial_speed
        self.ball_vx = direction * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)
        self._rally_hits = 0

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            gym.logger.warn("Call render without specifying render_mode.")
            return None

        h, w = self._render_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (24, 24, 32)

        def to_px(x: float, y: float) -> Tuple[int, int]:
            xi = int(np.clip(round(x * w), 0, w - 1))
            yi = int(np.clip(round((1.0 - y) * h), 0, h - 1))
            return xi, yi

        # Paddles
        pw = max(1, int(cfg.PADDLE_HALF_WIDTH * w * 2))
        ph = max(1, int(cfg.PADDLE_HALF_HEIGHT * h * 2))
        for paddle_x, paddle_y in ((self._human_x, self.human_y), (self._agent_x, self.agent_y)):
            cx, cy = to_px(paddle_x, paddle_y)
            x0 = max(0, cx - pw)
            x1 = min(w - 1, cx + pw)
            y0 = max(0, cy - ph)
            y1 = min(h - 1, cy + ph)
            img[y0 : y1 + 1, x0 : x1 + 1] = (220, 220, 230)

        br = max(2, int(cfg.BALL_RADIUS * min(h, w)))
        bx, by = to_px(self.ball_x, self.ball_y)
        x0 = max(0, bx - br)
        x1 = min(w - 1, bx + br)
        y0 = max(0, by - br)
        y1 = min(h - 1, by + br)
        img[y0 : y1 + 1, x0 : x1 + 1] = (250, 90, 90)

        if self.render_mode == "rgb_array":
            return img
        return img

    def close(self) -> None:
        pass
