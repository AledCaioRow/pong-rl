"""
Shared simulation and training constants.

Import this from envs and training scripts so physics and normalization stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- Game rules ---
MAX_SCORE: int = 30

# --- Time / step semantics ---
# One env.step() advances the simulation by this many "ticks" of a fixed dt.
# FPS is informational for human rendering (e.g. pygame); physics use DT * speeds.
FPS: int = 60
DT: float = 1.0 / FPS

# --- Court (internal world space; observations are normalised to [0, 1] where noted) ---
COURT_WIDTH: float = 1.0
COURT_HEIGHT: float = 1.0

# --- Paddle geometry & motion (same for agent and human/opponent unless overridden) ---
PADDLE_HALF_HEIGHT: float = 0.08
PADDLE_HALF_WIDTH: float = 0.012
PADDLE_SPEED: float = 0.022

# --- Ball ---
BALL_RADIUS: float = 0.015
BALL_INITIAL_SPEED: float = 0.018
# Used to map ball_vx, ball_vy into approximately [-1, 1] in the observation vector.
BALL_MAX_SPEED: float = 0.06

# --- Observation / reward placeholders (Phase 2+) ---
TARGET_MARGIN_MIN: int = 2
TARGET_MARGIN_MAX: int = 20

# --- Phase 1 opponent (rule-based; same physics as agent) ---
# Gaussian noise (court height units) on the opponent's ball-y target each step.
RULE_BASED_OPPONENT_TRACK_NOISE: float = 0.06

# --- Phase 1 training defaults ---
PHASE1_TOTAL_TIMESTEPS: int = 2_000_000
# Smoke / “~10 min” run (use training/train_phase1.py --quick); wall time varies by CPU.
PHASE1_QUICK_TIMESTEPS: int = 900_000
PHASE1_N_ENVS: int = 4
PHASE1_LEARNING_RATE: float = 3e-4
PHASE1_MODEL_PATH: str = "models/pong_competent"
SEED: int = 42

# --- Phase 2 margin targeting (fine-tune from Phase 1 checkpoint) ---
PHASE2_TOTAL_TIMESTEPS: int = 5_000_000
PHASE2_QUICK_TIMESTEPS: int = 600_000
PHASE2_N_ENVS: int = 4
# Slightly lower LR for fine-tuning on top of Phase 1 features.
PHASE2_LEARNING_RATE: float = 1e-4
PHASE2_MODEL_PATH: str = "models/pong_margin_targeting"
PHASE2_LOAD_PATH: str = "models/pong_competent"
# Rule-based opponent noise levels cycled randomly each episode (VecEnv-friendly mix).
PHASE2_OPPONENT_NOISE_LEVELS: tuple[float, ...] = (0.04, 0.06, 0.09)


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = PHASE1_TOTAL_TIMESTEPS
    n_envs: int = PHASE1_N_ENVS
    learning_rate: float = PHASE1_LEARNING_RATE
    seed: int = SEED


DEFAULT_TRAINING = TrainingConfig()