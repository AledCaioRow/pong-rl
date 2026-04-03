"""
Play the training environment locally: you control the RIGHT agent paddle
against the rule-based bot on the LEFT (same physics as Phase 1).

Requires pygame:
    py -m pip install -r requirements-render.txt

Run from project root:
    python play_human_vs_bot.py
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse

import numpy as np
import pygame

import config as cfg
from envs.pong_env import PongEnv


def _frame_to_surface(frame: np.ndarray) -> pygame.Surface:
    """RGB uint8 (H, W, 3) -> pygame Surface."""
    h, w = frame.shape[0], frame.shape[1]
    buf = np.ascontiguousarray(frame)
    surf = pygame.image.frombuffer(buf.tobytes(), (w, h), "RGB")
    return surf.convert()


def main() -> None:
    parser = argparse.ArgumentParser(description="Human vs rule-based bot (PongEnv).")
    parser.add_argument(
        "--opponent",
        choices=("rule_based", "idle"),
        default="rule_based",
        help="idle = left paddle does not move (easiest).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Reset seed (optional).")
    args = parser.parse_args()

    os.chdir(_ROOT)

    pygame.init()
    env = PongEnv(opponent=args.opponent, render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)

    first = env.render()
    h, w = first.shape[0], first.shape[1]
    screen = pygame.display.set_mode((w, h + 40))
    pygame.display.set_caption(
        "Pong RL — YOU: right paddle | BOT: left | ↑/↓ or W/S | R reset | Esc quit"
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui", 22)

    done = False
    running = True
    while running:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset(seed=args.seed)
                    done = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = 2

        if not done and running:
            obs, _r, terminated, _trunc, _info = env.step(action)
            done = bool(terminated)

        frame = env.render()
        surf = _frame_to_surface(frame)
        screen.blit(surf, (0, 0))

        sa, sh = int(obs[6]), int(obs[7])
        tm = int(round(float(obs[8])))
        margin = int(round(float(obs[9])))
        bar = font.render(
            f"Agent {sa}  Human {sh}  |  target_margin {tm}  margin {margin}"
            + ("  [GAME OVER — press R]" if done else ""),
            True,
            (220, 220, 230),
        )
        screen.fill((16, 16, 22), pygame.Rect(0, h, w, 40))
        screen.blit(bar, (8, h + 8))

        pygame.display.flip()
        clock.tick(cfg.FPS)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
