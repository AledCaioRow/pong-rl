"""
Play the training environment locally: you control the RIGHT agent paddle
against the rule-based bot on the LEFT (same physics as Phase 1).

Uses Tkinter only (stdlib on Windows) — no pygame required.

- Click the court to toggle **KEYS** ↔ **MOUSE** (mouse = follow paddle to cursor Y).
- Slower paddle and **slower ball** than training defaults (play-only scales).
- Ball is drawn as a **tall rectangle**; collisions are still the env’s circle.
- Optional **action hold** (`--hold-steps`) smooths rapid up/down flips (inference-only).

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
import tkinter as tk
from tkinter import font as tkfont

import config as cfg
from envs.pong_env import PongEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Human vs rule-based bot (PongEnv).")
    parser.add_argument(
        "--opponent",
        choices=("rule_based", "idle"),
        default="rule_based",
        help="idle = left paddle does not move (easiest).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Reset seed (optional).")
    parser.add_argument(
        "--fps",
        type=int,
        default=40,
        help="Display / physics tick rate (lower = calmer; default 40).",
    )
    parser.add_argument(
        "--paddle-speed-scale",
        type=float,
        default=0.58,
        help="Multiplies paddle speed vs training (default 0.58 = smoother).",
    )
    parser.add_argument(
        "--mouse-deadzone",
        type=float,
        default=0.017,
        help="Court-normalised Y deadzone for mouse follow (larger = less jitter).",
    )
    parser.add_argument(
        "--ball-speed-scale",
        type=float,
        default=0.38,
        help="Ball speed vs training (default 0.38 ≈ easier reactions).",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=cfg.ACTION_HOLD_STEPS,
        help="Action-hold frames before a new action is applied (1=no SmoothActionWrapper).",
    )
    args = parser.parse_args()

    os.chdir(_ROOT)

    env = PongEnv(
        opponent=args.opponent,
        render_mode=None,
        paddle_speed_scale=args.paddle_speed_scale,
        ball_speed_scale=args.ball_speed_scale,
    )
    if args.hold_steps > 1:
        from envs.smooth_action_wrapper import SmoothActionWrapper

        env = SmoothActionWrapper(env, hold_steps=args.hold_steps)
    obs, _ = env.reset(seed=args.seed)

    w_px, h_px = 400.0, 400.0
    root = tk.Tk()
    root.title("Pong RL — click court: KEYS ↔ MOUSE")
    canvas = tk.Canvas(
        root,
        width=int(w_px),
        height=int(h_px),
        bg="#181820",
        highlightthickness=2,
        highlightbackground="#2a2a36",
    )
    canvas.pack()

    status = tk.Label(
        root,
        text="",
        anchor="w",
        justify="left",
        bg="#101016",
        fg="#dcdce6",
        padx=8,
        pady=8,
        height=4,
        font=tkfont.Font(family="Segoe UI", size=11),
    )
    status.pack(fill="x")

    held: set[str] = set()
    done = False
    control_mode = "keys"  # "keys" | "mouse"
    mouse_target_y: float | None = None

    def redraw() -> None:
        canvas.delete("all")
        c = env.unwrapped
        assert isinstance(c, PongEnv)

        def to_cx(x: float) -> float:
            return x * w_px

        def to_cy(y: float) -> float:
            return (1.0 - y) * h_px

        half_w = cfg.PADDLE_HALF_WIDTH * w_px
        half_h = cfg.PADDLE_HALF_HEIGHT * h_px
        br = cfg.BALL_RADIUS * min(w_px, h_px)

        # Ball first; tall “portrait” rectangle (visual only — sim uses circular hit).
        bx, by = to_cx(c.ball_x), to_cy(c.ball_y)
        half_ball_w = br * 0.48
        half_ball_h = br * 1.55
        canvas.create_rectangle(
            bx - half_ball_w,
            by - half_ball_h,
            bx + half_ball_w,
            by + half_ball_h,
            fill="#fa5a5a",
            outline="#7a2020",
            width=2,
        )

        for px, py in ((c._human_x, c.human_y), (c._agent_x, c.agent_y)):
            cx, cy = to_cx(px), to_cy(py)
            canvas.create_rectangle(
                cx - half_w,
                cy - half_h,
                cx + half_w,
                cy + half_h,
                fill="#e8e8f0",
                outline="#505060",
                width=2,
            )

        sa, sh = int(obs[6]), int(obs[7])
        tm = int(round(float(obs[8])))
        margin = int(round(float(obs[9])))
        over = "  |  GAME OVER — press R" if done else ""
        mode_hint = "KEYS (↑/↓ or W/S)" if control_mode == "keys" else "MOUSE (move on court)"
        status.config(
            text=(
                f"Agent {sa}   Human {sh}   |   target_margin {tm}   margin {margin}{over}\n"
                f"Mode: {control_mode.upper()} — {mode_hint}\n"
                f"Click the playfield to toggle keys/mouse · R reset · Esc quit"
            )
        )

    def action_for_mouse(target_y: float) -> int:
        c = env.unwrapped
        assert isinstance(c, PongEnv)
        ay = c.agent_y
        dz = args.mouse_deadzone
        if target_y > ay + dz:
            return 1
        if target_y < ay - dz:
            return 2
        return 0

    def apply_action(action: int) -> None:
        nonlocal obs, done
        if done:
            return
        obs, _r, terminated, _trunc, _info = env.step(action)
        done = bool(terminated)
        redraw()

    def tick() -> None:
        if not running["ok"]:
            return
        if not done:
            if control_mode == "mouse" and mouse_target_y is not None:
                a = action_for_mouse(mouse_target_y)
            else:
                a = 0
                if "Up" in held or "w" in held:
                    a = 1
                elif "Down" in held or "s" in held:
                    a = 2
            apply_action(a)
        root.after(ms_per_frame, tick)

    ms_per_frame = max(1, int(round(1000.0 / max(12, args.fps))))

    running = {"ok": True}

    def on_canvas_click(_e: tk.Event) -> None:
        nonlocal control_mode, mouse_target_y
        control_mode = "mouse" if control_mode == "keys" else "keys"
        held.clear()
        if control_mode == "keys":
            mouse_target_y = None
        canvas.focus_set()
        redraw()

    def on_motion(e: tk.Event) -> None:
        nonlocal mouse_target_y
        if control_mode != "mouse":
            return
        c = env.unwrapped
        assert isinstance(c, PongEnv)
        y_court = 1.0 - (float(e.y) / h_px)
        y_court = max(c._paddle_min_y, min(c._paddle_max_y, y_court))
        mouse_target_y = y_court

    def on_key_down(e: tk.Event) -> None:
        nonlocal obs, done
        ks = e.keysym
        if ks == "Escape":
            running["ok"] = False
            root.destroy()
            return
        if ks in ("r", "R"):
            obs, _ = env.reset(seed=args.seed)
            done = False
            redraw()
            return
        if control_mode == "keys":
            if ks in ("Up", "Down", "w", "s"):
                held.add(ks)
            if ks == "W":
                held.add("w")
            if ks == "S":
                held.add("s")

    def on_key_up(e: tk.Event) -> None:
        ks = e.keysym
        if ks in ("Up", "Down", "w", "s"):
            held.discard(ks)
        if ks == "W":
            held.discard("w")
        if ks == "S":
            held.discard("s")

    canvas.bind("<Button-1>", on_canvas_click)
    canvas.bind("<Motion>", on_motion)

    root.bind("<KeyPress>", on_key_down)
    root.bind("<KeyRelease>", on_key_up)

    def quit_app() -> None:
        running["ok"] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", quit_app)

    canvas.focus_set()
    redraw()
    tick()
    root.mainloop()

    env.close()


if __name__ == "__main__":
    main()
