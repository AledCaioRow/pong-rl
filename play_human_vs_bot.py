"""
Play the training environment locally with Tkinter (stdlib on Windows).

Two modes:
- **Default:** you control the **right** paddle vs rule-based (or idle) bot on the **left**
  (same sides as Phase 1 training, but you substitute for the RL policy).
- **With --model:** load a PPO `.zip`; you control the **left** paddle vs the policy on the **right**
  (matches how the network was trained: agent = right).

Click the court to toggle **KEYS** ↔ **MOUSE**. Optional `--hold-steps` applies action hold
(SmoothActionWrapper in default mode; ActionSmoother on your inputs in --model mode).

Run from project root:
    python play_human_vs_bot.py
    python play_human_vs_bot.py --model models/pong_competent.zip
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import tkinter as tk
from tkinter import font as tkfont

import config as cfg
from envs.pong_env import PongEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Human playtest (Tkinter).")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="PATH",
        help="PPO checkpoint .zip — you play LEFT vs policy on RIGHT (e.g. models/pong_competent.zip).",
    )
    parser.add_argument(
        "--opponent",
        choices=("rule_based", "idle"),
        default="rule_based",
        help="Left paddle when not using --model. Ignored if --model is set.",
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
        help="Hold raw action N frames before switching (1 = off / no smoothing).",
    )
    args = parser.parse_args()

    os.chdir(_ROOT)

    vs_ppo = args.model is not None
    model = None
    human_smoother = None

    if vs_ppo:
        path = Path(args.model)
        if not path.is_file():
            print(f"Model not found: {path.resolve()}", file=sys.stderr)
            sys.exit(1)
        if args.opponent != "rule_based":
            print("[play] Note: --opponent is ignored when --model is set (you vs PPO).")
        from stable_baselines3 import PPO

        model = PPO.load(str(path))
        base = PongEnv(
            opponent="keyboard",
            render_mode=None,
            paddle_speed_scale=args.paddle_speed_scale,
            ball_speed_scale=args.ball_speed_scale,
        )
        env = base
        if args.hold_steps > 1:
            from envs.smooth_action_wrapper import ActionSmoother

            human_smoother = ActionSmoother(hold_steps=args.hold_steps)
    else:
        base = PongEnv(
            opponent=args.opponent,
            render_mode=None,
            paddle_speed_scale=args.paddle_speed_scale,
            ball_speed_scale=args.ball_speed_scale,
        )
        env = base
        if args.hold_steps > 1:
            from envs.smooth_action_wrapper import SmoothActionWrapper

            env = SmoothActionWrapper(env, hold_steps=args.hold_steps)

    obs, _ = env.reset(seed=args.seed)
    model_basename = Path(args.model).name if vs_ppo else ""

    w_px, h_px = 400.0, 400.0
    root = tk.Tk()
    title = "Pong RL — PPO vs human" if vs_ppo else "Pong RL — human vs bot"
    if vs_ppo and model_basename:
        title = f"{title} ({model_basename})"
    root.title(title)
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
        height=5,
        font=tkfont.Font(family="Segoe UI", size=11),
    )
    status.pack(fill="x")

    held: set[str] = set()
    done = False
    control_mode = "keys"  # "keys" | "mouse"
    mouse_target_y: float | None = None

    def paddle_y_for_controls() -> float:
        c = env.unwrapped
        assert isinstance(c, PongEnv)
        return c.human_y if vs_ppo else c.agent_y

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
        if vs_ppo:
            who = "You = LEFT (Human on scoreboard) | PPO = RIGHT (Agent on scoreboard)"
        else:
            who = "You = RIGHT (Agent) | Bot = LEFT (rule_based or idle)"
        status.config(
            text=(
                f"{who}\n"
                f"Agent {sa}   Human {sh}   |   target_margin {tm}   margin {margin}{over}\n"
                f"Mode: {control_mode.upper()} — {mode_hint}\n"
                f"Click playfield: keys ↔ mouse · R reset · Esc quit"
            )
        )

    def action_for_mouse(target_y: float) -> int:
        c = env.unwrapped
        assert isinstance(c, PongEnv)
        py_ref = paddle_y_for_controls()
        dz = args.mouse_deadzone
        if target_y > py_ref + dz:
            return 1
        if target_y < py_ref - dz:
            return 2
        return 0

    def raw_human_action() -> int:
        if control_mode == "mouse" and mouse_target_y is not None:
            return action_for_mouse(mouse_target_y)
        if "Up" in held or "w" in held:
            return 1
        if "Down" in held or "s" in held:
            return 2
        return 0

    def apply_step_vs_ppo(raw_h: int) -> None:
        nonlocal obs, done
        if done or model is None:
            return
        ha = int(human_smoother(raw_h)) if human_smoother is not None else int(raw_h)
        ppo_a, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, _trunc, _info = env.step(
            int(ppo_a),
            human_action=int(ha),
        )
        done = bool(terminated)

    def apply_step_human_right(raw_a: int) -> None:
        nonlocal obs, done
        if done:
            return
        obs, _r, terminated, _trunc, _info = env.step(raw_a)
        done = bool(terminated)

    def tick() -> None:
        if not running["ok"]:
            return
        if not done:
            h = raw_human_action()
            if vs_ppo:
                apply_step_vs_ppo(h)
            else:
                apply_step_human_right(h)
            redraw()
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
            if human_smoother is not None:
                human_smoother.reset()
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
