"""
Play the training environment locally with Tkinter (stdlib on Windows).

Terminal (PowerShell) from any folder:
    Set-Location "c:\\Users\\aled_\\Downloads\\Pong RL"

No neural net (you = right paddle vs rule-based bot):
    py play_human_vs_bot.py

All checkpoints in one session (recommended): pass a `--model` path **without** `--no-auto-pair`.
The script then loads **every** `models/*.zip` (and may add the paired default phase `.zip`), so
you can switch with the top bar or **M**:
    py play_human_vs_bot.py --model models/pong_competent.zip

Current `models/*.zip` included in that merge (after training, list files and extend this list):
    PowerShell: Get-ChildItem models\\*.zip
    - models/pong_competent.zip
    - models/pong_margin_targeting.zip

Single checkpoint only (`--no-auto-pair` turns off the `models/*.zip` merge and phase pairing):
    py play_human_vs_bot.py --no-auto-pair --model models/pong_competent.zip
    py play_human_vs_bot.py --no-auto-pair --model models/pong_margin_targeting.zip

Multiple explicit paths (merge still adds any other `models/*.zip` unless you pass `--no-auto-pair`):
    py play_human_vs_bot.py --model models/pong_competent.zip --model models/pong_margin_targeting.zip

Two modes:
- **Default:** you control the **right** paddle vs rule-based (or idle) bot on the **left**
  (same sides as Phase 1 training, but you substitute for the RL policy).
- **With --model:** load a PPO ``.zip``; you control the **left** paddle vs the policy on the **right**
  (matches how the network was trained: agent = right).

Click the court to toggle **KEYS** ↔ **MOUSE**. Optional `--hold-steps` applies action hold
(SmoothActionWrapper in default mode; ActionSmoother on your inputs in `--model` mode).

Also from project root with `python`:
    python play_human_vs_bot.py
    python play_human_vs_bot.py --model models/pong_competent.zip
  Use `--no-auto-pair` to disable merging extra `models/*.zip`. With 2+ loaded checkpoints,
  use **Previous model** / **Next model** or **M**. The window uses a wide layout with a centered
  rectangular playfield.
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


def _try_utf8_stdio() -> None:
    """Windows consoles often use cp1252; UTF-8 avoids UnicodeEncodeError on --help text."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def _safe_print(*args: object, **kwargs: object) -> None:
    """Some IDE / GUI launchers set sys.stdout to None; avoid crashing before the Tk window opens."""
    if sys.stdout is None:
        return
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except (OSError, TypeError, UnicodeEncodeError):
        pass


def _safe_eprint(*args: object, **kwargs: object) -> None:
    if sys.stderr is None:
        return
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except (OSError, TypeError, UnicodeEncodeError):
        pass


def main() -> None:
    _try_utf8_stdio()

    parser = argparse.ArgumentParser(description="Human playtest (Tkinter).")
    parser.add_argument(
        "--model",
        action="append",
        dest="model_paths",
        metavar="PATH",
        default=None,
        help="PPO checkpoint .zip — you play LEFT vs policy on RIGHT. "
        "Pass multiple times to cycle checkpoints in-game (buttons or M).",
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
        default=cfg.FPS,
        help=f"Display / physics tick rate (default {cfg.FPS}).",
    )
    parser.add_argument(
        "--paddle-speed-scale",
        type=float,
        default=1.0,
        help="Multiplies paddle speed (default 1.0 = shared baseline).",
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
        default=1.0,
        help="Ball speed scale (default 1.0 = shared baseline).",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=cfg.ACTION_HOLD_STEPS,
        help="Hold raw action N frames before switching (1 = off / no smoothing).",
    )
    parser.add_argument(
        "--no-auto-pair",
        action="store_true",
        help="Only load paths you pass: no phase-2 pairing, no extra models/*.zip merge.",
    )
    args = parser.parse_args()

    os.chdir(_ROOT)

    def _as_zip(p: Path) -> Path:
        return p if p.suffix.lower() == ".zip" else p.with_suffix(".zip")

    raw_paths = args.model_paths or []
    model_paths: list[Path] = []
    seen_resolved: set[Path] = set()
    for s in raw_paths:
        p = Path(s)
        r = p.resolve()
        if r in seen_resolved:
            continue
        seen_resolved.add(r)
        model_paths.append(p)

    if (
        len(model_paths) == 1
        and not args.no_auto_pair
    ):
        user_r = model_paths[0].resolve()
        for rel in (cfg.PHASE2_MODEL_PATH, cfg.PHASE1_MODEL_PATH):
            cand = _as_zip(Path(rel))
            cr = cand.resolve()
            if cr != user_r and cand.is_file():
                seen_resolved.add(cr)
                model_paths.append(cand)
                break

    # Any other checkpoints in models/ → same session can switch without listing each CLI path.
    if model_paths and not args.no_auto_pair:
        models_dir = Path(_ROOT) / "models"
        if models_dir.is_dir():
            for p in sorted(models_dir.glob("*.zip"), key=lambda x: x.name.lower()):
                r = p.resolve()
                if r not in seen_resolved:
                    seen_resolved.add(r)
                    model_paths.append(p)

    vs_ppo = len(model_paths) > 0
    if vs_ppo:
        n_m = len(model_paths)
        _safe_print(
            f"[play] PPO mode: {n_m} checkpoint(s). "
            + (
                "Use the top bar or M to switch."
                if n_m > 1
                else "Only one .zip — add another under models/ or pass --model twice."
            ),
        )
    else:
        _safe_print(
            "[play] You = right paddle vs rule-based bot (no neural net). "
            "Run with --model models/pong_competent.zip to play vs PPO and switch models.",
        )
    model = None
    human_smoother = None
    model_index = 0

    if vs_ppo:
        for p in model_paths:
            if not p.is_file():
                _safe_eprint(f"Model not found: {p.resolve()}")
                sys.exit(1)
        if args.opponent != "rule_based":
            _safe_print(
                "[play] Note: --opponent is ignored when --model is set (you vs PPO).",
            )
        from stable_baselines3 import PPO

        model = PPO.load(str(model_paths[model_index]))
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

    # Wide window with a centered rectangular gameplay field.
    window_w_px, window_h_px = 1200.0, 700.0
    field_w_px, field_h_px = 1030.0, 560.0
    field_left_px = (window_w_px - field_w_px) / 2.0
    field_top_px = (window_h_px - field_h_px) / 2.0
    field_right_px = field_left_px + field_w_px
    field_bottom_px = field_top_px + field_h_px

    root = tk.Tk()
    root.geometry(f"{int(window_w_px)}x{int(window_h_px)}")
    root.minsize(int(window_w_px), int(window_h_px))

    def window_title() -> str:
        t = "Pong RL — PPO vs human" if vs_ppo else "Pong RL — human vs bot"
        if vs_ppo and model_paths:
            t = f"{t} ({model_paths[model_index].name})"
        return t

    root.title(window_title())
    canvas = tk.Canvas(
        root,
        width=int(window_w_px),
        height=int(window_h_px),
        bg="#000000",
        highlightthickness=2,
        highlightbackground="#f2f2f2",
    )

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
            # Map court x to the rectangular field width.
            return field_left_px + (x * field_w_px)

        def to_cy(y: float) -> float:
            # Map court y to the rectangular field height.
            return field_top_px + ((1.0 - y) * field_h_px)

        # Minimal classic Pong field.
        canvas.create_rectangle(
            0,
            0,
            window_w_px,
            window_h_px,
            fill="#000000",
            outline="",
        )
        canvas.create_rectangle(
            field_left_px,
            field_top_px,
            field_right_px,
            field_bottom_px,
            fill="#000000",
            outline="#000000",
            width=0,
        )
        mid_x = (field_left_px + field_right_px) * 0.5
        canvas.create_line(
            mid_x,
            field_top_px,
            mid_x,
            field_bottom_px,
            fill="#f2f2f2",
            width=4,
            dash=(9, 12),
        )

        half_w = cfg.PADDLE_HALF_WIDTH * field_w_px
        half_h = cfg.PADDLE_HALF_HEIGHT * field_h_px
        br = cfg.BALL_RADIUS * min(field_w_px, field_h_px)

        bx, by = to_cx(c.ball_x), to_cy(c.ball_y)
        half_ball_w = br * 0.48
        half_ball_h = br * 1.55
        canvas.create_rectangle(
            bx - half_ball_w,
            by - half_ball_h,
            bx + half_ball_w,
            by + half_ball_h,
            fill="#f2f2f2",
            outline="#f2f2f2",
            width=0,
        )

        for px, py in ((c._human_x, c.human_y), (c._agent_x, c.agent_y)):
            cx, cy = to_cx(px), to_cy(py)
            canvas.create_rectangle(
                cx - half_w,
                cy - half_h,
                cx + half_w,
                cy + half_h,
                fill="#f2f2f2",
                outline="#f2f2f2",
                width=0,
            )

        sa, sh = int(obs[6]), int(obs[7])
        top_y = field_top_px + 26
        canvas.create_text(
            mid_x - 80,
            top_y,
            text=str(sh),
            fill="#f2f2f2",
            font=("Consolas", 28, "bold"),
        )
        canvas.create_text(
            mid_x + 80,
            top_y,
            text=str(sa),
            fill="#f2f2f2",
            font=("Consolas", 28, "bold"),
        )
        if done:
            canvas.create_text(
                mid_x,
                field_bottom_px - 20,
                text="GAME OVER - R to reset, Esc to quit",
                fill="#f2f2f2",
                font=("Consolas", 12, "bold"),
            )

    def cycle_model(delta: int) -> None:
        nonlocal model, obs, done, model_index
        if not vs_ppo or len(model_paths) < 2:
            return
        from stable_baselines3 import PPO

        model_index = (model_index + delta) % len(model_paths)
        model = PPO.load(str(model_paths[model_index]))
        obs, _ = env.reset(seed=args.seed)
        done = False
        if human_smoother is not None:
            human_smoother.reset()
        root.title(window_title())
        redraw()

    btn_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
    hint_font = tkfont.Font(family="Segoe UI", size=9)
    model_bar = tk.Frame(root, bg="#000000", pady=6, padx=4)
    if vs_ppo:
        multi_ckpt = len(model_paths) > 1
        model_bar.pack(fill="x", side=tk.TOP)
        hint = (
            "Switch PPO · M = next"
            if multi_ckpt
            else "Need 2+ .zip in models/ (or --model A --model B) to switch"
        )

        def _btn(**kw: object) -> tk.Button:
            b = tk.Button(model_bar, **kw)
            try:
                b.configure(
                    relief=tk.RAISED,
                    borderwidth=2,
                    highlightthickness=1,
                    highlightbackground="#606078",
                )
            except tk.TclError:
                pass
            return b

        prev_b = _btn(
            text="◀  Previous model",
            command=lambda: cycle_model(-1),
            font=btn_font,
            bg="#1a1a1a",
            fg="#ffffff",
            activebackground="#2a2a2a",
            activeforeground="#ffffff",
            padx=14,
            pady=8,
            cursor="hand2" if multi_ckpt else "arrow",
            takefocus=True,
            state=tk.NORMAL if multi_ckpt else tk.DISABLED,
        )
        prev_b.pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(
            model_bar,
            text=hint,
            bg="#000000",
            fg="#d0d0d0",
            font=hint_font,
        ).pack(side=tk.LEFT, expand=True)
        next_b = _btn(
            text="Next model  ▶",
            command=lambda: cycle_model(1),
            font=btn_font,
            bg="#1a1a1a",
            fg="#ffffff",
            activebackground="#2a2a2a",
            activeforeground="#ffffff",
            padx=14,
            pady=8,
            cursor="hand2" if multi_ckpt else "arrow",
            takefocus=True,
            state=tk.NORMAL if multi_ckpt else tk.DISABLED,
        )
        next_b.pack(side=tk.RIGHT, padx=(8, 4))

    canvas.pack()

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

    def on_canvas_click(e: tk.Event) -> None:
        nonlocal control_mode, mouse_target_y
        # Keep click-to-toggle scoped to the playable rectangle.
        if not (
            field_left_px <= float(e.x) <= field_right_px
            and field_top_px <= float(e.y) <= field_bottom_px
        ):
            return
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
        y_px = min(max(float(e.y), field_top_px), field_bottom_px)
        y_rel = (y_px - field_top_px) / max(1.0, field_h_px)
        y_court = 1.0 - y_rel
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
        if vs_ppo and len(model_paths) > 1 and ks in ("m", "M"):
            cycle_model(1)
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
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        _try_utf8_stdio()
        import traceback

        try:
            traceback.print_exc()
        except OSError:
            pass
        raise SystemExit(1)
