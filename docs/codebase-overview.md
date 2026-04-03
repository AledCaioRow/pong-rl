# Pong RL — codebase overview

Map of the repository: what each part does and how it fits together.

**Working directory:** run scripts from the **project root** (`Pong RL/`) so `import config` and `from envs…` resolve.

---

## Architecture (data flow)

```
config.py               ← shared numbers (court, paddle, ball, training, smoothing)
       ↓
envs/pong_env.py        ← Gymnasium MDP: physics, observations, Phase-1-style rewards
       ↓
envs/margin_env.py      ← Optional wrapper: sampled target margin + Phase-2 rewards
       ↓
envs/smooth_action_wrapper.py  ← Inference-only motor filter (hold, reversal penalty, jitter)
       ↓
training/train_phase1.py → PPO → models/pong_competent.zip
training/train_phase2.py → loads competent → PPO → models/pong_margin_targeting.zip
       ↓
evaluation/eval_agent.py   ← headless N-game eval, optional smoothing, JSON output
evaluation/eval_report.py  ← aggregate / compare eval runs
       ↓
play_human_vs_bot.py       ← Tkinter human vs PPO (profiles, multi-checkpoint, mouse/keys)
```

**Design spec:** `pong_rl_spec.md` describes the full vision (Strategist, cognitive difficulty, Phase 3, eval metrics). Not all sections are implemented as code yet.

---

## Root files

| File | Purpose |
|------|---------|
| **`config.py`** | All shared constants: `MAX_SCORE`, court size, paddle/ball geometry and speeds, rally speedup (`RALLY_SPEEDUP_PER_HIT`, `RALLY_SPEEDUP_VELOCITY_CAP_MULT`), rule-based opponent noise, action smoothing (`ACTION_HOLD_STEPS`, `ACTION_REVERSAL_HOLD_STEPS`, `ACTION_HOLD_JITTER`), Phase 1/2 timestep defaults and quick-run sizes, model save paths, eval defaults. Imported by `envs/`, `training/`, and `evaluation/`. |
| **`requirements.txt`** | `gymnasium`, `stable-baselines3`, `torch`, `numpy`, `tensorboard`. |
| **`requirements-render.txt`** | Core deps plus optional **pygame**; human play uses **Tkinter** in `play_human_vs_bot.py`. |
| **`pong_rl_spec.md`** | Full product/technical specification. |
| **`play_human_vs_bot.py`** | **Human vs PPO:** you control the **left** paddle, PPO plays the **right**. Supports **profiles** (`--profile`), multi-checkpoint switching (M key or buttons), mouse/keyboard toggle, `--ball-speed-scale` (default 0.68 for human play), `--paddle-speed-scale`, `--rally-speedup-per-hit`, `--hold-steps`, `--fps`. Without `--model`, you play the right paddle against a `rule_based` or `idle` bot instead. |
| **`.python-version`** | Python version hint for pyenv-style tooling. |
| **`.gitignore`** | venvs, caches, `models/*.zip`, `training/logs/`, `evaluation/results/`, etc. |
| **`models/`** | Checkpoints: `pong_competent.zip` (Phase 1), `pong_margin_targeting.zip` (Phase 2), plus any user-renamed copies. Often gitignored — back up important runs. |

---

## `docs/`

| File | Purpose |
|------|---------|
| **`action-smoothing-options.md`** | When/how to reduce jittery discrete actions (repeat, sticky, reward penalty, post-hoc wrapper, cognitive layer); retrain implications; prompt template. |
| **`codebase-overview.md`** | This file: high-level codebase map. |
| **`evaluation-sdd.md`** | SDD for `evaluation/eval_agent.py` and `eval_report.py`: CLI interface, per-game data schema, typical workflow. |
| **`claude-prompt-phase2-training.md`** | Ready-to-paste prompt + context for asking Claude about Phase 2 margin targeting in a separate chat. |

---

## `envs/` — Gymnasium

| File | Purpose |
|------|---------|
| **`__init__.py`** | Exports `PongEnv`, `MarginTargetingWrapper`, `SmoothActionWrapper`, `ActionSmoother`. |
| **`pong_env.py`** | **Core env:** 10-D float observation (ball pos/vel, paddles, scores, `target_margin`, margin), `Discrete(3)` actions (stay/up/down), first-to-`MAX_SCORE`, wall/paddle physics with rally speedup (`RALLY_SPEEDUP_PER_HIT` multiplier, capped at `RALLY_SPEEDUP_VELOCITY_CAP_MULT`× base max), `info["rally_hits"]` on scoring steps. `paddle_speed_scale`, `ball_speed_scale`, `rally_speedup_per_hit` constructor args for play vs training. |
| **`margin_env.py`** | **`MarginTargetingWrapper`:** samples `target_margin` on reset; optional random opponent noise mix from `PHASE2_OPPONENT_NOISE_LEVELS`; replaces rewards with margin shaping + rally bonus + terminal proximity bonus. Wraps `PongEnv`. |
| **`smooth_action_wrapper.py`** | **Inference-only motor filter** (not used during training). **`SmoothActionWrapper`** (Gym wrapper) and **`ActionSmoother`** (standalone callable). Three constraints: (1) **base hold** — action must be held `ACTION_HOLD_STEPS` frames before switching, (2) **reversal penalty** — up↔down switches require `ACTION_REVERSAL_HOLD_STEPS` frames (~133 ms at 60 FPS), (3) **random jitter** — ±`ACTION_HOLD_JITTER` frames per hold to avoid metronomic patterns. |

---

## `training/` — Stable-Baselines3 PPO

| File | Purpose |
|------|---------|
| **`__init__.py`** | Package marker. |
| **`train_phase1.py`** | `make_vec_env` + **PPO** from scratch on `PongEnv(opponent=rule_based)`. Saves `models/pong_competent`. Flags: `--quick`, `--timesteps`, `--n-envs`, TensorBoard logdir, `--eval-games`, etc. Default 2M timesteps. |
| **`train_phase2.py`** | **Loads** Phase 1 zip, vectorized **`MarginTargetingWrapper(PongEnv())`**, continues PPO, saves margin checkpoint. Needs `pong_competent.zip`. Default 5M timesteps. |
| **`output_model_path.py`** | Resolves where SB3 should save a checkpoint: CLI path, quick-run default, or interactive prompt. Handles path normalization. |
| **`clean_artifacts.py`** | Remove generated logs / eval outputs / optional model zips. `--logs`, `--eval-results`, `--models`, `--dry-run`, `--yes`. |
| **`.gitkeep`** | Keeps `training/` tracked when logs are gitignored. |

**TensorBoard:** use `py -m tensorboard.main --logdir training/logs/phase1_ppo` (or `phase2_ppo`). `py -m tensorboard` may fail on some Python installs.

---

## `evaluation/` — headless metrics

| File | Purpose |
|------|---------|
| **`__init__.py`** | Package marker. |
| **`eval_agent.py`** | Load `PPO` `.zip`, run **`--games`** full episodes (`--phase` 1 or 2), optional **`SmoothActionWrapper`** via `--hold-steps`, `--paddle-speed-scale`, `--ball-speed-scale`, `--deterministic` / `--no-deterministic`. JSON to **`--out`** or stdout. |
| **`eval_report.py`** | Summarize one JSON (`--input`) or **`--compare`** two runs; `--format table|json|md`. Aggregates win rate, scores, margin error, rally stats, motor smoothness. |
| **`results/`** | Output directory (gitignored). |

---

## `player_profiles/` — per-player playtest configs

Each subfolder is a profile name (used with `--profile <name>`). Contains a `profile.json`:

```json
{
  "model": "models/Ping.zip",
  "label": "Display Name (optional)",
  "no_auto_pair": false
}
```

| Key | Required | Purpose |
|-----|----------|---------|
| `model` | yes | Path to PPO `.zip` checkpoint (relative to repo root). |
| `label` | no | Display name shown in Tkinter title bar. Defaults to folder name. |
| `no_auto_pair` | no | If `true`, skips auto-discovery of other `models/*.zip` files and phase pairing. |

Created automatically on first `--profile` run if the folder doesn't exist.

---

## Spec / planned (not all present in repo)

| Planned (see `pong_rl_spec.md`) | Status |
|----------------------------------|--------|
| `envs/cognitive_difficulty.py` | Not added |
| `training/train_phase3.py` | Not added |
| `frontend/pong_renderer.py` | Not added (Tkinter + env `rgb_array` exist) |

---

## Typical commands

```powershell
Set-Location "c:\Users\aled_\Downloads\Pong RL"
py -m pip install -r requirements.txt

# --- Training ---
py training/train_phase1.py --quick
py training/train_phase2.py --quick
py -m tensorboard.main --logdir training/logs/phase1_ppo

# --- Human play ---
py play_human_vs_bot.py                                          # right paddle vs rule-based bot
py play_human_vs_bot.py --model models/pong_competent.zip        # left paddle vs PPO (auto-pairs other .zips)
py play_human_vs_bot.py --no-auto-pair --profile Ping            # left paddle vs single profile model only

# --- Evaluation ---
py evaluation/eval_agent.py --model models/pong_competent.zip --phase 1 --games 50 --out evaluation/results/p1.json
py evaluation/eval_agent.py --model models/pong_competent.zip --phase 1 --games 50 --hold-steps 3 --out evaluation/results/p1_smooth.json
py evaluation/eval_report.py --input evaluation/results/p1.json
py evaluation/eval_report.py --compare evaluation/results/p1.json evaluation/results/p1_smooth.json

# --- Cleanup ---
py training/clean_artifacts.py --logs --eval-results --dry-run
```

---

## `sys.path` convention

`training/*.py` and `evaluation/*.py` insert the **repo root** on `sys.path` so `import config` and `from envs.pong_env import PongEnv` work when you run e.g. `python training/train_phase1.py`.
