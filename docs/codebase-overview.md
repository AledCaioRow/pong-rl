# Pong RL — codebase overview

Map of the repository: what each part does and how it fits together.

**Working directory:** run scripts from the **project root** (`Pong RL/`) so `import config` and `from envs…` resolve.

---

## Architecture (data flow)

```
config.py          ← shared numbers (court, paddle, ball, training caps)
       ↓
envs/pong_env.py   ← Gymnasium MDP: physics, observations, Phase-1-style rewards
       ↓
envs/margin_env.py ← Optional wrapper: sampled target margin + Phase-2 rewards
       ↓
training/train_phase1.py → PPO → models/pong_competent.zip
training/train_phase2.py → loads competent → PPO → models/pong_margin_targeting.zip
       ↓
play_human_vs_bot.py      ← Human vs bot (playtest; optional slower ball/paddle)
```

**Design spec:** `pong_rl_spec.md` describes the full vision (Strategist, cognitive difficulty, Phase 3, eval metrics). Not all sections are implemented as code yet.

---

## Root files

| File | Purpose |
|------|---------|
| **`config.py`** | **MAX_SCORE**, court size, paddle/ball geometry and speeds, Phase 1/2 timestep defaults and quick-run sizes (`PHASE1_QUICK_TIMESTEPS`, etc.), rule-based opponent noise, model save paths. Imported by `envs/` and `training/`. |
| **`requirements.txt`** | `gymnasium`, `stable-baselines3`, `torch`, `numpy`, `tensorboard`. |
| **`requirements-render.txt`** | Core deps plus optional **pygame**; human play uses **Tkinter** in `play_human_vs_bot.py`. |
| **`pong_rl_spec.md`** | Full product/technical specification. |
| **`play_human_vs_bot.py`** | **You** control the **right** paddle (same side as the RL **agent**); left is `rule_based` or `idle`. Tkinter UI; mouse/keys toggle; `--ball-speed-scale`, `--paddle-speed-scale`, `--fps`, etc. Defaults are **easier** than full-speed training env unless you match training scales. |
| **`.python-version`** | Python version hint for pyenv-style tooling. |
| **`.gitignore`** | venvs, caches, `models/*.zip`, `training/logs/`, etc. |
| **`models/`** | Checkpoints: `pong_competent.zip` (Phase 1), `pong_margin_targeting.zip` (Phase 2). Often ignored by git—back up important runs. |

---

## `docs/`

| File | Purpose |
|------|---------|
| **`action-smoothing-options.md`** | When/how to reduce jittery discrete actions (repeat, sticky, reward penalty, post-hoc wrapper, cognitive layer); retrain implications; prompt template. |
| **`codebase-overview.md`** | This file: high-level codebase map. |

---

## `envs/` — Gymnasium

| File | Purpose |
|------|---------|
| **`__init__.py`** | Exports `PongEnv`, `MarginTargetingWrapper`. |
| **`pong_env.py`** | **Core env:** 10-D float observation (ball, paddles, scores, `target_margin`, margin), `Discrete(3)` actions, first-to-30, wall/ball physics, paddle collisions, rally `info["rally_hits"]` on scoring steps. **`paddle_speed_scale`**, **`ball_speed_scale`** for play vs training speed. |
| **`margin_env.py`** | **`MarginTargetingWrapper`:** samples `target_margin` on reset; optional opponent noise mix; replaces rewards with margin shaping + rally bonus on points + terminal proximity bonus. Wraps `PongEnv`. |
| **`smooth_action_wrapper.py`** | **`SmoothActionWrapper`** / **`ActionSmoother`:** Option D inference-only action hold (reduces jittery flips). Used in `play_human_vs_bot.py` when `--hold-steps` > 1; **not** used in Phase 1/2 training. |

---

## `training/` — Stable-Baselines3 PPO

| File | Purpose |
|------|---------|
| **`train_phase1.py`** | `make_vec_env` + **PPO** from scratch on `PongEnv(opponent=rule_based)`. Saves `models/pong_competent`. Flags: `--quick`, `--timesteps`, `--n-envs`, TensorBoard path + import fallback, `--eval-games`, etc. |
| **`train_phase2.py`** | **Loads** Phase 1 zip, vectorized **`MarginTargetingWrapper(PongEnv())`**, continues PPO, saves margin checkpoint. Needs `pong_competent.zip`. |

**TensorBoard:** use `py -m tensorboard.main --logdir training/logs/phase1_ppo` (or `phase2_ppo`). `py -m tensorboard` may fail on some Python installs.

---

## Spec / planned (not all present in repo)

| Planned (see `pong_rl_spec.md`) | Status |
|----------------------------------|--------|
| `envs/cognitive_difficulty.py` | Not added |
| `training/train_phase3.py` | Not added |
| `evaluation/*.py` | Not added |
| `frontend/pong_renderer.py` | Not added (Tkinter + env `rgb_array` exist) |

---

## Typical commands

```powershell
Set-Location "c:\Users\aled_\Downloads\Pong RL"
py -m pip install -r requirements.txt
py training/train_phase1.py --quick
py training/train_phase2.py --quick
py play_human_vs_bot.py
py -m tensorboard.main --logdir training/logs/phase1_ppo
```

---

## `sys.path` convention

`training/*.py` inserts the **repo root** on `sys.path` so `import config` and `from envs.pong_env import PongEnv` work when you run `python training/train_phase1.py`.
