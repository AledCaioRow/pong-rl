# SDD — Evaluation Scripts (`evaluation/`)

**Status:** Implemented in this repo (`evaluation/eval_agent.py`, `eval_report.py`). `SmoothActionWrapper.last_executed_action` supports motor / action-change stats.

## Context

No formal eval exists yet — performance is judged by eyeballing TensorBoard
curves and playing `play_human_vs_bot.py`.  This SDD adds headless evaluation
that runs N games against the rule-based opponent, collects stats, and prints
a summary.  Covers Phase 1 (competent agent) and Phase 2 (margin targeting).

The eval scripts sit **outside** the training loop — they load a saved `.zip`
checkpoint and play full games with `deterministic=True` prediction.

---

## Architecture fit

```
config.py
    ↓
envs/pong_env.py
envs/margin_env.py
envs/smooth_action_wrapper.py   ← optional, if --hold-steps > 1
    ↓
evaluation/eval_agent.py        ← run N games, collect per-game stats
evaluation/eval_report.py       ← aggregate stats, print/save summary
    ↓
models/pong_competent.zip
models/pong_margin_targeting.zip
```

Training scripts unchanged.  Eval imports `PongEnv`, `MarginTargetingWrapper`,
optionally `SmoothActionWrapper`, and loads checkpoints via SB3 `PPO.load()`.

---

## Specification

### `evaluation/eval_agent.py`

Single entry point.  Loads a checkpoint, runs N episodes, returns structured
per-game data.

**CLI interface:**

```
py evaluation/eval_agent.py \
    --model models/pong_competent.zip \
    --games 50 \
    --phase 1 \
    --hold-steps 1 \
    --opponent rule_based \
    --seed 42 \
    --out evaluation/results/phase1_50g.json
```

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | (required) | Path to `.zip` checkpoint |
| `--games` | `config.EVAL_GAMES` (50) | Number of full episodes (first-to-`MAX_SCORE`) |
| `--phase` | `1` | `1` = raw `PongEnv`, `2` = `MarginTargetingWrapper(PongEnv())` |
| `--hold-steps` | `1` | `SmoothActionWrapper` when `> 1` (1 = off) |
| `--opponent` | `rule_based` | Opponent type passed to `PongEnv` |
| `--seed` | `None` | Optional seed for reproducibility |
| `--out` | `None` | Optional JSON output path.  If omitted, prints JSON to stdout only. If set, prints a one-line confirmation. |
| `--deterministic` / `--no-deterministic` | `config.EVAL_DETERMINISTIC` | SB3 `model.predict(deterministic=...)` |

**Per-game data:** see JSON schema in SDD original; `action_changes` counts **motor** (post–`SmoothActionWrapper`) transitions.

---

### `evaluation/eval_report.py`

```
py evaluation/eval_report.py --input evaluation/results/phase1_50g.json
py evaluation/eval_report.py --compare evaluation/results/p1_raw.json evaluation/results/p1_smooth3.json
```

**Aggregate stats:** win rate, mean scores/margin, rally stats (mean/median/max, bins), motor smoothness (`mean_action_change_rate`), Phase 2 margin error stats when `phase==2`.

**`--format`:** `table` (default), `json`, `md`.

**`--compare`:** side-by-side numeric deltas (table format).

---

### `evaluation/results/`

Gitignored; JSON outputs from `--out`.

### `evaluation/__init__.py`

Package marker.

### `config.py`

`EVAL_GAMES`, `EVAL_DETERMINISTIC`.

### `.gitignore`

`evaluation/results/`


## Typical workflow

```powershell
# 1. Train (quick, already works)
py training/train_phase1.py --quick
py training/train_phase2.py --quick

# 2. Eval Phase 1 — no smoothing
py evaluation/eval_agent.py --model models/pong_competent.zip --phase 1 --games 50 --out evaluation/results/p1_raw.json

# 3. Eval Phase 1 — with smoothing
py evaluation/eval_agent.py --model models/pong_competent.zip --phase 1 --games 50 --hold-steps 3 --out evaluation/results/p1_smooth3.json

# 4. Compare
py evaluation/eval_report.py --compare evaluation/results/p1_raw.json evaluation/results/p1_smooth3.json

# 5. Eval Phase 2
py evaluation/eval_agent.py --model models/pong_margin_targeting.zip --phase 2 --games 50 --out evaluation/results/p2_raw.json

# 6. Report
py evaluation/eval_report.py --input evaluation/results/p2_raw.json --format md
```

---

## What this gives you

- **Numbers instead of vibes** — win rate, margin error, rally stats per run.
- **Smoothing validation** — compare action change rate with/without wrapper,
  confirm win rate doesn't tank.
- **Phase 2 margin accuracy** — how close does the agent land to the sampled
  target margin?
- **Repeatability** — seeded runs, JSON output, side-by-side comparison.
- **Write-up ready** — `--format md` gives you tables to paste straight into
  your report or dissertation.
