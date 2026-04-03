# Action smoothing for Pong RL (human-like paddle motion)

## Situation

The agent uses **discrete** actions each env step (stay / up / down). PPO often learns **bang-bang** control: flip up/down quickly because that maximizes reward, even though **humans** move more smoothly (momentum, hesitation, fewer direction changes).

This doc compares **when** to apply smoothing: before training, **as part of training**, or **after** training only.

---

## Options at a glance

| Approach | Where it lives | Retrain? | Pros | Cons |
|----------|----------------|----------|------|------|
| **A. Action repeat / frame-skip** | Inside `env.step()` (same action repeated *k* times per policy decision, or policy only queried every *k* steps) | Yes, if you change the trained MDP | Policy learns under **real** deployment dynamics; strong reduction of micro-jitter | Changes **effective** MDP (fewer decisions per rally); need consistent *k* everywhere (train + eval + prod) |
| **B. Sticky actions** | Env: probability (or hold) of **keeping** last action | Yes, if baked into env | Simple, Atari-style; reduces flip-flop | Hyperparameter tuning; can slightly slow reactions |
| **C. Reward: action-change penalty** | Add `-λ * 1[action ≠ prev]` or smooth penalty each step | Yes | Directly optimizes “smooth” without changing transition math as much as repeat | Can **hurt** score or margin objectives if λ too large; tune carefully with margin Phase 2 |
| **D. Post-hoc policy wrapper** | After `model.predict()`: debounce, majority vote over last *n* actions, or “only change if same raw action *m* steps” | **No** (quick to try) | Fast experiment; no training run | Policy wasn’t trained for filtered commands; may **worsen** play if too aggressive; can desync from what TensorBoard logged |
| **E. Cognitive / perception layer** (spec) | Noise, delay, wobble on **inputs** to policy, not visible physics | Optional / separate | Misses look **human**; can mask twitchy policy | Doesn’t remove twitchy **motor** if policy still oscillates; depends on Strategist / wrapper design |

---

## “Before”, “during”, or “after” training?

- **Before training** (as *design choice before long runs*):  
  Decide **whether** smoothing is part of the **official MDP**. If yes, implement **A or B** in the env **before** your “real” Phase 1/2 runs so checkpoints match production.

- **During training**:  
  **A, B, C** are all seen while learning. **C** is pure reward shaping; **A/B** change the transition function. Prefer **A/B in env** if production will always use the same rule.

- **After training**:  
  **D** only — wrap `predict`. Good for **prototypes** and A/B tests. If the wrapper stays in prod, consider **retraining with A/B** so the policy isn’t fighting the filter.

---

## Recommendations by goal

1. **Fast sanity check (no retrain):** try **D** (e.g. hold raw action 2–3 steps or 2-of-3 vote). Measure win rate / margin error vs unwrapped policy.

2. **Production must match sim:** implement **A or B** in `PongEnv` (or a thin `gym.Wrapper`), then **retrain** Phase 1 → Phase 2. Cleaner to train Phase 1 with the **final** env than to fine-tune from a policy trained on a different MDP.

3. **Phase 2 margin + smoothness:** add **C** lightly or combine **A** with small **C**; watch for reward hacking (smooth but losing on purpose).

4. **“Looks human” for audience:** combine **E** (`pong_rl_spec.md` cognitive difficulty) with **A/B** or mild **D** — invisible difficulty alone doesn’t guarantee smooth **motor** output.

---

## Current implementation (Option D, enhanced)

`envs/smooth_action_wrapper.py` implements **post-hoc inference-only** smoothing with three layers:

| Lever | Config constant | Default | Effect |
|-------|----------------|---------|--------|
| **Base hold** | `ACTION_HOLD_STEPS` | 3 | Every action must be held this many frames before a different one is accepted. |
| **Reversal penalty** | `ACTION_REVERSAL_HOLD_STEPS` | 8 (~133 ms at 60 FPS) | Switching between up (1) and down (2) requires extra hold time — humans can't physically reverse direction as fast as they can stop/start. |
| **Random jitter** | `ACTION_HOLD_JITTER` | 3 | Each hold target gets 0 to `jitter` random extra frames so direction changes aren't metronomic. |

Two forms:
- **`SmoothActionWrapper`** — standard `gym.Wrapper`, used by `eval_agent.py` when `--hold-steps > 1`.
- **`ActionSmoother`** — standalone callable, used in `play_human_vs_bot.py` for the human paddle when `--hold-steps > 1`.

### Limitation: inference-only gap

The agent trains without constraints and may learn strategies that rely on instant reversals. The wrapper then blocks those at play time, degrading performance. For stronger results, consider **training with the constraint** (Option A/B) so the policy learns strategies compatible with human-like motor limits, then optionally applying a slightly stricter version at inference.

---

## What changes require retraining?

- **Env dynamics or decision frequency** consistent across training → **yes**, new MDP (retrain or fine-tune from a compatible checkpoint).
- **Inference-only wrapper** → **no**, but validate behavior.
- **Reward-only change (C)** → **yes** for that training phase (new objective).

---

## Prompt template for Claude / Cursor

```
Project: Pong RL, Gymnasium PongEnv, SB3 PPO, discrete {0,1,2}.

Goal: Reduce paddle up/down twitchiness so motion looks more human.

Constraints:
- [Choose one] Production will use: (action repeat k=__) / (sticky p=__) / (post-hoc wrapper only) / (reward penalty λ=__).
- Training scripts: training/train_phase1.py, training/train_phase2.py must keep working.
- If env changes: default k=1 or sticky=off so behavior matches current checkpoints unless explicitly enabled.

Tasks:
1. Implement the chosen approach with minimal surface area (env vs wrapper vs train script).
2. Add CLI flags or config constants; document defaults.
3. Note whether existing .zip checkpoints are still valid or user must retrain.

Deliver: short summary + files changed.
```

---

## Related spec note

`pong_rl_spec.md` **Phase 3** discusses **believable** losses (anti-tanking), not necessarily **smooth motor**. Smoothing is complementary: Phase 3 = don’t look like you’re throwing; smoothing = don’t look like a vibrating servo.
