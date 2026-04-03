# Questions for Claude: Phase 2 training (Pong RL)

Use this file as **context you paste** plus **questions** when you talk to Claude in another chat. The “Quick facts from this repo” section is grounded in the current codebase so Claude can reason about *this* project, not generic RL.

---

## Copy-paste prompt (edit if you want shorter)

I’m working on the **Pong RL** repo. I need a clear explanation of **Phase 2 training** in plain language.

Please answer using the facts below (and correct me if anything is wrong):

1. **Phase 1 vs Phase 2:** Does Phase 2 **fine-tune on top of** the Phase 1 policy (same network weights, continued PPO), or does it **train a new model from scratch**? What gets saved where?
2. **What is “margin” here?** I might be confusing **win rate** (e.g. 50% vs 60% of games) with something else. In this project, what quantity does the agent try to match?
3. **Who picks the target and when?** How does the environment communicate “try to end around this margin” during an episode? Is it in the observation, the reward only, or both?
4. **Reward intuition:** Walk through one episode conceptually: when the agent scores or concedes, what signal pushes it toward **not** running up the score above the target, and toward **not** falling too far below?
5. **Generalization:** The target changes every episode (sampled range). How does that relate to the policy learning one behavior vs a family of behaviors?

**Repo facts (verify against code if needed):**

- Phase 2 script: `training/train_phase2.py` — uses `PPO.load(...)` from `config.PHASE2_LOAD_PATH` (default `models/pong_competent.zip`), then `model.learn(...)`, saves to `PHASE2_MODEL_PATH` (default `models/pong_margin_targeting`).
- Wrapper: `envs/margin_env.py` — `MarginTargetingWrapper` around `PongEnv`.
- Each reset, if no `target_margin` in options, it samples uniformly: `TARGET_MARGIN_MIN`..`TARGET_MARGIN_MAX` from `config.py` (currently **2..20** integer point margin).
- Rewards mix: margin-aware rewards on each point, rally bonus capped at 0.5, terminal bonus `max(0, 5 - |final_margin - target|)`.
- Observation includes target and current margin (see `envs/pong_env.py` observation layout — target margin is exposed so the policy can condition on it).

Please use small examples (e.g. “target 5, current margin 6 after a point”) if that helps.

---

## Quick facts from this repo (for you, the human)

| Topic | What the code does |
|--------|---------------------|
| New model vs fine-tune | **Fine-tune:** `PPO.load` Phase 1 zip, same algorithm, continued training on the margin-wrapped env. Output is a **different checkpoint file** (`pong_margin_targeting.zip`), not overwriting Phase 1 unless you point `--model-path` there. |
| “50% vs 60%” | **Not win rate.** Phase 2 targets **final score difference** (agent minus opponent), sampled per episode between **2 and 20** points (`TARGET_MARGIN_MIN` / `TARGET_MARGIN_MAX` in `config.py`). |
| Who sets the target | **The wrapper** samples it each `reset` (unless you pass `options={"target_margin": ...}`). It’s stored in the env and appears in the observation so the policy sees it every step. |
| How it trains | PPO still maximizes **discounted return**, but return is now from the **Phase 2 reward** in `MarginTargetingWrapper._compute_reward`, not the sparse Phase 1 scoring reward alone. |

---

## File map for deeper Claude follow-ups

| File | Role |
|------|------|
| `training/train_phase2.py` | Load Phase 1 → vectorized margin env → `learn` → save Phase 2 |
| `envs/margin_env.py` | Target sampling, margin-based rewards, optional opponent noise mix |
| `envs/pong_env.py` | Core Pong MDP; observation includes `target_margin` and `current_margin` |
| `config.py` | `PHASE2_*`, `TARGET_MARGIN_MIN`, `TARGET_MARGIN_MAX` |
| `pong_rl_spec.md` | Written spec (reward design rationale, Phase 2 section) |
| `docs/codebase-overview.md` | Short architecture summary |

---

## Optional follow-up questions (add to your Claude message)

- If I play human vs bot with a Phase 2 checkpoint, how do I set or fix `target_margin` for that session (CLI / `reset` options / play script)?
- What happens if I evaluate a Phase 1 model on the margin-wrapped env without Phase 2 training?
- How sensitive is the policy to the terminal proximity bonus vs per-point rewards?
