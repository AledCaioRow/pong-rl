# RL Pong Agent — Technical Specification

## Overview

This is not a standard "beat the human" Pong agent. The agent's objective is to **win by a specific margin** — a target set dynamically before each game by the Assessor and Strategist. Winning 30–0 is a failure. Losing is a failure. Winning 30–18 when the target margin was 12 is success.

This makes the training problem significantly harder and more interesting than vanilla Pong RL.

---

## Environment Design

### Why Custom (Not Atari)

The Atari Pong environment from Gymnasium has fixed physics, pixel-based observations, and no hooks for difficulty modulation. You need a custom environment because:

- You need clean state observations (positions, velocities) not raw pixels — pixel-based training is orders of magnitude slower and unnecessary here.
- You need to emit events (point scored, rally length) for the Commentator and logging.
- You need to modulate the agent's cognitive behaviour per-point based on the Strategist's difficulty schedule.

### Environment Specification

```
Observation Space (Box):
  - ball_x:        [0, 1]  normalised
  - ball_y:        [0, 1]  normalised
  - ball_vx:       [-1, 1] normalised
  - ball_vy:       [-1, 1] normalised
  - agent_paddle_y:[0, 1]  normalised
  - human_paddle_y:[0, 1]  normalised
  - score_agent:   [0, 30] integer
  - score_human:   [0, 30] integer
  - target_margin: [0, 30] integer (from Strategist)
  - current_margin:[−30, 30] agent_score − human_score

Action Space (Discrete, 3):
  0 = stay
  1 = move up
  2 = move down

Episode termination:
  Either player reaches 30 points.
```

---

## Invisible Difficulty Levers

The core design principle: **never touch what the player can see, only touch what they can't.** The agent has the same paddle speed, the same paddle size, and the same ball speed at every difficulty level. What changes is the agent's "cognition" — how it perceives and decides, not how it moves.

### What the player CAN see (never change these):
- Paddle speed
- Paddle size
- Ball speed
- Ball physics / angles

### What the player CANNOT see (modulate these):

| Lever                | Easy (for human)                          | Hard (for human)                     | Why it's invisible                                    |
|----------------------|-------------------------------------------|--------------------------------------|-------------------------------------------------------|
| **Reaction delay**   | Agent waits 2–5 frames before responding to a ball direction change | Agent responds immediately (0 frames) | Humans have reaction delay too — this looks completely natural |
| **Prediction noise** | Agent targets the ball's future position ± a random offset (e.g. ±15% of court height) | Agent targets exact predicted position | Looks like the agent "misread" the ball, which humans do constantly |
| **Decision wobble**  | Agent occasionally hesitates for a few frames before committing to a direction | Agent commits instantly | Mimics human indecision — "should I go up or down?" |
| **Recovery priority**| After hitting the ball, agent drifts slowly back to centre | Agent snaps to optimal position immediately | Looks like natural relaxation between shots |

The Strategist outputs a `strength` value per point (0.0–1.0), which interpolates across these cognitive parameters. At `strength=0.3`, the agent plays like a distracted human. At `strength=1.0`, it plays at full machine precision.

### Implementation sketch

```python
class CognitiveDifficulty:
    def __init__(self, strength: float):
        # Interpolate cognitive parameters from strength
        self.reaction_delay = int(lerp(5, 0, strength))        # frames
        self.prediction_noise = lerp(0.15, 0.0, strength)      # fraction of court height
        self.wobble_probability = lerp(0.3, 0.0, strength)     # chance per frame
        self.wobble_duration = int(lerp(8, 0, strength))        # frames
        self.recovery_speed = lerp(0.3, 1.0, strength)         # fraction of max speed

    def apply_to_observation(self, true_ball_y_predicted):
        """Add noise to what the agent 'sees' as the target position."""
        noise = random.gauss(0, self.prediction_noise)
        return true_ball_y_predicted + noise

    def should_wobble(self):
        """Randomly trigger hesitation."""
        return random.random() < self.wobble_probability
```

### Why this approach works

A human watching the game sees an agent that sometimes misjudges the ball, sometimes hesitates, sometimes reacts a fraction late — all things human players do. The difficulty feels like the AI is having a good or bad day, not like someone turned a dial. Compare this to slowing the paddle down, which immediately feels artificial and breaks immersion.

---

## Training Pipeline

### Phase 1: Competence (Standard Pong)

**Goal:** Train an agent that can reliably win Pong.

**Setup:**
- Algorithm: PPO (Proximal Policy Optimisation) via Stable-Baselines3.
- Opponent: rule-based bot that tracks ball y-position with noise.
- Reward: `+1` for scoring, `−1` for conceding.
- Training: ~1–2M timesteps (expect a few hours on CPU).

**Why PPO:** It's stable, well-documented, and handles continuous observation spaces well. DQN is an alternative but PPO is the standard recommendation for Stable-Baselines3 beginners.

```python
from stable_baselines3 import PPO
from pong_env import PongEnv

env = PongEnv(opponent="rule_based")
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=2_000_000)
model.save("pong_competent")
```

**Validation:** The competent agent should beat the rule-based bot >90% of the time.

---

### Phase 2: Margin Targeting (The Novel Bit)

**Goal:** Fine-tune the competent agent to win by a specific margin, not to maximise score.

**Custom reward function:**

```python
def calculate_reward(self, scorer, current_margin, target_margin):
    margin_diff = current_margin - target_margin

    if scorer == "agent":
        if current_margin <= target_margin:
            # Behind schedule or on target — scoring is good
            reward = +1.0
        else:
            # Ahead of schedule — scoring is bad (running up the score)
            reward = -0.5
    elif scorer == "human":
        if current_margin > target_margin:
            # Ahead of schedule — conceding is fine (easing off)
            reward = +0.3
        else:
            # Behind schedule — conceding is bad
            reward = -1.0

    # Rally bonus: reward longer rallies regardless of outcome
    rally_bonus = min(self.rally_length / 50.0, 0.5)
    reward += rally_bonus

    # Proximity bonus at game end: reward final margins close to target
    if self.game_over:
        final_margin = self.agent_score - self.human_score
        margin_error = abs(final_margin - target_margin)
        proximity_reward = max(0, 5.0 - margin_error)  # up to +5 for exact margin
        reward += proximity_reward

    return reward
```

**Key design decisions:**

- The rally bonus prevents the agent from learning to score instantly (which is easy but boring). Long rallies = engaging gameplay.
- The proximity reward at game end is the strongest signal — it anchors the whole training process around the target margin.
- Conceding when ahead is rewarded (not just "not punished"), which teaches the agent to actively ease off rather than accidentally losing focus.

**Training:**
- Start from the Phase 1 checkpoint (don't train from scratch).
- Randomise `target_margin` each episode: sample uniformly from [2, 20].
- Opponent: mix of rule-based bots at different skill levels AND the Phase 1 agent playing against itself.
- Training: ~3–5M additional timesteps.

```python
model = PPO.load("pong_competent", env=margin_env)
model.learn(total_timesteps=5_000_000)
model.save("pong_margin_targeting")
```

---

### Phase 3: Believable Play (Optional but Recommended)

**Goal:** The agent should lose points in a way that looks natural, not like obvious tanking.

**Problem:** A margin-targeting agent might learn to "throw" points by freezing or moving away from the ball. Humans notice this immediately and it breaks immersion.

**Approach:** Add an adversarial discriminator (optional, advanced).

- Record trajectories of the Phase 1 agent playing normally.
- Record trajectories of the Phase 2 agent intentionally losing.
- Train a small classifier to distinguish "real effort" from "throwing."
- Add a penalty to the reward function when the discriminator detects throwing.

Simpler alternative: add a "movement penalty" — penalise the agent for moving less than a threshold amount during a rally. This forces it to look active even when trying to lose.

```python
# Simple anti-tanking penalty
if self.agent_movement_this_rally < MINIMUM_MOVEMENT_THRESHOLD:
    reward -= 0.3
```

**Note:** The invisible difficulty levers already help significantly here — the agent doesn't need to "choose" to lose, the cognitive noise causes natural-looking misses. Phase 3 is mainly needed for cases where the reward function creates degenerate tanking behaviour on top of the cognitive levers.

---

## Integration with Strategist

The Strategist provides a list of difficulty values, one per point:

```python
# Example: 60-point game (first to 30), target margin of 10
difficulty_schedule = strategist.generate_plan(target_margin=10, game_length=60)
# Returns something like: [0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.8, 0.9, ...]
```

Each value is fed into the `CognitiveDifficulty` class, which modulates the agent's perception and decision-making for that point. The RL agent's policy is unchanged — it always tries its best given what it "sees." The difficulty comes from how noisy and delayed its perception is, not from the agent deliberately playing worse.

This is an important distinction: **the agent is always playing optimally given its inputs. The Strategist controls the quality of those inputs.**

---

## Evaluation Metrics

| Metric                    | Target              | How to Measure                          |
|---------------------------|---------------------|-----------------------------------------|
| Margin accuracy           | ±3 of target        | `abs(actual_margin - target_margin)`     |
| Rally length (mean)       | >10 hits per rally  | Count ball contacts per point            |
| Tanking detection rate    | <10%                | Human evaluation or discriminator score  |
| Win rate vs rule-based    | >85%                | Standard eval games                      |
| Game "excitement"         | >3 lead changes     | Count score-lead reversals per game      |

---

## Folder Structure

```
pong-rl/
├── envs/
│   ├── pong_env.py              # Base Gymnasium environment
│   ├── margin_env.py            # Wrapper adding margin-targeting reward
│   └── cognitive_difficulty.py  # Invisible difficulty lever system
├── training/
│   ├── train_phase1.py          # Competence training
│   ├── train_phase2.py          # Margin targeting fine-tune
│   └── train_phase3.py          # Believable play (optional)
├── models/
│   ├── pong_competent.zip       # Phase 1 checkpoint
│   └── pong_margin.zip          # Phase 2 checkpoint
├── evaluation/
│   ├── eval_margin.py           # Test margin accuracy across target values
│   ├── eval_excitement.py       # Measure rally lengths, lead changes
│   └── eval_tanking.py          # Detect obvious throwing behaviour
├── frontend/
│   └── pong_renderer.py         # Pygame or browser-based rendering
└── config.py                    # Hyperparameters, cognitive lever ranges
```

---

## Dependencies

```
gymnasium>=0.29
stable-baselines3>=2.3
torch>=2.0
numpy
pygame          # for local rendering / debugging
```

---

## Estimated Timeline

| Phase                        | Time Estimate |
|------------------------------|---------------|
| Custom Gymnasium env         | 2–3 days      |
| Cognitive difficulty system  | 1–2 days      |
| Phase 1 training + validation| 2–3 days      |
| Phase 2 reward design + training | 1 week   |
| Phase 3 believable play      | 3–4 days      |
| Strategist integration       | 1–2 days      |
| Evaluation suite             | 2–3 days      |
| **Total**                    | **~3–4 weeks**|

---

## Known Risks

1. **Reward hacking:** The agent might find degenerate strategies to hit the target margin (e.g. alternating between scoring and tanking in a fixed pattern). Mitigation: the rally bonus and anti-tanking penalty, plus randomised opponents.

2. **Target margin infeasibility:** If the Assessor says "20 questions needed" and the game is first to 30, the agent needs to win 30–10. Against a decent human, winning by 20 might be impossible without obviously superhuman play. Mitigation: cap the target margin at a reasonable maximum (e.g. 15) and have the Assessor flag "this person may be too obscure for the system."

3. **Sim-to-real gap:** The agent trains against bots, then plays against humans. Human behaviour is less predictable. Mitigation: include a diverse set of bot opponents (aggressive, defensive, random, tracking) and consider a short online fine-tuning phase with real human data.

4. **Cognitive lever calibration:** The mapping from `strength` value to cognitive parameters needs careful tuning — too much prediction noise at low strength might make the agent look drunk rather than human. Mitigation: playtest extensively at each strength level and adjust the interpolation curves.
