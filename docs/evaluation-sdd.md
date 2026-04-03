# SDD — Evaluation Scripts (`evaluation/`)

See the implemented scripts:

- `evaluation/eval_agent.py` — run N games, write JSON (+ stdout if no `--out`)
- `evaluation/eval_report.py` — aggregate one file or `--compare` two

Defaults: `config.EVAL_GAMES`, `config.EVAL_DETERMINISTIC`; results gitignored under `evaluation/results/`.

Full specification and workflow were specified in the design doc used to implement this suite.
