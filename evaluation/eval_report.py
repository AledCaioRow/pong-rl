"""
Aggregate evaluation JSON from eval_agent.py; print or compare runs.

    py evaluation/eval_report.py --input evaluation/results/p1.json
    py evaluation/eval_report.py --compare a.json b.json --format md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flat_rallies(games: list[dict[str, Any]]) -> list[int]:
    out: list[int] = []
    for g in games:
        out.extend(g.get("rallies") or [])
    return out


def aggregate(games: list[dict[str, Any]], phase: int) -> dict[str, Any]:
    n = len(games)
    if n == 0:
        return {"n_games": 0}

    wins = sum(1 for g in games if g.get("win"))
    agent_scores = [int(g["agent_score"]) for g in games]
    opp_scores = [int(g["opponent_score"]) for g in games]
    margins = [int(g["margin"]) for g in games]

    out: dict[str, Any] = {
        "n_games": n,
        "win_rate_pct": 100.0 * wins / n,
        "mean_agent_score": statistics.mean(agent_scores),
        "mean_opponent_score": statistics.mean(opp_scores),
        "mean_margin": statistics.mean(margins),
        "std_margin": statistics.stdev(margins) if n > 1 else 0.0,
        "mean_episode_steps": statistics.mean(float(g["episode_duration_steps"]) for g in games),
        "mean_points_per_game": statistics.mean(float(g["total_points"]) for g in games),
    }

    all_r = _flat_rallies(games)
    if all_r:
        out["mean_rally_length"] = statistics.mean(all_r)
        out["std_rally_length"] = statistics.stdev(all_r) if len(all_r) > 1 else 0.0
        out["median_rally_length"] = statistics.median(all_r)
        out["max_rally_length"] = max(all_r)
        short = sum(1 for x in all_r if x <= 5)
        med = sum(1 for x in all_r if 6 <= x <= 15)
        long = sum(1 for x in all_r if x >= 16)
        out["rally_bin_short_le5"] = short
        out["rally_bin_med_6_15"] = med
        out["rally_bin_long_ge16"] = long
    else:
        out["mean_rally_length"] = 0.0
        out["std_rally_length"] = 0.0
        out["median_rally_length"] = 0.0
        out["max_rally_length"] = 0
        out["rally_bin_short_le5"] = 0
        out["rally_bin_med_6_15"] = 0
        out["rally_bin_long_ge16"] = 0

    rates = [float(g["action_change_rate"]) for g in games]
    changes = [int(g["action_changes"]) for g in games]
    out["mean_action_change_rate"] = statistics.mean(rates)
    out["mean_action_changes_per_game"] = statistics.mean(changes)

    if phase == 2:
        errs = [g["margin_error"] for g in games if g.get("margin_error") is not None]
        targets = [g["target_margin"] for g in games if g.get("target_margin") is not None]
        if errs:
            out["mean_target_margin"] = statistics.mean(float(t) for t in targets) if targets else 0.0
            out["mean_actual_margin"] = statistics.mean(margins)
            out["mean_margin_error"] = statistics.mean(float(e) for e in errs)
            out["std_margin_error"] = statistics.stdev(errs) if len(errs) > 1 else 0.0
            out["pct_margin_error_le_2"] = 100.0 * sum(1 for e in errs if e <= 2) / len(errs)
            out["pct_margin_error_le_5"] = 100.0 * sum(1 for e in errs if e <= 5) / len(errs)

    return out


def print_table(title: str, agg: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    for k in sorted(agg.keys()):
        v = agg[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def format_md(title: str, agg: dict[str, Any]) -> str:
    lines = [f"## {title}", "", "| Metric | Value |", "|--------|-------|"]
    for k in sorted(agg.keys()):
        v = agg[k]
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.append("")
    return "\n".join(lines)


def print_compare(
    label_a: str, agg_a: dict[str, Any], label_b: str, agg_b: dict[str, Any]
) -> None:
    keys = sorted(set(agg_a.keys()) | set(agg_b.keys()))
    print(f"\n{'':20} | {label_a:>14} | {label_b:>14} | {'delta':>10}")
    print("-" * 70)
    for k in keys:
        va, vb = agg_a.get(k), agg_b.get(k)
        if isinstance(va, float) and isinstance(vb, float):
            d = vb - va
            print(f"{k:20} | {va:14.4f} | {vb:14.4f} | {d:+10.4f}")
        else:
            print(f"{k:20} | {str(va):>14} | {str(vb):>14} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval JSON from eval_agent.py.")
    parser.add_argument("--input", type=str, default=None, help="Single eval JSON file.")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="Two eval JSON files (Run A vs Run B).",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "md"),
        default="table",
    )
    args = parser.parse_args()

    if args.compare:
        pa, pb = Path(args.compare[0]), Path(args.compare[1])
        da = load_payload(pa)
        db = load_payload(pb)
        phase_a = int(da["meta"].get("phase", 1))
        phase_b = int(db["meta"].get("phase", 1))
        agg_a = aggregate(da["games"], phase_a)
        agg_b = aggregate(db["games"], phase_b)
        if args.format == "json":
            print(
                json.dumps(
                    {"run_a": agg_a, "run_b": agg_b, "labels": [pa.name, pb.name]},
                    indent=2,
                )
            )
        elif args.format == "md":
            print(format_md(f"Run A — {pa.name}", agg_a))
            print(format_md(f"Run B — {pb.name}", agg_b))
        else:
            print_compare(pa.name, agg_a, pb.name, agg_b)
        return

    if not args.input:
        print("Provide --input FILE or --compare A B", file=sys.stderr)
        sys.exit(1)

    p = Path(args.input)
    data = load_payload(p)
    phase = int(data["meta"].get("phase", 1))
    agg = aggregate(data["games"], phase)

    if args.format == "json":
        print(json.dumps(agg, indent=2))
    elif args.format == "md":
        print(format_md(p.name, agg))
    else:
        print_table(p.name, agg)


if __name__ == "__main__":
    main()
