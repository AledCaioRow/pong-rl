"""
Remove generated logs / eval outputs / optional model zips. Run from repo root:

    py training/clean_artifacts.py --logs --eval-results --dry-run
    py training/clean_artifacts.py --logs --yes

TensorBoard event files under training/logs/ are safe to delete anytime; a new run
recreates subfolders (e.g. PPO_2).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _rm_tree(path: Path, dry_run: bool) -> int:
    if not path.exists():
        return 0
    n = sum(1 for _ in path.rglob("*") if _.is_file())
    if dry_run:
        print(f"  [dry-run] would remove {n} files under {path}")
        return n
    shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"  removed {n} files, recreated empty {path}")
    return n


def _rm_glob(pattern: str, dry_run: bool) -> int:
    files = list(ROOT.glob(pattern))
    if dry_run:
        for f in files:
            print(f"  [dry-run] would delete {f}")
        return len(files)
    for f in files:
        f.unlink()
        print(f"  deleted {f}")
    return len(files)


def main() -> int:
    p = argparse.ArgumentParser(description="Clean TensorBoard logs, eval JSON, or model zips.")
    p.add_argument("--logs", action="store_true", help="Delete contents of training/logs/")
    p.add_argument(
        "--eval-results",
        action="store_true",
        help="Delete contents of evaluation/results/",
    )
    p.add_argument(
        "--models",
        action="store_true",
        help="Delete models/*.zip (keep backups elsewhere first).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Perform deletions (otherwise dry-run only).",
    )
    args = p.parse_args()

    if not any((args.logs, args.eval_results, args.models)):
        p.print_help()
        print("\nSelect at least one of --logs, --eval-results, --models.", file=sys.stderr)
        return 2

    dry_run = not args.yes
    if dry_run:
        print("Dry run (pass --yes to delete):\n")
    else:
        print("Deleting:\n")

    if args.logs:
        print("training/logs/")
        log_root = ROOT / "training" / "logs"
        if log_root.exists():
            for child in log_root.iterdir():
                if child.is_dir():
                    _rm_tree(child, dry_run)
                elif child.is_file():
                    if dry_run:
                        print(f"  [dry-run] would delete {child.name}")
                    else:
                        child.unlink()
                        print(f"  deleted {child.name}")

    if args.eval_results:
        print("evaluation/results/")
        _rm_tree(ROOT / "evaluation" / "results", dry_run)

    if args.models:
        print("models/*.zip")
        _rm_glob("models/*.zip", dry_run)

    if dry_run:
        print("\nNo files were deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
