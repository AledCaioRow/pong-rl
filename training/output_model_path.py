"""Resolve where SB3 should save a checkpoint: CLI, quick-run default, or interactive prompt."""

from __future__ import annotations

import os
import re
import sys

_INVALID_NAME = re.compile(r'[\\/:*?"<>|]')


def normalize_save_stem(path: str) -> str:
    """Return a path stem suitable for ``model.save()`` (no ``.zip`` suffix)."""
    p = path.strip().replace("\\", "/")
    if not p:
        raise ValueError("Model path is empty.")
    if p.lower().endswith(".zip"):
        p = p[:-4]
    p = p.rstrip("/")
    parent, base = os.path.split(p)
    if not base or ".." in p.split("/"):
        raise ValueError(f"Invalid model path: {path!r}")
    if _INVALID_NAME.search(base):
        raise ValueError("Name must not contain \\ / : * ? \" < > |")
    if parent in ("", "."):
        p = os.path.join("models", base)
    else:
        p = os.path.normpath(p)
    return p


def resolve_output_model_path(
    *,
    quick: bool,
    cli_provided: bool,
    cli_value: str | None,
    quick_default: str,
) -> str:
    """Non-``--quick`` runs without ``--model-path`` prompt in the terminal (TTY only)."""
    if quick:
        return normalize_save_stem(quick_default)
    if cli_provided:
        if cli_value is None:
            raise ValueError("--model-path requires a value")
        return normalize_save_stem(cli_value)
    if not sys.stdin.isatty():
        print(
            "Error: output checkpoint name was not set. "
            "Pass --model-path for non-interactive runs (e.g. CI).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(
        "\nCheckpoint name — use a new name for each long run so you do not overwrite earlier models.\n"
        "Stored under models/ unless you include a folder (e.g. models/experiments/my_run).\n"
    )
    while True:
        try:
            raw = input("Model name: ").strip()
        except EOFError:
            print("\nError: no model name (stdin closed).", file=sys.stderr)
            raise SystemExit(2) from None
        if not raw:
            print("Please enter a non-empty name.")
            continue
        try:
            return normalize_save_stem(raw)
        except ValueError as exc:
            print(exc)
