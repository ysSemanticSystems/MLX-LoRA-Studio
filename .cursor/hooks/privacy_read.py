#!/usr/bin/env python3
"""Deny agent reads of datasets, weights, run archives, and secret files.

MLX_STUDIO_ALLOW_CONTENT=1 allows the read. Source, tests, docs, and
this .cursor directory stay readable. Pytest temp files stay readable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DATASET_NAMES = {"train.jsonl", "valid.jsonl", "test.jsonl"}
WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".npz"}
REPO_MARKERS = ("/Sources/", "/Backend/", "/Tests/", "/docs/", "/script/", "/.cursor/")


def private_roots() -> list[Path]:
    home = Path.home()
    return [
        home / ".mlxlorastudio",
        home / "Library" / "Application Support" / "MLXLoRAStudio",
        home / ".cache" / "huggingface",
        home / "Library" / "Caches" / "huggingface",
    ]


def is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    if name in DATASET_NAMES or name == ".env" or name.startswith(".env."):
        return True
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return True
    return False


def under_private_root(path: Path) -> bool:
    for root in private_roots():
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except (OSError, ValueError):
            continue
    return False


def allowed_location(path: Path) -> bool:
    text = str(path)
    if any(marker in text for marker in REPO_MARKERS):
        return True
    if "/pytest-of-" in text or text.startswith("/private/var/folders") or "/T/pytest" in text:
        return True
    return False


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        print(json.dumps({"permission": "allow"}))
        return
    path = Path(str(payload.get("file_path") or ""))
    if os.environ.get("MLX_STUDIO_ALLOW_CONTENT") == "1":
        print(json.dumps({"permission": "allow"}))
        return
    sensitive = is_sensitive(path) or under_private_root(path)
    if not sensitive or allowed_location(path):
        print(json.dumps({"permission": "allow"}))
        return
    print(
        json.dumps(
            {
                "permission": "deny",
                "user_message": (
                    f"Privacy policy blocked reading {path.name}. "
                    "Datasets, weights, and run archives stay on this Mac. "
                    "Set MLX_STUDIO_ALLOW_CONTENT=1 to inspect this file in this session."
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
