#!/usr/bin/env python3
"""Block shell commands that dump secrets, stage private artifacts, or upload them.

Ordinary tests, builds, and localhost calls stay allowed. Uploading a
dataset or weight file is denied regardless of MLX_STUDIO_ALLOW_CONTENT.
"""

from __future__ import annotations

import json
import re
import sys

CONTENT_FILE = re.compile(
    r"train\.jsonl|valid\.jsonl|test\.jsonl|\.safetensors|\.gguf|\.npz|\.mlxlorastudio|(^|[/\s])\.env\b|MLXLoRAStudio/runs",
    re.IGNORECASE,
)
UPLOAD = re.compile(
    r"\b(scp|rsync|curl|wget|huggingface-cli\s+upload|hf\s+upload)\b",
    re.IGNORECASE,
)
LOCAL_ONLY = re.compile(r"127\.0\.0\.1|localhost", re.IGNORECASE)
GIT_ADD = re.compile(r"\bgit\s+add\b", re.IGNORECASE)
GIT_ADD_ALL = re.compile(
    r"\bgit\s+add\b(?:\s+--)?\s+(?:-A|--all|\.)(?:\s|$)",
    re.IGNORECASE,
)
KEYCHAIN = re.compile(
    r"\bsecurity\s+(find-generic-password|dump-keychain|find-internet-password)\b",
    re.IGNORECASE,
)
PRINT_SECRET = re.compile(
    r"\$(?:\{)?(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|SYNTHETIC_[A-Z0-9_]*API_KEY)",
)


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        print(json.dumps({"permission": "allow"}))
        return
    command = str(payload.get("command") or "")
    if KEYCHAIN.search(command) or PRINT_SECRET.search(command):
        _deny("Reading Keychain items or token variables is blocked.")
        return
    if GIT_ADD_ALL.search(command):
        _deny("Blanket git add is blocked. Stage explicit source paths so datasets and weights stay untracked.")
        return
    if GIT_ADD.search(command) and CONTENT_FILE.search(command):
        _deny("git add of datasets, weights, or env files is blocked.")
        return
    if UPLOAD.search(command) and CONTENT_FILE.search(command) and not LOCAL_ONLY.search(command):
        _deny("Uploading datasets or weights is blocked. Publish from the Upload tab when you mean to.")
        return
    print(json.dumps({"permission": "allow"}))


def _deny(message: str) -> None:
    print(
        json.dumps(
            {
                "permission": "deny",
                "user_message": message,
                "agent_message": "Privacy hook denied this command because it would reveal, stage, or upload private training material.",
            }
        )
    )


if __name__ == "__main__":
    main()
