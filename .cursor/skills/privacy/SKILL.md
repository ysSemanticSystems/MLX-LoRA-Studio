---
name: privacy
description: Keeps Hugging Face tokens, provider API keys, datasets, adapter weights, and run logs on this Mac. Use when reading a run folder, a JSONL dataset, the Hugging Face cache, or before a commit, upload, or cloud agent.
---

# Privacy

Private material is anything derived from a user's model work: `~/.mlxlorastudio/`, `~/Library/Application Support/MLXLoRAStudio/`, the Hugging Face cache, `train.jsonl` / `valid.jsonl` / `test.jsonl`, `*.safetensors`, `*.gguf`, run logs, and Keychain items for `com.goekdeniz.mlx-lora-studio`.

## Do

- Report paths, row counts, tensor shapes, metric names, and exit codes.
- Read `Sources/`, `Backend/`, `Tests/`, `docs/`, and `script/` freely.
- Run `swift test` and `python3 -m unittest Tests/test_training_runner.py`.

## Do not

- Call `security find-generic-password` or dump the environment to recover `HF_TOKEN` or `SYNTHETIC_*_API_KEY`.
- Paste a dataset row, weight listing, or token into chat, a commit, or a PR.
- `git add` datasets, weights, `.env`, `dist/`, or `vendor/`.
- Upload a dataset or adapter unless the user explicitly asked to publish that artifact. Prefer the Upload tab.

## Debugging a real run

Read `run_spec.json` keys and `metrics.json` names. To open a dataset or weight file, the user must set `MLX_STUDIO_ALLOW_CONTENT=1` for that session; the read hook otherwise denies it. Even then, quote the smallest span that answers the question and do not repeat it in the final message.
