---
name: training-contract
description: Keeps Swift TrainingConfig, the JSON run spec, and Backend/training_runner.py in lockstep when a training mode, loss, or dataset field changes. Use when editing TrainMode, runSpecData, training_runner.py, algorithm docs, or mlx-lm-lora settings.
---

# Training contract

Read [decisions.md](decisions.md) before changing a mode. The GUI, the spec, and the runner move together. The vendored trainer does not.

## When a field changes

1. Add or adjust the property on `TrainingConfig` and the key in `runSpecData()`. Decode it on the way back in.
2. Teach `Backend/training_runner.py` (`_normalize_spec` and the mode's `run_*`) to read that key. Update the mode set at the top if reference, judge, reward, or micro-batch behavior changed.
3. Mirror a forced flag in both places. DSLA clears long context. KLPO and non-QAT modes clear `qat_enable`. A quantization change updates `runSpecData()`, the decoder, and `_quantization_config` together, then the Quantization section in decisions.md. The mlx and mlx-lm pins there are the current choice; change them when the user or a newer upstream release calls for it.
4. Add a Swift round-trip assertion and, if dispatch changed, a Python pipeline test.
5. Update the matching page in `docs/` when the user-facing behavior changed.

## Check

```bash
swift test
python3 -m unittest Tests/test_training_runner.py
```

A live training run is manual, uses a tiny local fixture, and is not required to call the change done. Do not print dataset rows in the summary.
