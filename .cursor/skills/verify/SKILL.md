---
name: verify
description: Runs the MLX LoRA Studio unit suites and reports what passed, what failed, and what a launch check still would not prove. Use after a spec, runner, or Swift change, before calling a step done, or when the user asks to verify.
---

# Verify

Do not start a training job and do not add features in this pass.

1. From the repo root, run `swift test`.
2. Run `python3 -m unittest Tests/test_training_runner.py`. The Python file stubs MLX, so a missing Apple GPU is not a failure.
3. If a test fails, name the file, the assertion, and the contract it protects (spec round-trip, reference model, judge, QAT flag, ResourceGuard).
4. Use `./script/build_and_run.sh --verify` only when the change is in the bundle, the `swiftc` file list, or app launch. Say that it only checks the process is alive.

Report: passed or failed, the failing test name, and the one next action.
