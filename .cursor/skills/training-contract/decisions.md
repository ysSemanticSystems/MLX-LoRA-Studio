# Settled training decisions

Do not reopen these unless the user does.

## Modes

| Mode | Reference | Judge | QAT | Long context | Dataset shape |
| --- | --- | --- | --- | --- | --- |
| sft | no | no | yes | yes | prompt/completion or chat |
| dpo | yes | no | yes | yes | prompt, chosen, rejected |
| dsla | only when loss is dpo | no | yes | forced off | same as its DPO, ORPO, or CPO loss |
| ftpo | yes | no | no | no | Antidoom preference rows |
| cpo | no | no | no | yes | preference, no separate reference path |
| orpo | no | no | yes | yes | chosen + rejected, prompt not required |
| grpo | yes | no | no | no | prompt + answer, reward functions |
| klpo | no | no | forced off | no | same family as GRPO, reward functions |
| online_dpo, xpo, rlhf_reinforce, ppo | yes | required | no | no | a prompt field; judge scores samples |

- DSLA losses: `dpo`, `orpo`, `cpo`. Latent variants: `similarity`, `direction`, `both`.
- KLPO routes: `token`, `sequence`. Estimators: `mc`, `topk`, `binary`, `full`.
- `micro_batch_size` is written only for judge modes and only when greater than zero.
- SFT losses: `nll`, `chunked_nll`, `dft`.
- Default datasets are `TrainMode.defaultDataset`. A wrong default fails in `create_dataset()` with `Unsupported data format`.

## Spec

One JSON object from `TrainingConfig.runSpecData()`, passed as `training_runner.py --spec`. Empty optional strings are omitted. `epochs > 0` clears `iters`, and the reverse.

VLM runs set `model_family` to `vision_language`, train the text tower, and export with `vlm_model`. Fuse dequantize stays off when QAT is on.

## Quantization

Current as of October 2026. Replace this section when the user or a newer MLX / mlx-lm release changes the pin or the mode list. Until then, keep the GUI, the spec, and the runner on the same set.

Load-time choices are None, 4-bit, 6-bit, 8-bit, MXFP4, and NVFP4. The spec writes exactly one `load_in_*` boolean. Swift encode, Swift decode, `_quantization_config`, the memory estimate, and the Hugging Face card label name that same set.

NVFP4 currently means `{bits: 4, group_size: 16, mode: nvfp4}` and quantizes a full-precision model at load. A checkpoint that is already quantized stays on None so mlx-lm keeps the published scales, including the NVFP4 per-tensor global scale.

The installer currently uses `mlx>=0.32.3` and mlx-lm git `ba8a08fd`, because PyPI mlx-lm 0.32.0 does not pass that global scale for MoEs. 1-bit affine is not in a released wheel yet. A later pin or a new mode is a normal change: update the code and this section in the same edit.

## Memory

ResourceGuard is part of the runner, not a log line. It releases the MLX cache and raises when peak memory crosses the configured fraction (default 0.78).
