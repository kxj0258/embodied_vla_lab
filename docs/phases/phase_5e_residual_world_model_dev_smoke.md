# Phase 5E — Residual World Model Development Smoke

## Purpose

Phase 5C showed that unconstrained full-frame prediction can optimize while becoming much worse than the copy-frame baseline. Phase 5E tests one new, preregistered mechanism: predict a bounded residual around the current frame, with an exactly zero residual at initialization.

This is a development-only engineering smoke. It is not formal training, checkpoint selection, planning evaluation, reward prediction, or environment evaluation.

## Frozen protocol

- Real LIBERO agent-view RGB and 7D actions; horizon 4.
- The original Phase 5A 40 training episodes are nested into 32 training and 8 development episodes.
- The original 10 validation episodes are marked `prior_exposed_confirmation` because Phase 5C already evaluated them. Phase 5E does not read or evaluate them.
- A deterministic fixture contains 256 training and 128 development windows.
- Residual action-free and residual action-conditioned models receive 100 AdamW updates each.
- Batch order and shared encoder/decoder initialization are identical.
- Both models are evaluated on development H1–H4 before and after training and compared with copy-current-frame.

The conditioned hypothesis earns `DEVELOPMENT_GO_ACTION_CONDITIONING` only when its post-training H4 pixel MAE is lower than both the residual action-free model and copy-current-frame. Any other finite execution is a development NO-GO. A GO still would not authorize confirmation evaluation or formal training automatically.

## Static gate

The CPU-only checker validates fixture integrity, exact split isolation, H4 window safety, paired fairness, the decision rule, fresh output paths, frozen source hashes, and the prohibition on confirmation/formal execution. It imports neither PyTorch nor CUDA and performs zero model predictions, optimizer steps, or environment steps.

Canonical inputs:

- `configs/train/phase5e_residual_world_model_dev_smoke_v1.yaml`
- `outputs/reference/phase5e_residual_world_model_train256_dev128_fixture_v1_seed20260902/manifest.json`
- `scripts/train_phase5e_residual_world_model_smoke.py`

Static v1 exposed a packaging omission before GPU execution: the trainer wrote only its internal JSON and not the declared canonical YAML. No GPU/model work occurred. The omission was fixed without changing the experiment protocol, and static v2 is the execution-authorizing gate. The actual GPU smoke requires explicit host execution. All outputs and both checkpoints must be retained regardless of outcome.

## Execution result

Execution PASSed, while the preregistered hypothesis decision is `DEVELOPMENT_NO_GO_ACTION_CONDITIONING`. H4 MAE
was 1.59125 for copy, 1.59223 for residual action-free, and 1.59303 for residual action-conditioned. Conditioned
was 0.0506% worse than free and 0.1124% worse than copy.

First-10 to last-10 mean loss changed by -3.73% free and -3.67% conditioned. All gradients were finite; pre-clip
ranges were about 0.00116–0.13641. Both step-100 checkpoints include model and optimizer states and CPU-load
completely. Internal/host wall was 2.825/6.23 seconds; peak allocated/reserved VRAM was
133,392,896/159,383,552 bytes, host max RSS 1,814,380 KiB, and swap/network counts were zero.

No confirmation episode, environment step, success/reward measurement, checkpoint selection or planning evaluation
occurred. This mechanism is frozen as a development NO-GO; longer training is not authorized. Canonical summary:
`results/phase5e_residual_world_model_dev_summary_v1_manifest.yaml`.
