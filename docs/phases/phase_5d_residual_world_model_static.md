# Phase 5D: copy-preserving residual world model

Date: 2026-08-29  
Status: `STATIC_PASS`

## New hypothesis

Phase 5C showed that full-frame reconstruction dominates the learned models and loses badly to copy-frame. Phase 5D
does not extend those checkpoints. It introduces a new parameterization whose output is
`clamp(image_t + bounded_residual, 0, 1)`, with an exactly zero-initialized residual head. Initial predictions must
therefore equal copy-frame bit-for-bit; learning can only add evidence-supported corrections.

The original 40 training episodes are deterministically nested into 32 train and 8 development episodes. The
original 10 validation episodes were already exposed through Phase 5A/5C aggregate evaluation, so they are labeled
prior-exposed confirmation—not an untouched final. Phase 5D selection, if later authorized, must use development
only. Confirmation and formal training remain forbidden.

## Static result

All 12 checks PASSed. Both ~1.29M-parameter models produced `[2,4,3,128,128]` rollouts exactly equal to copy-frame at
initialization (`max_abs=0`). Backward values were finite; because the residual head is zero initialized, only its
two parameter tensors receive nonzero gradient on the first backward, as intended. The conditioned latent trajectory
changes under recorded versus zero actions, confirming a connected action path.

Internal/host CPU wall was 0.119/1.15 s; max RSS 730,080 KiB, zero swap/network and no CUDA/training/environment
work. Decision: `STATIC_PASS_DEVELOPMENT_SMOKE_IMPLEMENTATION_ALLOWED`. A new development-only smoke may be
implemented, but training and prior-exposed confirmation execution remain unauthorized until separately frozen.
Canonical result: `results/phase5d_residual_world_model_static_v1.yaml`.
