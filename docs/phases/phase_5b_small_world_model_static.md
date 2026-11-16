# Phase 5B: small world-model architecture static gate

Date: 2026-08-28  
Status: `STATIC_PASS`

## Architecture

A four-stage convolutional encoder maps 128×128 RGB to a 128D latent. A matched decoder reconstructs RGB. Two
open-loop dynamics models share encoder/decoder architecture: action-free residual latent dynamics and an
action-conditioned variant with a 7D action embedding. Both unroll the same H=8 action window before decoding all
future frames.

The static gate uses real Phase 5A HDF5 frames on CPU and requires finite reconstructions, H=8 predictions, losses
and gradients; correct shapes/ranges; a nonzero action-path effect; fewer than two million parameters; and within
5% parameter-count matching between learned baselines. Random-weight loss is only an implementation check.

## Frozen future comparison

Any training gate must use the Phase 5A 40/10 episode split and identical H=1/2/4/8 validation windows. The
action-conditioned model must be compared against both exact copy-frame and learned action-free baselines. Primary
selection is validation H4 pixel MAE, with all horizons and MAE/MSE/PSNR reported. Candidate-action ranking and
environment execution remain forbidden until a separate prediction-quality gate.

## Static result

All 16 checks PASSed on real episode-0 frames. Both models produced finite `[2,8,3,128,128]` RGB rollouts in
`[0,1]` and finite backward gradients. Action-free has 1,284,355 parameters; action-conditioned has 1,292,803,
only 0.66% more. The conditioned output changed under the same image with recorded versus zero actions, confirming
that the action path is connected. Random-weight combined losses were 0.3358/0.3404 and are not performance metrics.

Internal/host CPU wall was 0.90/2.37 s; peak RSS was 968,836 KiB, with zero swap/network and no CUDA initialization.
There were zero training/optimizer steps and no environment/model-policy predictions.

Decision: `ARCHITECTURE_PASS_TRAINING_SMOKE_IMPLEMENTATION_ALLOWED`. The next gate may implement a short,
new-output training smoke with both learned baselines under identical batches, but formal training remains
unauthorized. Canonical result: `results/phase5b_small_world_model_static_v1.yaml`.
